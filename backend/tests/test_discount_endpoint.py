"""API tests for the True Discount Checker endpoint (Task 8.5).

Exercises ``POST /api/v1/discount-check`` end-to-end through a FastAPI
``TestClient``. The endpoint is a thin, cached boundary over
``app.services.discount_service.check_discount`` (covered by
``tests/test_discount_service.py``) plus the SHAP breakdown from
``app.ml.explainer.explain`` (covered by ``tests/test_model_serving.py``), so
these tests focus on what the endpoint itself owns:

* wiring the ``get_db`` dependency to a real session and returning the service
  result as JSON (Req 14.4);
* attaching the SHAP ``explanation`` to a *scored* result - one plain-language
  contribution per model feature, each with a direction, reconciling in margin
  space (Req 3.1, 3.2, 3.3, 3.5), and dropping the internal ``features`` key so
  the response is the clean shopper-facing set;
* the effective-discount identity in the response (Req 2.4);
* the 422 ``DISCOUNT_NOT_EVALUABLE`` domain error for a missing / too-low
  reference (Req 2.5), rendered by the central handler as the structured payload
  (Req 15.3);
* the 200 ``verification_limited`` body (no score, no explanation) when the
  product's category has no statistics (Req 2.6); and
* the 422 ``VALIDATION_ERROR`` for a malformed body (non-positive displayed
  price, missing category) rejected at the Pydantic boundary (Req 18.1).

The ``get_db`` dependency is overridden with an in-memory SQLite session
(``StaticPool`` so the ``:memory:`` database is shared across the seed session
and each request-scoped session) seeded with a single ``CategoryPriceStats``
row. The client is entered as ``with TestClient(app) as client:`` so the app
lifespan runs and the *real* trained model and its SHAP explainer are loaded -
the scored path and its explanation are exercised against the genuine model, not
a stub.

No live Redis is required. The endpoint wraps its computation in
``app.services.data_service.cached_or_compute``; the ``client`` fixture patches
the underlying cache get/set to a clean miss / no-op so every request computes
fresh and deterministically, independent of any Redis instance.
``test_endpoint_degrades_without_redis`` additionally forces the cache calls to
raise and asserts the endpoint still returns a correct scored 200, proving the
graceful degradation the design requires (Req 11.3, 12.3).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, CategoryPriceStats, Product
from app.db.session import get_db
from app.main import app
from app.ml.discount_model import FEATURE_NAMES
from app.ml.explainer import RECONCILIATION_TOLERANCE, TOWARD_GENUINE, TOWARD_INFLATED
from app.ml.feature_labels import FEATURE_LABELS
from app.services import data_service
from app.services.discount_service import (
    CLASSIFICATION_VERIFICATION_LIMITED,
    NOT_EVALUABLE_CODE,
    SCORE_MAX,
    SCORE_MIN,
)

# A category seeded with price statistics so the scored path is available.
_CATEGORY = "electronics/headphones"
# A category deliberately left without statistics (Req 2.6).
_CATEGORY_NO_STATS = "obscure/uncategorised"

# The valid classification bands the scored path may return (Req 2.2).
_SCORED_CLASSIFICATIONS = {"genuine", "moderate", "likely_inflated"}
# Directions a SHAP contribution may carry (Req 3.2).
_DIRECTIONS = {TOWARD_GENUINE, TOWARD_INFLATED}


def _seed(session) -> None:
    """Seed one category's price statistics (and a matching product row).

    The discount service scores from ``category_price_stats`` keyed by category
    label, so a single stats row is enough to make the scored path available.
    A product row is added for realism (the endpoint accepts a ``product_id``)
    though the service reads only the category statistics.
    """

    session.add(
        Product(
            id="amz_B08XYZ",
            name="Noise-Cancelling Headphones",
            normalized_name="noise cancelling headphones",
            brand="BrandX",
            category=_CATEGORY,
        )
    )
    session.add(
        CategoryPriceStats(
            category=_CATEGORY,
            mean_price=2000.0,
            median_price=1800.0,
            std_price=800.0,
            p25_price=1200.0,
            p75_price=2600.0,
            # Discount stats are stored as percentages in [0, 100].
            mean_discount_pct=40.0,
            std_discount_pct=15.0,
            mean_rating=4.0,
            mean_rating_count=500.0,
            sample_size=100,
        )
    )
    session.commit()


@pytest.fixture()
def client(monkeypatch):
    """Yield a TestClient with ``get_db`` overridden by a seeded in-memory DB.

    A single ``StaticPool`` engine backs both the seed session and every
    request-scoped session the override yields, so all requests observe the same
    ``:memory:`` database. Entering the client as a context manager runs the app
    lifespan, loading the real discount model and SHAP explainer.

    The cache get/set are patched to a clean miss / no-op so each request
    computes fresh regardless of any Redis instance; teardown clears the
    dependency override and disposes the engine so no state leaks.
    """

    # Isolate from Redis: a clean cache miss on read, a skipped write. This is
    # the "empty cache, Redis reachable" first-request path (Req 12.3).
    monkeypatch.setattr(data_service, "cache_get", lambda key: None)
    monkeypatch.setattr(data_service, "cache_set", lambda key, value, ttl: None)

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session_local = sessionmaker(bind=engine, expire_on_commit=False)

    seed_session = testing_session_local()
    try:
        _seed(seed_session)
    finally:
        seed_session.close()

    def _override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_scored_request_returns_score_band_and_reconciling_explanation(client):
    """Valid category + reference > displayed -> 200 scored with SHAP breakdown.

    Covers the scored contract: a genuineness score in [0, 100] (Req 2.1) with a
    classification band (Req 2.2), the effective-discount identity (Req 2.4), the
    clean response shape (no raw ``features``), and an ``explanation`` with one
    plain-language contribution per model feature (Req 3.1, 3.5), each carrying a
    direction (Req 3.2), whose impacts reconcile to the margin final score
    (Req 3.3).
    """
    displayed, reference = 1499.0, 4999.0
    resp = client.post(
        "/api/v1/discount-check",
        json={
            "product_id": "amz_B08XYZ",
            "category": _CATEGORY,
            "displayed_price": displayed,
            "reference_price": reference,
        },
    )

    assert resp.status_code == 200
    body = resp.json()

    # Prices echoed back and the effective-discount identity holds (Req 2.4).
    assert body["displayed_price"] == displayed
    assert body["reference_price"] == reference
    expected_discount = (reference - displayed) / reference * 100.0
    assert body["effective_discount_pct"] == pytest.approx(expected_discount, abs=1e-9)

    # Score in [0, 100] with a valid band (Req 2.1, 2.2).
    score = body["genuineness_score"]
    assert isinstance(score, int)
    assert SCORE_MIN <= score <= SCORE_MAX
    assert body["classification"] in _SCORED_CLASSIFICATIONS

    # The internal engineered features must not leak into the response.
    assert "features" not in body

    # Explanation present with the reconciling structure (Req 3.1, 3.3).
    assert "explanation" in body
    explanation = body["explanation"]
    assert {"base_value", "final_score", "contributions"} <= explanation.keys()

    contributions = explanation["contributions"]
    # One contribution per model feature (Req 3.1).
    assert len(contributions) == len(FEATURE_NAMES)

    raw_names = set(FEATURE_NAMES)
    plain_labels = set(FEATURE_LABELS.values())
    for contribution in contributions:
        assert {"feature", "impact", "direction"} <= contribution.keys()
        label = contribution["feature"]
        # Plain-language label, never a raw model identifier (Req 3.5).
        assert label not in raw_names
        assert label in plain_labels
        assert isinstance(contribution["impact"], (int, float))
        # Direction is present and consistent with the impact's sign (Req 3.2).
        assert contribution["direction"] in _DIRECTIONS
        if contribution["impact"] >= 0:
            assert contribution["direction"] == TOWARD_GENUINE
        else:
            assert contribution["direction"] == TOWARD_INFLATED

    # Exactly one contribution per feature, each label distinct (Req 3.1).
    assert {c["feature"] for c in contributions} == plain_labels

    # base_value + sum(impacts) reconciles to the margin final score (Req 3.3).
    total = explanation["base_value"] + sum(c["impact"] for c in contributions)
    assert total == pytest.approx(explanation["final_score"], abs=RECONCILIATION_TOLERANCE)


def test_missing_reference_is_not_evaluable_422(client):
    """No reference price -> 422 structured DISCOUNT_NOT_EVALUABLE error (Req 2.5)."""
    resp = client.post(
        "/api/v1/discount-check",
        json={"category": _CATEGORY, "displayed_price": 1499.0},
    )

    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == NOT_EVALUABLE_CODE
    assert error["status"] == 422
    assert error["message"]  # a human-readable reason is present (Req 15.3)


def test_reference_not_greater_than_displayed_is_not_evaluable_422(client):
    """Reference <= displayed (both positive) -> 422 DISCOUNT_NOT_EVALUABLE (Req 2.5).

    Both prices pass boundary validation (``gt=0``), so this is the *domain*
    not-evaluable pre-condition raised by the service, not a validation error.
    """
    resp = client.post(
        "/api/v1/discount-check",
        json={
            "category": _CATEGORY,
            "displayed_price": 1499.0,
            "reference_price": 1000.0,
        },
    )

    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == NOT_EVALUABLE_CODE
    assert error["status"] == 422


def test_missing_category_stats_returns_verification_limited(client):
    """Category without stats -> 200 verification_limited, null score, no explanation (Req 2.6)."""
    displayed, reference = 1499.0, 4999.0
    resp = client.post(
        "/api/v1/discount-check",
        json={
            "category": _CATEGORY_NO_STATS,
            "displayed_price": displayed,
            "reference_price": reference,
        },
    )

    assert resp.status_code == 200
    body = resp.json()

    assert body["genuineness_score"] is None
    assert body["classification"] == CLASSIFICATION_VERIFICATION_LIMITED
    # No score means nothing to explain (Req 2.6).
    assert "explanation" not in body
    # Price context is still returned (Req 2.6).
    assert body["displayed_price"] == displayed
    assert body["reference_price"] == reference
    expected_discount = (reference - displayed) / reference * 100.0
    assert body["effective_discount_pct"] == pytest.approx(expected_discount, abs=1e-9)
    assert body["price_context"] == {
        "displayed_price": displayed,
        "reference_price": reference,
    }


def test_non_positive_displayed_price_is_validation_error_422(client):
    """displayed_price <= 0 -> 422 VALIDATION_ERROR at the boundary (Req 18.1)."""
    resp = client.post(
        "/api/v1/discount-check",
        json={
            "category": _CATEGORY,
            "displayed_price": 0,
            "reference_price": 4999.0,
        },
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_missing_category_is_validation_error_422(client):
    """Missing required category -> 422 VALIDATION_ERROR at the boundary (Req 18.1)."""
    resp = client.post(
        "/api/v1/discount-check",
        json={"displayed_price": 1499.0, "reference_price": 4999.0},
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_non_positive_reference_price_is_validation_error_422(client):
    """reference_price <= 0 -> 422 VALIDATION_ERROR (boundary), not not-evaluable (Req 18.1).

    A present but non-positive reference is a malformed value rejected by the
    ``gt=0`` boundary constraint, distinct from the domain not-evaluable case
    (missing / too-low-but-positive reference) which returns
    ``DISCOUNT_NOT_EVALUABLE``.
    """
    resp = client.post(
        "/api/v1/discount-check",
        json={
            "category": _CATEGORY,
            "displayed_price": 1499.0,
            "reference_price": -10.0,
        },
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"


def test_endpoint_degrades_without_redis(client, monkeypatch):
    """A down/absent Redis still yields a correct scored 200 (Req 11.3, 12.3).

    Forcing the cache get/set to raise simulates an unreachable Redis. The
    endpoint wraps its work in :func:`cached_or_compute`, whose helpers degrade
    to a miss on read failure and skip on write failure, so the result is
    computed directly - explanation attached - rather than erroring.
    """

    def _boom(*args, **kwargs):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(data_service, "cache_get", _boom)
    monkeypatch.setattr(data_service, "cache_set", _boom)

    resp = client.post(
        "/api/v1/discount-check",
        json={
            "category": _CATEGORY,
            "displayed_price": 1499.0,
            "reference_price": 4999.0,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert SCORE_MIN <= body["genuineness_score"] <= SCORE_MAX
    assert body["classification"] in _SCORED_CLASSIFICATIONS
    assert len(body["explanation"]["contributions"]) == len(FEATURE_NAMES)
