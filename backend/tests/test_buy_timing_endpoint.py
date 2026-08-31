"""API tests for the Buy Timing Signal endpoint (Task 11.4).

Exercises ``GET /api/v1/buy-timing/{category}`` end-to-end through a FastAPI
``TestClient``. The endpoint is a thin boundary over
``app.services.buy_timing_service.recommend_buy_timing`` (already covered by
``tests/test_buy_timing_service.py`` and the property tests in
``tests/test_buy_timing_property.py``), so these tests focus on what the
endpoint itself owns:

* wiring the ``get_db`` dependency to a real session and returning the service
  result as JSON (Req 6.1, 6.4, 14.4),
* letting ``current_month`` default (the endpoint exposes no month parameter),
* correctly handling a category label that contains a slash via the ``:path``
  converter, and
* responding 200 for both the available *and* the no-seasonality "unavailable"
  cases - a category without a profile is a normal result, not an error
  (Req 6.6).

The ``get_db`` dependency is overridden with an in-memory SQLite session
(``StaticPool`` so the ``:memory:`` database is shared across the seed session
and each request-scoped session) seeded with one category's full twelve-month
profile built by :func:`app.ml.seasonality.build_category_profile`, whose
deepest-discount window is October (Big Billion Days).

``raise_server_exceptions=True`` (the TestClient default) means any unhandled
exception escaping the endpoint would fail the request, so a 200 also proves
the endpoint returned cleanly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, CategorySeasonality
from app.db.session import get_db
from app.main import app
from app.ml.seasonality import build_category_profile
from app.services.buy_timing_service import DISCLOSURE, MESSAGE_UNAVAILABLE

# Seeded with a slash so the ``:path`` converter is genuinely exercised.
_SEEDED_CATEGORY = "electronics/headphones"
_UNKNOWN_CATEGORY = "there/is/no/such/category"

# build_category_profile()'s deepest-discount window with no temporal signal.
_BEST_MONTH = 10
_BEST_INDEX = 0.80
_BEST_SALE_EVENT = "Big Billion Days"


def _seed(session) -> None:
    """Seed one category's full twelve-month seasonal profile.

    Uses the same pure builder task 3.4 uses to populate the table, so the
    deepest-discount window is October (index 0.80, Big Billion Days).
    """

    for point in build_category_profile():
        session.add(CategorySeasonality(category=_SEEDED_CATEGORY, **point))
    session.commit()


@pytest.fixture()
def client():
    """Yield a TestClient with ``get_db`` overridden by a seeded in-memory DB.

    A single ``StaticPool`` engine backs both the seed session and every
    request-scoped session the override yields, so all requests observe the
    same ``:memory:`` database. The override is removed and the engine disposed
    on teardown so no state leaks into other tests.
    """

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
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_seeded_category_returns_recommendation_and_disclosure(client):
    """A seeded category yields 200 with a recommendation, window, and disclosure.

    Confirms the success contract (Req 6.1, 6.4, 14.4): ``available`` is True,
    the recommendation is one of buy-now/wait, the deepest-discount window is
    the seeded October (Big Billion Days) window (Req 6.2, 6.5), and every
    result carries the category-level + snapshot-data disclosure (Req 6.4,
    10.1). The slash-containing category also proves the ``:path`` converter
    captures the whole label.
    """
    resp = client.get(f"/api/v1/buy-timing/{_SEEDED_CATEGORY}")

    assert resp.status_code == 200
    body = resp.json()

    assert body["category"] == _SEEDED_CATEGORY
    assert body["available"] is True
    assert body["level"] == "category"
    assert body["recommendation"] in ("buy_now", "wait")
    assert 1 <= body["current_month"] <= 12

    # Req 6.2 / 6.5: the deepest-discount window is the seeded October window.
    window = body["best_window"]
    assert window is not None
    assert window["month"] == _BEST_MONTH
    assert window["month_name"] == "October"
    assert window["relative_price_index"] == _BEST_INDEX
    assert window["sale_event"] == _BEST_SALE_EVENT
    assert window["expected_reduction_pct"] == pytest.approx(20.0)

    # Req 6.4 / 10.1: the category-level + snapshot-data disclosure is present.
    assert body["disclosure"] == DISCLOSURE
    lowered = body["disclosure"].lower()
    assert "category" in lowered
    assert "snapshot" in lowered

    # Correctness Property 17 at the boundary: a "wait" points to a future
    # deepest-discount window.
    if body["recommendation"] == "wait":
        assert window["month"] > body["current_month"]


def test_unknown_category_returns_unavailable_with_disclosure(client):
    """An unknown category yields 200 with an unavailable result (Req 6.6).

    A category with no stored seasonal profile is a normal, non-error result:
    the endpoint returns 200 with ``available=False``, no recommendation, no
    best window, and the standard unavailable message - yet still carries the
    disclosure statement, which holds regardless of data availability.
    """
    resp = client.get(f"/api/v1/buy-timing/{_UNKNOWN_CATEGORY}")

    assert resp.status_code == 200
    body = resp.json()

    assert body["category"] == _UNKNOWN_CATEGORY
    assert body["available"] is False
    assert body["level"] == "category"
    assert body["recommendation"] is None
    assert body["best_window"] is None
    assert body["message"] == MESSAGE_UNAVAILABLE
    # The disclosure is attached even when no recommendation could be produced.
    assert body["disclosure"] == DISCLOSURE
