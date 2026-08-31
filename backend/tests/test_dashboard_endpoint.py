"""API tests for the composite Dashboard endpoint (Task 13.3, dashboard portion).

Exercises ``GET /api/v1/dashboard/{product_id}`` end-to-end through a FastAPI
``TestClient``. The endpoint composes all five feature services for one product
into a single response, each module contained independently, so these tests
focus on what the endpoint itself owns (Req 8.1, 8.5, 15.1, 14.4):

* a fully-seeded product returns 200 with all six slots (the product plus the
  five feature modules), the discount slot carrying a genuineness score *and*
  its SHAP explanation, cross-platform marking a best deal, shrinkflation
  carrying timeline points, and buy-timing carrying a recommendation and the
  category-level/snapshot disclosure;
* **containment** (Req 15.1, 8.5): when one module's service is made to raise,
  that module's slot degrades to an unavailable state while every other module
  still returns its result and the response stays 200; and
* an unknown product id is a 404 ``PRODUCT_NOT_FOUND`` structured error.

The ``get_db`` dependency is overridden with an in-memory SQLite session
(``StaticPool`` so the ``:memory:`` database is shared across the seed session
and each request-scoped session). The client is entered as
``with TestClient(app) as client:`` so the app lifespan runs and the *real*
trained discount model and its SHAP explainer are loaded - the discount slot's
score and explanation are exercised against the genuine model, not a stub. The
dashboard composes the feature *services* directly (it does not go through the
cached per-feature endpoints), so no Redis is involved. Teardown clears the
dependency override and disposes the engine so no state leaks.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1 import dashboard
from app.db.models import (
    Base,
    CategoryPriceStats,
    CategorySeasonality,
    PackSizeHistory,
    PlatformPrice,
    PriceSnapshot,
    Product,
)
from app.db.session import get_db
from app.main import app
from app.ml.discount_model import FEATURE_NAMES
from app.ml.seasonality import build_category_profile

# Product ids like ``amz_0001`` contain no slashes, so the default ``{product_id}``
# converter captures them cleanly.
_PRODUCT_ID = "amz_0001"
_CATEGORY = "grocery/snacks"
# The seeded (valid) discount pair: reference strictly above the displayed price.
_DISPLAYED_PRICE = 90.0
_REFERENCE_PRICE = 150.0
# The seeded pack variants collapse to three distinct comparison entries.
_EXPECTED_VARIANTS = 3


def _seed(session) -> None:
    """Seed one product with data for every feature module.

    A single product carries: a ``CategoryPriceStats`` row (so the discount
    scored path is available), two ``price_snapshots`` (one with a valid
    reference above the displayed price, one undated with no reference, so the
    "most relevant" selection is genuinely exercised), a three-point
    ``pack_size_history`` (distinct pack sizes for the unit-price and
    shrinkflation modules), two ``platform_prices`` (so cross-platform can mark
    a best deal), and a full twelve-month ``category_seasonality`` profile built
    by :func:`app.ml.seasonality.build_category_profile` (so buy-timing has a
    profile).
    """

    session.add(
        Product(
            id=_PRODUCT_ID,
            name="Crunchy Potato Chips",
            normalized_name="crunchy potato chips",
            brand="SnackCo",
            category=_CATEGORY,
        )
    )
    session.add(
        CategoryPriceStats(
            category=_CATEGORY,
            mean_price=100.0,
            median_price=95.0,
            std_price=30.0,
            p25_price=80.0,
            p75_price=120.0,
            # Discount stats are stored as percentages in [0, 100].
            mean_discount_pct=20.0,
            std_discount_pct=10.0,
            mean_rating=4.0,
            mean_rating_count=600.0,
            sample_size=150,
        )
    )
    # Two snapshots: the dated one with a valid reference must win selection over
    # the more recent one that has no reference.
    session.add_all(
        [
            PriceSnapshot(
                product_id=_PRODUCT_ID,
                platform="Amazon",
                reference_price=_REFERENCE_PRICE,
                displayed_price=_DISPLAYED_PRICE,
                discount_pct=40.0,
                rating=4.2,
                rating_count=850,
                captured_at=date(2024, 3, 1),
                source_dataset="amazon_kaggle",
            ),
            PriceSnapshot(
                product_id=_PRODUCT_ID,
                platform="Flipkart",
                reference_price=None,
                displayed_price=88.0,
                discount_pct=None,
                rating=4.1,
                rating_count=400,
                captured_at=date(2024, 6, 1),
                source_dataset="flipkart_kaggle",
            ),
        ]
    )
    # Three distinct pack sizes at a steady price - classic shrinkflation - which
    # also give the unit-price comparator three variants to rank.
    session.add_all(
        [
            PackSizeHistory(
                product_id=_PRODUCT_ID,
                observed_at=date(2022, 1, 1),
                pack_quantity=200.0,
                pack_unit="g",
                selling_price=50.0,
                unit_price=0.25,
                source_type="cited_public_record",
                source_citation="Company pack-size disclosure, 2022",
            ),
            PackSizeHistory(
                product_id=_PRODUCT_ID,
                observed_at=date(2023, 1, 1),
                pack_quantity=180.0,
                pack_unit="g",
                selling_price=50.0,
                unit_price=50.0 / 180.0,
                source_type="off",
                source_citation=None,
            ),
            PackSizeHistory(
                product_id=_PRODUCT_ID,
                observed_at=date(2024, 1, 1),
                pack_quantity=160.0,
                pack_unit="g",
                selling_price=50.0,
                unit_price=50.0 / 160.0,
                source_type="off",
                source_citation=None,
            ),
        ]
    )
    # Two platforms so a cross-platform comparison (and best deal) is possible.
    session.add_all(
        [
            PlatformPrice(
                product_id=_PRODUCT_ID,
                platform="Amazon",
                price=95.0,
                product_url="https://amazon.example/dp/amz_0001",
                genuineness_score=80,
                captured_at=date(2024, 3, 1),
            ),
            PlatformPrice(
                product_id=_PRODUCT_ID,
                platform="Flipkart",
                price=92.0,
                product_url="https://flipkart.example/p/amz_0001",
                captured_at=date(2024, 3, 1),
            ),
        ]
    )
    # A full twelve-month seasonal profile for the category (deepest window is
    # October / Big Billion Days).
    for point in build_category_profile():
        session.add(CategorySeasonality(category=_CATEGORY, **point))

    session.commit()


@pytest.fixture()
def client():
    """Yield a TestClient with ``get_db`` overridden by a seeded in-memory DB.

    A single ``StaticPool`` engine backs both the seed session and every
    request-scoped session the override yields, so all requests observe the same
    ``:memory:`` database. Entering the client as a context manager runs the app
    lifespan, loading the real discount model and SHAP explainer so the discount
    slot is scored and explained against the genuine model. Teardown clears the
    dependency override and disposes the engine so no state leaks into other
    tests.
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
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


def test_dashboard_returns_all_six_slots_populated(client):
    """A fully-seeded product yields 200 with every slot populated (Req 8.1, 14.4).

    Asserts all six slots are present (the product plus the five feature
    modules), and that each module produced a genuine result: the discount slot
    carries a genuineness score *and* a reconciling SHAP explanation (with the
    internal engineered features dropped), shrinkflation carries its timeline
    points, unit-price ranks the seeded variants with a best value, buy-timing
    returns a recommendation and the disclosure, and cross-platform marks a best
    deal.
    """

    resp = client.get(f"/api/v1/dashboard/{_PRODUCT_ID}")

    assert resp.status_code == 200
    body = resp.json()

    # All six slots present (product + five feature modules) (Req 8.1).
    assert body["product_id"] == _PRODUCT_ID
    for slot in (
        "product",
        "discount",
        "shrinkflation",
        "unit_price",
        "buy_timing",
        "cross_platform",
    ):
        assert slot in body, f"missing dashboard slot: {slot}"

    # product slot: identifying fields.
    product = body["product"]
    assert product["id"] == _PRODUCT_ID
    assert product["name"] == "Crunchy Potato Chips"
    assert product["brand"] == "SnackCo"
    assert product["category"] == _CATEGORY

    # discount slot: scored with a genuineness score + SHAP explanation (Req 2, 3).
    discount = body["discount"]
    score = discount["genuineness_score"]
    assert isinstance(score, int)
    assert 0 <= score <= 100
    assert discount["classification"] in {"genuine", "moderate", "likely_inflated"}
    # The valid reference (150) > displayed (90) snapshot was selected.
    assert discount["displayed_price"] == _DISPLAYED_PRICE
    assert discount["reference_price"] == _REFERENCE_PRICE
    # Explanation is attached and the raw engineered features are dropped (Req 3.1).
    assert "features" not in discount
    assert "explanation" in discount
    explanation = discount["explanation"]
    assert {"base_value", "final_score", "contributions"} <= explanation.keys()
    assert len(explanation["contributions"]) == len(FEATURE_NAMES)

    # shrinkflation slot: chronological timeline points present (Req 4).
    shrinkflation = body["shrinkflation"]
    assert shrinkflation["status"] == "ok"
    assert len(shrinkflation["points"]) == _EXPECTED_VARIANTS
    assert shrinkflation["total_change"] is not None

    # unit_price slot: the three variants ranked with exactly one best value (Req 5).
    unit_price = body["unit_price"]
    assert len(unit_price["comparison"]) == _EXPECTED_VARIANTS
    best = [v for v in unit_price["comparison"] if v.get("best_value")]
    assert len(best) == 1
    # The largest pack at the steady price is the cheapest per gram.
    assert best[0]["quantity_std"] == 200.0

    # buy_timing slot: a recommendation + the category-level/snapshot disclosure (Req 6).
    buy_timing = body["buy_timing"]
    assert buy_timing["available"] is True
    assert buy_timing["recommendation"] in {"buy_now", "wait"}
    assert buy_timing["disclosure"]
    assert "snapshot" in buy_timing["disclosure"].lower()

    # cross_platform slot: two platforms compared, a best deal marked (Req 7).
    cross_platform = body["cross_platform"]
    assert cross_platform["comparison_available"] is True
    assert cross_platform["best_deal_platform"] == "Flipkart"  # cheapest at 92
    marked = [p for p in cross_platform["platforms"] if p.get("best_deal")]
    assert len(marked) == 1
    assert marked[0]["platform"] == "Flipkart"


def test_one_failing_module_is_contained_others_still_return(client, monkeypatch):
    """A raising module is contained; the others still return (Req 15.1, 8.5).

    Monkeypatching this endpoint's reference to the shrinkflation service so it
    raises simulates an unhandled error in one feature module. The shrinkflation
    slot must degrade to an unavailable state while every other module still
    returns its normal result and the overall response stays 200 - the
    reliability guarantee that one broken module never breaks the dashboard.
    """

    def _boom(*args, **kwargs):
        raise RuntimeError("shrinkflation module blew up")

    monkeypatch.setattr(dashboard, "get_shrinkflation_timeline", _boom)

    resp = client.get(f"/api/v1/dashboard/{_PRODUCT_ID}")

    # The whole request still succeeds (Req 15.1).
    assert resp.status_code == 200
    body = resp.json()

    # The failing module is contained into an unavailable slot (Req 8.5).
    shrinkflation = body["shrinkflation"]
    assert shrinkflation["available"] is False
    assert shrinkflation["message"]
    # The contained slot must not leak a normal timeline result.
    assert "points" not in shrinkflation

    # Every other module still returned its normal result (Req 15.1).
    assert isinstance(body["discount"]["genuineness_score"], int)
    assert "explanation" in body["discount"]
    assert len(body["unit_price"]["comparison"]) == _EXPECTED_VARIANTS
    assert body["buy_timing"]["recommendation"] in {"buy_now", "wait"}
    assert body["cross_platform"]["best_deal_platform"] == "Flipkart"
    assert body["product"]["id"] == _PRODUCT_ID


def test_unknown_product_returns_404_product_not_found(client):
    """An unknown product id -> 404 structured PRODUCT_NOT_FOUND error (Req 15.3)."""

    resp = client.get("/api/v1/dashboard/does_not_exist")

    assert resp.status_code == 404
    error = resp.json()["error"]
    assert error["code"] == dashboard.PRODUCT_NOT_FOUND_CODE
    assert error["status"] == 404
    assert error["message"]  # a human-readable reason is present (Req 15.3)
