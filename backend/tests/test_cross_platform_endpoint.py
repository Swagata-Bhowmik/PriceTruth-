"""API tests for the Cross-Platform Aggregator endpoint (Task 10.4).

Exercises ``GET /api/v1/cross-platform/{product_id}`` end-to-end through a
FastAPI ``TestClient``. The endpoint is a thin, cached boundary over
``app.services.cross_platform_service.aggregate_cross_platform`` (already
covered by ``tests/test_cross_platform_service.py`` and the property tests in
``tests/test_cross_platform_property.py``), so these tests focus on what the
endpoint itself owns:

* wiring the ``get_db`` dependency to a real session and returning the service
  result as JSON (Req 7.1, 7.3, 14.4);
* the multi-platform contract - best deal marked at the minimum price, a
  product link on every entry, and a genuineness score present only where the
  listing has one (Req 7.1-7.4);
* the single-platform no-comparison message (Req 7.5); and
* the no-platform-data unavailable message (Req 7.6).

The ``get_db`` dependency is overridden with an in-memory SQLite session
(``StaticPool`` so the ``:memory:`` database is shared across the seed session
and each request-scoped session) seeded with three products: one listed on
several platforms (with the minimum price deliberately *not* inserted first and
only some listings carrying a genuineness score), one on a single platform, and
one product row with no platform prices at all.

No live Redis is required. The endpoint wraps its computation in
``app.services.data_service.cached_or_compute``, whose cache helpers degrade to
a miss on read failure and skip on write failure, so a missing Redis simply
means every request computes directly. ``test_endpoint_works_without_live_redis``
makes that explicit by forcing the cache get/set to raise and asserting the
endpoint still returns a correct 200.

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

from app.db.models import Base, PlatformPrice, Product
from app.db.session import get_db
from app.main import app
from app.services import data_service
from app.services.cross_platform_service import (
    MESSAGE_NO_DATA,
    MESSAGE_SINGLE_PLATFORM,
)

# Product seeded across several platforms; Flipkart holds the true minimum.
_MULTI_PRODUCT_ID = "xplat-multi"
# Product listed on exactly one platform.
_SINGLE_PRODUCT_ID = "xplat-single"
# Product row that exists but has no platform_prices rows.
_NO_DATA_PRODUCT_ID = "xplat-none"

_MIN_PRICE = 1399.0
_BEST_DEAL_PLATFORM = "Flipkart"


def _add_product(session, product_id: str) -> None:
    session.add(
        Product(
            id=product_id,
            name=f"Product {product_id}",
            normalized_name=f"product {product_id}",
            brand="BrandX",
            category="electronics/headphones",
        )
    )


def _add_platform_price(
    session,
    product_id: str,
    platform: str,
    price: float,
    *,
    genuineness_score: int | None = None,
) -> None:
    session.add(
        PlatformPrice(
            product_id=product_id,
            platform=platform,
            price=price,
            product_url=(
                f"https://example.com/{platform.lower().replace(' ', '')}/{product_id}"
            ),
            genuineness_score=genuineness_score,
        )
    )


def _seed(session) -> None:
    """Seed the three cross-platform scenarios into one in-memory database.

    The multi-platform product's rows are inserted so that the minimum price
    (Flipkart, 1399) is *not* first, so a passing best-deal assertion cannot be
    an artefact of insertion order. Only Amazon and Croma carry a genuineness
    score; the cheapest listing (Flipkart) has none, so score presence must
    mirror the data rather than track the best deal.
    """

    # Multi-platform product (Req 7.1-7.4).
    _add_product(session, _MULTI_PRODUCT_ID)
    _add_platform_price(session, _MULTI_PRODUCT_ID, "Amazon", 1499.0, genuineness_score=88)
    _add_platform_price(session, _MULTI_PRODUCT_ID, _BEST_DEAL_PLATFORM, _MIN_PRICE)
    _add_platform_price(session, _MULTI_PRODUCT_ID, "Croma", 1599.0, genuineness_score=75)

    # Single-platform product (Req 7.5).
    _add_product(session, _SINGLE_PRODUCT_ID)
    _add_platform_price(session, _SINGLE_PRODUCT_ID, "Amazon", 799.0, genuineness_score=91)

    # Product with no platform data (Req 7.6).
    _add_product(session, _NO_DATA_PRODUCT_ID)

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


def test_multi_platform_marks_best_deal_and_shapes_entries(client):
    """Several platforms -> 200 with best deal at the minimum price (Req 7.1-7.4).

    Confirms the multi-platform contract: ``status`` 200, one entry per listing,
    a product link on every entry (Req 7.1, 7.3), the cheapest listing marked as
    the single best deal at the true minimum price (Req 7.2), and a genuineness
    score exposed only for listings that actually carry one (Req 7.4).
    """
    resp = client.get(f"/api/v1/cross-platform/{_MULTI_PRODUCT_ID}")

    assert resp.status_code == 200
    body = resp.json()

    assert body["product_id"] == _MULTI_PRODUCT_ID
    assert body["available"] is True
    assert body["comparison_available"] is True
    assert body["best_deal_platform"] == _BEST_DEAL_PLATFORM
    assert body["message"] is None

    entries = body["platforms"]
    assert {e["platform"] for e in entries} == {"Amazon", _BEST_DEAL_PLATFORM, "Croma"}

    # Req 7.3: every entry carries a non-empty link to its product page.
    assert all(e["product_url"] for e in entries)

    # Req 7.2: exactly one entry is the best deal and it holds the minimum price.
    best_entries = [e for e in entries if e.get("best_deal")]
    assert len(best_entries) == 1
    best = best_entries[0]
    assert best["platform"] == _BEST_DEAL_PLATFORM
    assert best["price"] == _MIN_PRICE == min(e["price"] for e in entries)

    # Req 7.4: score present only where the listing has one. Amazon and Croma
    # carry a score; the cheapest listing (Flipkart) does not, so the key must
    # be absent there even though it is the best deal.
    by_platform = {e["platform"]: e for e in entries}
    assert by_platform["Amazon"]["genuineness_score"] == 88
    assert by_platform["Croma"]["genuineness_score"] == 75
    assert "genuineness_score" not in by_platform[_BEST_DEAL_PLATFORM]


def test_single_platform_returns_no_comparison_message(client):
    """One platform -> 200 with the single price and no-comparison message (Req 7.5)."""
    resp = client.get(f"/api/v1/cross-platform/{_SINGLE_PRODUCT_ID}")

    assert resp.status_code == 200
    body = resp.json()

    assert body["available"] is True
    assert body["comparison_available"] is False
    assert body["best_deal_platform"] is None
    assert body["message"] == MESSAGE_SINGLE_PLATFORM

    assert len(body["platforms"]) == 1
    only_entry = body["platforms"][0]
    assert only_entry["platform"] == "Amazon"
    assert only_entry["price"] == 799.0
    assert only_entry["product_url"]
    assert only_entry["genuineness_score"] == 91
    # Nothing is marked as the best deal when there is nothing to compare.
    assert "best_deal" not in only_entry


def test_no_platform_data_returns_unavailable_message(client):
    """No platform data -> 200 with the unavailable message (Req 7.6)."""
    resp = client.get(f"/api/v1/cross-platform/{_NO_DATA_PRODUCT_ID}")

    assert resp.status_code == 200
    body = resp.json()

    assert body["available"] is False
    assert body["comparison_available"] is False
    assert body["best_deal_platform"] is None
    assert body["platforms"] == []
    assert body["message"] == MESSAGE_NO_DATA


def test_endpoint_works_without_live_redis(client, monkeypatch):
    """The endpoint still returns a correct 200 when Redis is unavailable (Req 12.3).

    The cache layer is best-effort: :func:`cached_or_compute` reads through
    ``cache_get_json`` (degrades to a miss on failure) and writes through
    ``cache_set_json`` (skips on failure). Forcing the underlying Redis get/set
    to raise simulates a down/absent Redis and proves the endpoint falls back to
    computing the result directly rather than erroring.
    """

    def _boom(*args, **kwargs):
        raise ConnectionError("redis unavailable")

    # data_service binds cache_get/cache_set at module scope, so patching them
    # here exercises the real cache_get_json/cache_set_json error handling.
    monkeypatch.setattr(data_service, "cache_get", _boom)
    monkeypatch.setattr(data_service, "cache_set", _boom)

    resp = client.get(f"/api/v1/cross-platform/{_MULTI_PRODUCT_ID}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["available"] is True
    assert body["comparison_available"] is True
    assert body["best_deal_platform"] == _BEST_DEAL_PLATFORM
    assert body["platforms"][0]["platform"] == _BEST_DEAL_PLATFORM
    assert body["platforms"][0]["price"] == _MIN_PRICE
