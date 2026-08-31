"""API tests for the Product Search endpoints (Task 12.4).

Exercises ``GET /api/v1/search`` and ``POST /api/v1/manual-entry`` end-to-end
through a FastAPI ``TestClient``. Both endpoints are thin boundaries over
``app.services.search_service`` (already covered by
``tests/test_search_service.py`` and the property test in
``tests/test_search_property.py``), so these tests focus on what the endpoints
own:

* wiring ``get_db`` and returning the search result as JSON (Req 1.1, 1.2, 14.4),
* the empty-query prompt (Req 1.4) and the no-results + manual-entry message
  (Req 1.5),
* the manual-entry happy path returning a ``SelectedProduct`` (Req 1.6), and
* boundary + domain validation mapping an invalid manual entry onto the
  structured 422 error payload (Req 15.3, 18.1).

The ``get_db`` dependency is overridden with an in-memory SQLite session
(``StaticPool`` so the ``:memory:`` database is shared across the seed session
and each request-scoped session) seeded with three products, two of which share
the brand "Amul" so a substring query returns more than one match.

No live Redis is required. The search endpoint wraps its computation in
``app.services.data_service.cached_or_compute``, whose cache helpers degrade to
a miss on read failure and skip on write failure, so a missing Redis simply
means every request computes directly. ``test_search_works_without_live_redis``
makes that explicit by forcing the cache get/set to raise and asserting the
endpoint still returns a correct 200. The manual-entry endpoint touches neither
the database nor Redis.

``raise_server_exceptions=True`` (the TestClient default) means any unhandled
exception escaping an endpoint would fail the request, so a 200 also proves the
endpoint returned cleanly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Product
from app.db.session import get_db
from app.main import app
from app.services import data_service
from app.services.search_service import (
    NO_RESULTS_MESSAGE,
    PROMPT_ENTER_QUERY_MESSAGE,
    SOURCE_MANUAL,
    STATUS_EMPTY_QUERY,
    STATUS_NO_RESULTS,
    STATUS_OK,
)


def _add_product(session, product_id, name, normalized_name, brand, category):
    session.add(
        Product(
            id=product_id,
            name=name,
            normalized_name=normalized_name,
            brand=brand,
            category=category,
        )
    )


def _seed(session) -> None:
    """Seed three products; two share the brand "Amul" so a query matches both."""

    _add_product(
        session,
        "amul-butter-500",
        "Amul Butter 500 g",
        "amul butter 500 g",
        "Amul",
        "fmcg/dairy",
    )
    _add_product(
        session,
        "amul-cheese-200",
        "Amul Cheese Slices 200 g",
        "amul cheese slices 200 g",
        "Amul",
        "fmcg/dairy",
    )
    _add_product(
        session,
        "tata-salt-1kg",
        "Tata Salt 1 kg",
        "tata salt 1 kg",
        "Tata",
        "fmcg/staples",
    )
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


def test_matching_query_returns_results_with_identifying_fields(client):
    """A matching query yields 200 with results carrying name/brand/category.

    Confirms the success contract (Req 1.1, 1.2, 14.4): ``status`` is ``"ok"``,
    the two "Amul" products are returned, and every result carries a non-empty
    id, name, brand, and category (Correctness Property 1).
    """
    resp = client.get("/api/v1/search", params={"q": "amul"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == STATUS_OK
    assert body["query"] == "amul"
    assert body["manual_entry"] is False

    results = body["results"]
    assert len(results) == 2
    for result in results:
        assert result["id"]
        assert result["name"]
        assert result["brand"]
        assert result["category"]
    # Both matches are the seeded Amul products.
    assert {r["id"] for r in results} == {"amul-butter-500", "amul-cheese-200"}
    assert all(r["brand"] == "Amul" for r in results)


def test_empty_query_returns_prompt(client):
    """An empty query yields 200 with the enter-a-name prompt (Req 1.4)."""
    resp = client.get("/api/v1/search", params={"q": ""})

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == STATUS_EMPTY_QUERY
    assert body["results"] == []
    assert body["message"] == PROMPT_ENTER_QUERY_MESSAGE
    assert body["manual_entry"] is False


def test_no_match_returns_no_results_and_manual_entry(client):
    """A query with no match yields the no-results message + affordance (Req 1.5)."""
    resp = client.get("/api/v1/search", params={"q": "zzzznonexistent"})

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == STATUS_NO_RESULTS
    assert body["results"] == []
    assert body["message"] == NO_RESULTS_MESSAGE
    assert body["manual_entry"] is True


def test_manual_entry_valid_returns_selected_product(client):
    """A valid manual entry yields 200 with a manual-source SelectedProduct (Req 1.6)."""
    payload = {
        "name": "Homemade Mango Jam",
        "displayed_price": 120.0,
        "reference_price": 200.0,
        "pack_quantity": 250.0,
        "pack_unit": "g",
        "category": "fmcg/spreads",
        "brand": "HomeMade",
    }
    resp = client.post("/api/v1/manual-entry", json=payload)

    assert resp.status_code == 200
    body = resp.json()

    assert body["source"] == SOURCE_MANUAL
    assert body["name"] == "Homemade Mango Jam"
    assert body["displayed_price"] == 120.0
    assert body["reference_price"] == 200.0
    assert body["pack_quantity"] == 250.0
    assert body["pack_unit"] == "g"
    assert body["category"] == "fmcg/spreads"
    assert body["brand"] == "HomeMade"
    # A manual entry is assigned a synthetic id namespaced under "manual:".
    assert body["id"].startswith("manual:")


def test_manual_entry_blank_name_returns_structured_422(client):
    """A whitespace-only name is rejected as a structured 422 (Req 15.3).

    A name that is only whitespace passes the boundary's ``min_length`` check
    but is rejected by the service, whose ``ManualEntryError`` the endpoint maps
    onto the structured payload with the offending field preserved.
    """
    resp = client.post(
        "/api/v1/manual-entry",
        json={"name": "   ", "displayed_price": 10.0},
    )

    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "INVALID_MANUAL_ENTRY"
    assert error["status"] == 422
    assert error["details"]["field"] == "name"


def test_manual_entry_non_positive_price_returns_structured_422(client):
    """A non-positive displayed price is rejected at the boundary as 422 (Req 18.1).

    ``displayed_price`` carries a ``gt=0`` constraint, so a zero price fails
    Pydantic validation and is returned as the structured validation payload.
    """
    resp = client.post(
        "/api/v1/manual-entry",
        json={"name": "Valid Name", "displayed_price": 0},
    )

    assert resp.status_code == 422
    error = resp.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["status"] == 422


def test_search_works_without_live_redis(client, monkeypatch):
    """The search endpoint still returns a correct 200 when Redis is unavailable.

    The cache layer is best-effort: :func:`cached_or_compute` reads through
    ``cache_get_json`` (degrades to a miss on failure) and writes through
    ``cache_set_json`` (skips on failure). Forcing the underlying Redis get/set
    to raise simulates a down/absent Redis and proves the endpoint falls back to
    computing the result directly rather than erroring (Req 12.3).
    """

    def _boom(*args, **kwargs):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr(data_service, "cache_get", _boom)
    monkeypatch.setattr(data_service, "cache_set", _boom)

    resp = client.get("/api/v1/search", params={"q": "amul"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == STATUS_OK
    assert {r["id"] for r in body["results"]} == {
        "amul-butter-500",
        "amul-cheese-200",
    }
