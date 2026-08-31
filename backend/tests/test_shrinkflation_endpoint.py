"""API tests for the Shrinkflation Timeline endpoint (Task 9.3).

Exercises ``GET /api/v1/shrinkflation/{product_id}`` end-to-end through a
FastAPI ``TestClient``. The endpoint is a thin boundary over
``app.services.shrinkflation_service.get_shrinkflation_timeline`` (already
covered by ``tests/test_shrinkflation_service.py`` and the property tests in
``tests/test_shrinkflation_property.py``), so these tests focus on what the
endpoint itself owns:

* wiring the ``get_db`` dependency to a real session and returning the service
  result as JSON (Req 4.1, 4.4, 14.4), and
* responding 200 for both the available *and* the no-history "unavailable"
  cases - a product without history is a normal result, not an error (Req 4.5).

The ``get_db`` dependency is overridden with an in-memory SQLite session
(``StaticPool`` so the ``:memory:`` database is shared across the seed session
and each request-scoped session) seeded with one product and a multi-point
pack-size series inserted out of chronological order, so the chronological
ordering guarantee is genuinely tested rather than trivially satisfied.

``raise_server_exceptions=True`` (the TestClient default) means any unhandled
exception escaping the endpoint would fail the request, so a 200 also proves
the endpoint returned cleanly.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, PackSizeHistory, Product
from app.db.session import get_db
from app.main import app
from app.services.shrinkflation_service import (
    STATUS_OK,
    STATUS_UNAVAILABLE,
    UNAVAILABLE_MESSAGE,
)

_SEEDED_PRODUCT_ID = "parle-g"
_UNKNOWN_PRODUCT_ID = "does-not-exist"


def _seed(session) -> None:
    """Seed one product whose pack shrank 100g -> 90g -> 75g at a steady price.

    Points are inserted *out* of chronological order (2023, then 2018, then
    2021) so a passing chronological assertion cannot be an artefact of
    insertion order.
    """

    session.add(
        Product(
            id=_SEEDED_PRODUCT_ID,
            name="Parle-G Original Glucose Biscuits",
            normalized_name="parle-g original glucose biscuits",
            brand="Parle",
            category="fmcg/biscuits",
            source="cited_public_record",
        )
    )
    session.add_all(
        [
            PackSizeHistory(
                product_id=_SEEDED_PRODUCT_ID,
                observed_at=date(2023, 1, 1),
                pack_quantity=75.0,
                pack_unit="g",
                selling_price=10.0,
                unit_price=10.0 / 75.0,
                source_type="cited_public_record",
                source_citation="Business Standard, 2023 shrinkflation coverage",
            ),
            PackSizeHistory(
                product_id=_SEEDED_PRODUCT_ID,
                observed_at=date(2018, 1, 1),
                pack_quantity=100.0,
                pack_unit="g",
                selling_price=10.0,
                unit_price=0.10,
                source_type="cited_public_record",
                source_citation="Economic Times, 2018 pack-size report",
            ),
            PackSizeHistory(
                product_id=_SEEDED_PRODUCT_ID,
                observed_at=date(2021, 1, 1),
                pack_quantity=90.0,
                pack_unit="g",
                selling_price=10.0,
                unit_price=10.0 / 90.0,
                source_type="off",
                source_citation=None,
            ),
        ]
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


def test_seeded_product_returns_ok_chronological_with_total_change(client):
    """A seeded product yields 200 with ordered points and a total change.

    Confirms the success contract (Req 4.1, 4.4, 14.4): ``status`` is ``"ok"``,
    the points come back in non-decreasing chronological order (even though they
    were inserted out of order), each carries its source attribution, and the
    total-change block is present because three points span a period (Req 4.3).
    """
    resp = client.get(f"/api/v1/shrinkflation/{_SEEDED_PRODUCT_ID}")

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == STATUS_OK
    assert body["product_id"] == _SEEDED_PRODUCT_ID
    assert body["message"] is None

    points = body["points"]
    assert len(points) == 3

    # Dates serialise to ISO-8601 strings, which sort identically to calendar
    # order; assert the returned order is exactly chronological (Req 4.1).
    observed = [point["observed_at"] for point in points]
    assert observed == ["2018-01-01", "2021-01-01", "2023-01-01"]
    assert observed == sorted(observed)

    # Every point carries a source attribution via source_type (Req 4.4).
    for point in points:
        assert point["source_type"]

    # Two or more points -> a total-change block spanning the full period
    # (Req 4.3). Pack quantity 100g -> 75g is a 25% reduction.
    total_change = body["total_change"]
    assert total_change is not None
    assert total_change["period_start"] == "2018-01-01"
    assert total_change["period_end"] == "2023-01-01"
    assert total_change["pack_quantity_pct"] == pytest.approx(-25.0)
    # Unit price rises even though the shelf price is unchanged (shrinkflation).
    assert total_change["unit_price_pct"] > 0


def test_unknown_product_returns_ok_status_with_unavailable_message(client):
    """An unknown product yields 200 with the unavailable status and message.

    A product with no recorded history is a normal, non-error result: the
    endpoint returns 200 with ``status="unavailable"``, the standard message,
    an empty points list, and no total change (Req 4.5).
    """
    resp = client.get(f"/api/v1/shrinkflation/{_UNKNOWN_PRODUCT_ID}")

    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == STATUS_UNAVAILABLE
    assert body["message"] == UNAVAILABLE_MESSAGE
    assert body["product_id"] == _UNKNOWN_PRODUCT_ID
    assert body["points"] == []
    assert body["total_change"] is None
