"""Verification for the Shrinkflation Timeline service (Task 9.1, Req 4.1-4.5).

These are example-based unit tests that seed an in-memory SQLite database from
``app.db.models.Base.metadata`` and drive
``app.services.shrinkflation_service.get_shrinkflation_timeline`` end-to-end
through the real repository helper. They confirm:

* points come back in chronological order even when inserted out of order
  (Req 4.1);
* each point's unit price equals selling_price / pack_quantity - recomputed by
  the service, not read from the stored column (Req 4.2);
* the total percentage change in pack quantity and unit price is correct across
  the full period (Req 4.3);
* every point carries a non-empty source attribution (Req 4.4);
* a product with no recorded history returns the unavailable message (Req 4.5).

The numbered property-based tests (Correctness Properties 9-11) are a separate
task (9.2); these tests are concrete examples that anchor the behaviour.
"""

import math
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, PackSizeHistory, Product
from app.services.shrinkflation_service import (
    STATUS_OK,
    STATUS_UNAVAILABLE,
    UNAVAILABLE_MESSAGE,
    get_shrinkflation_timeline,
)

# A deliberately wrong stored unit_price used to prove the service *computes*
# the unit price from selling_price / pack_quantity rather than trusting the
# precomputed column.
_WRONG_STORED_UNIT_PRICE = -1.0


def _make_session() -> tuple[Session, "object"]:
    """Create a fresh in-memory SQLite session with all tables provisioned.

    ``StaticPool`` keeps a single underlying connection so the ``:memory:``
    database survives across sessions/queries within one test.
    """

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return factory(), engine


@pytest.fixture()
def db_session():
    session, engine = _make_session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_parle_g(session: Session) -> None:
    """Seed a Parle-G product whose pack shrank 100g -> 75g at a steady price.

    Rows are added *out* of chronological order (2023 before 2019) to prove the
    service returns them chronologically. The 2019 row stores the correct unit
    price; the 2023 row stores a deliberately wrong value to prove the service
    recomputes it.
    """

    session.add(
        Product(
            id="parle-g",
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
                product_id="parle-g",
                observed_at=date(2023, 1, 1),
                pack_quantity=75.0,
                pack_unit="g",
                selling_price=10.0,
                unit_price=_WRONG_STORED_UNIT_PRICE,  # service must recompute
                source_type="cited_public_record",
                source_citation="Business Standard, 2023 shrinkflation coverage",
            ),
            PackSizeHistory(
                product_id="parle-g",
                observed_at=date(2019, 1, 1),
                pack_quantity=100.0,
                pack_unit="g",
                selling_price=10.0,
                unit_price=0.10,
                source_type="cited_public_record",
                source_citation="Economic Times, 2019 pack-size report",
            ),
        ]
    )
    session.commit()


def test_timeline_is_chronological_with_attribution(db_session):
    """Req 4.1, 4.4: ordered points, each with non-empty attribution."""
    _seed_parle_g(db_session)

    result = get_shrinkflation_timeline(db_session, "parle-g")

    assert result["status"] == STATUS_OK
    assert result["message"] is None

    points = result["points"]
    assert [p["observed_at"] for p in points] == [date(2019, 1, 1), date(2023, 1, 1)]
    # Non-decreasing chronological order (Property 9).
    observed = [p["observed_at"] for p in points]
    assert observed == sorted(observed)

    # Every point carries a non-empty source attribution (Req 4.4).
    for point in points:
        assert point["source_type"]
        assert point["source_citation"]


def test_unit_price_identity_per_point(db_session):
    """Req 4.2: unit_price == selling_price / pack_quantity, recomputed."""
    _seed_parle_g(db_session)

    points = get_shrinkflation_timeline(db_session, "parle-g")["points"]

    for point in points:
        assert math.isclose(
            point["unit_price"],
            point["selling_price"] / point["pack_quantity"],
            rel_tol=1e-9,
        )

    # The 2023 point's stored column was wrong; the service ignored it.
    latest = points[-1]
    assert latest["pack_quantity"] == 75.0
    assert not math.isclose(latest["unit_price"], _WRONG_STORED_UNIT_PRICE)
    assert math.isclose(latest["unit_price"], 10.0 / 75.0, rel_tol=1e-9)


def test_total_percentage_changes(db_session):
    """Req 4.3: total % change in pack quantity and unit price over the period."""
    _seed_parle_g(db_session)

    total = get_shrinkflation_timeline(db_session, "parle-g")["total_change"]

    assert total is not None
    assert total["period_start"] == date(2019, 1, 1)
    assert total["period_end"] == date(2023, 1, 1)

    # Pack quantity 100g -> 75g: (75 - 100) / 100 * 100 = -25%.
    assert math.isclose(total["pack_quantity_pct"], -25.0, rel_tol=1e-9)

    # Unit price 0.10 -> 0.13333...: +33.333...% at a steady shelf price.
    first_up, last_up = 10.0 / 100.0, 10.0 / 75.0
    expected_up_pct = (last_up - first_up) / first_up * 100.0
    assert math.isclose(total["unit_price_pct"], expected_up_pct, rel_tol=1e-9)
    assert math.isclose(total["unit_price_pct"], 100.0 / 3.0, rel_tol=1e-9)


def test_single_point_has_no_total_change(db_session):
    """Req 4.3: with fewer than two points there is no period to measure."""
    db_session.add(
        Product(
            id="single",
            name="Single Point Product",
            normalized_name="single point product",
            brand="BrandX",
            category="fmcg/snacks",
        )
    )
    db_session.add(
        PackSizeHistory(
            product_id="single",
            observed_at=date(2022, 6, 1),
            pack_quantity=200.0,
            pack_unit="g",
            selling_price=50.0,
            unit_price=0.25,
            source_type="off",
            source_citation=None,
        )
    )
    db_session.commit()

    result = get_shrinkflation_timeline(db_session, "single")

    assert result["status"] == STATUS_OK
    assert len(result["points"]) == 1
    assert result["total_change"] is None
    # OFF-sourced points still carry an attribution via source_type (Req 4.4).
    assert result["points"][0]["source_type"] == "off"


def test_no_history_returns_unavailable_message(db_session):
    """Req 4.5: a product with no recorded history is reported unavailable."""
    result = get_shrinkflation_timeline(db_session, "does-not-exist")

    assert result["status"] == STATUS_UNAVAILABLE
    assert result["message"] == UNAVAILABLE_MESSAGE
    assert result["points"] == []
    assert result["total_change"] is None


if __name__ == "__main__":  # pragma: no cover - manual "report outputs" run
    import json

    session, engine = _make_session()
    try:
        _seed_parle_g(session)
        available = get_shrinkflation_timeline(session, "parle-g")
        unavailable = get_shrinkflation_timeline(session, "does-not-exist")
    finally:
        session.close()
        engine.dispose()

    print("--- available (parle-g) ---")
    print(json.dumps(available, indent=2, default=str))
    print("--- unavailable (unknown product) ---")
    print(json.dumps(unavailable, indent=2, default=str))
