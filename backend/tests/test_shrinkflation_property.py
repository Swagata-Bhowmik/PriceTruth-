"""Property-based tests for the Shrinkflation Timeline service (Task 9.2).

These tests exercise
``app.services.shrinkflation_service.get_shrinkflation_timeline`` with
Hypothesis, implementing three of the design's Correctness Properties:

* Property 9  - Shrinkflation timeline is ordered and attributed (Req 4.1, 4.4)
* Property 10 - Unit price identity at each timeline point       (Req 4.2)
* Property 11 - Total pack-size and unit-price change identity   (Req 4.3)

Each property is implemented by exactly one property-based test carrying its
traceability comment and running at least 100 generated examples.

Rather than mock the data layer, each example provisions a fresh in-memory
SQLite database from ``app.db.models.Base.metadata`` (``StaticPool`` so the
``:memory:`` connection survives across queries within one example), seeds a
product plus the generated pack-size points, and drives the service end-to-end
through the real repository helper. The stored ``unit_price`` column is
deliberately seeded with a wrong value so Property 10 proves the service
*recomputes* the unit price rather than trusting the stored column.
"""

import math
from datetime import date

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, PackSizeHistory, Product
from app.services.shrinkflation_service import STATUS_OK, get_shrinkflation_timeline

# The single product whose timeline every example builds. A constant id keeps
# the foreign key valid; the properties do not depend on the id varying.
_PRODUCT_ID = "prod-under-test"

# A deliberately invalid stored unit price. Real ratios (positive selling price
# / positive pack quantity) are always positive, so a negative sentinel can
# never coincide with the recomputed value - proving the service ignores the
# stored column (Req 4.2).
_WRONG_STORED_UNIT_PRICE = -1.0


# --- Shared strategies -----------------------------------------------------

# Positive, finite magnitudes for pack quantity and selling price. Kept
# strictly positive so a unit price (price / quantity) and a percentage change
# against the first point are always well-defined.
_positive = st.floats(
    min_value=1e-2, max_value=1e6, allow_nan=False, allow_infinity=False
)

# Observation dates span a wide window and are drawn independently per point,
# so a generated list is frequently unordered and often contains duplicate
# dates - exercising the service's non-decreasing ordering guarantee and its
# stable handling of ties.
_observed_at = st.dates(min_value=date(2000, 1, 1), max_value=date(2035, 12, 31))

# Both source families the timeline attributes points to (Req 4.4).
_source_type = st.sampled_from(["off", "cited_public_record"])

# The measure units a pack quantity may be expressed in.
_pack_unit = st.sampled_from(["g", "kg", "ml", "l"])

# Optional citation text (OFF-sourced points legitimately have none). Restricted
# to printable ASCII so the value stores cleanly in SQLite.
_citation = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=40,
    ),
)

# One generated pack-size point.
_point = st.fixed_dictionaries(
    {
        "observed_at": _observed_at,
        "pack_quantity": _positive,
        "pack_unit": _pack_unit,
        "selling_price": _positive,
        "source_type": _source_type,
        "source_citation": _citation,
    }
)


def _run_timeline(points):
    """Seed a fresh in-memory DB with ``points`` and return the service result.

    Rows are inserted in the generated order (which may be non-chronological)
    so the service's ordering guarantee is genuinely exercised. The stored
    ``unit_price`` is intentionally wrong so unit prices in the result can only
    be correct if the service recomputes them.
    """

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        session.add(
            Product(
                id=_PRODUCT_ID,
                name="Product Under Test",
                normalized_name="product under test",
                category="fmcg/test",
                source="cited_public_record",
            )
        )
        for point in points:
            session.add(
                PackSizeHistory(
                    product_id=_PRODUCT_ID,
                    observed_at=point["observed_at"],
                    pack_quantity=point["pack_quantity"],
                    pack_unit=point["pack_unit"],
                    selling_price=point["selling_price"],
                    unit_price=_WRONG_STORED_UNIT_PRICE,  # service must recompute
                    source_type=point["source_type"],
                    source_citation=point["source_citation"],
                )
            )
        session.commit()
        return get_shrinkflation_timeline(session, _PRODUCT_ID)
    finally:
        session.close()
        engine.dispose()


# Feature: price-truth-platform, Property 9: Shrinkflation timeline is ordered and attributed
@settings(max_examples=150, deadline=None)
@given(st.lists(_point, min_size=1, max_size=8))
def test_timeline_is_ordered_and_attributed(points):
    """Validates: Requirements 4.1, 4.4

    For any product with recorded pack-size history, the returned points are in
    non-decreasing chronological order, and every point carries a non-empty
    source attribution.
    """
    result = _run_timeline(points)

    assert result["status"] == STATUS_OK
    returned = result["points"]
    # Every seeded point is returned; none is dropped.
    assert len(returned) == len(points)

    # Non-decreasing chronological order, including across equal dates.
    observed = [p["observed_at"] for p in returned]
    assert observed == sorted(observed)

    # Every point carries a non-empty source attribution (source_type).
    for point in returned:
        assert point["source_type"]
        assert isinstance(point["source_type"], str)


# Feature: price-truth-platform, Property 10: Unit price identity at each timeline point
@settings(max_examples=150, deadline=None)
@given(st.lists(_point, min_size=1, max_size=8))
def test_unit_price_identity_per_point(points):
    """Validates: Requirements 4.2

    For any recorded pack-size point with a positive pack quantity, the
    displayed unit price equals selling price divided by pack quantity within a
    small tolerance (and is recomputed, not read from the stored column).
    """
    returned = _run_timeline(points)["points"]

    for point in returned:
        assert point["pack_quantity"] > 0
        # Recomputed by the service, so it never equals the wrong stored value.
        assert point["unit_price"] != _WRONG_STORED_UNIT_PRICE
        assert math.isclose(
            point["unit_price"],
            point["selling_price"] / point["pack_quantity"],
            rel_tol=1e-9,
            abs_tol=0.0,
        )


# Feature: price-truth-platform, Property 11: Total pack-size and unit-price change identity
@settings(max_examples=150, deadline=None)
@given(st.lists(_point, min_size=2, max_size=8))
def test_total_change_identity(points):
    """Validates: Requirements 4.3

    For any pack-size history with two or more points, the reported total
    percentage change in pack quantity equals (last - first) / first * 100 for
    quantity, and likewise for unit price - measured across the ordered
    timeline's first and last points.
    """
    result = _run_timeline(points)
    returned = result["points"]
    total = result["total_change"]

    # With two or more points, a total change over the full period exists.
    assert len(returned) >= 2
    assert total is not None

    first, last = returned[0], returned[-1]

    # The period spans the earliest and latest points of the ordered timeline.
    assert total["period_start"] == first["observed_at"]
    assert total["period_end"] == last["observed_at"]

    expected_pack_pct = (
        (last["pack_quantity"] - first["pack_quantity"])
        / first["pack_quantity"]
        * 100.0
    )
    expected_unit_pct = (
        (last["unit_price"] - first["unit_price"]) / first["unit_price"] * 100.0
    )

    assert math.isclose(
        total["pack_quantity_pct"], expected_pack_pct, rel_tol=1e-9, abs_tol=0.0
    )
    assert math.isclose(
        total["unit_price_pct"], expected_unit_pct, rel_tol=1e-9, abs_tol=0.0
    )
