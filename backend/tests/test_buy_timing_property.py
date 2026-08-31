"""Property-based tests for the Buy Timing Signal service (Task 11.2).

These tests exercise
``app.services.buy_timing_service.recommend_buy_timing`` with Hypothesis,
implementing two of the design's Correctness Properties:

* Property 16 - Buy-timing output is category-level, bounded, and disclosed
  (Req 6.1, 6.3, 6.4, 10.1)
* Property 17 - A "wait" recommendation points to the deepest-discount window
  (Req 6.2)

Each property is implemented by exactly one property-based test carrying its
traceability comment and running at least 100 generated examples.

Rather than mock the data layer, every example provisions a fresh in-memory
SQLite database from ``app.db.models.Base.metadata`` (``StaticPool`` so the
``:memory:`` connection survives across queries within one example), seeds a
category's twelve ``category_seasonality`` rows directly (month 1-12 with
generated positive ``relative_price_index`` values), and drives the service
end-to-end through the real repository helper.

Property 16 generates arbitrary positive monthly profiles so both the
``buy_now`` and ``wait`` branches are exercised. Property 17 generates profiles
constructed so the service must recommend ``wait`` - the unique deepest-discount
window lies in a future month and is materially cheaper than the current month -
so the "displayed window is the deepest discount" guarantee is checked on a
genuine, non-vacuous ``wait`` result.
"""

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, CategorySeasonality
from app.services.buy_timing_service import DISCLOSURE, recommend_buy_timing

# The single category whose profile every example seeds. A constant label keeps
# the seeded rows and the service call in agreement; the properties do not
# depend on the label varying.
_CATEGORY = "electronics/test"

# Positive, finite relative price indices (1.0 == the category average). Kept
# strictly positive as the profile requires, and bounded to a realistic-ish
# range so generated values round-trip cleanly through SQLite REAL storage.
_rel_index = st.floats(
    min_value=0.05, max_value=5.0, allow_nan=False, allow_infinity=False
)


@st.composite
def _seasonal_profiles(draw):
    """Generate an arbitrary 12-month profile plus a month to evaluate.

    Returns ``(indices_by_month, current_month)`` where ``indices_by_month``
    maps each month 1-12 to a generated positive ``relative_price_index``. The
    indices are unconstrained, so a generated profile can drive the service
    into either the ``buy_now`` or the ``wait`` branch.
    """

    indices = draw(st.lists(_rel_index, min_size=12, max_size=12))
    indices_by_month = {month: indices[month - 1] for month in range(1, 13)}
    current_month = draw(st.integers(min_value=1, max_value=12))
    return indices_by_month, current_month


@st.composite
def _wait_profiles(draw):
    """Generate a 12-month profile the service must answer with ``wait``.

    The profile is built so that:

    * the deepest-discount window (lowest ``relative_price_index``) is a single
      future month (``best_month > current_month``), and
    * the current month sits materially above that window (by well over the
      service's materiality margin),

    which are exactly the two conditions ``recommend_buy_timing`` requires to
    return ``wait``. Every other month is placed strictly above the deepest
    window so it remains the unique global minimum.

    Returns ``(indices_by_month, current_month, best_month, min_index)``.
    """

    current_month = draw(st.integers(min_value=1, max_value=11))
    best_month = draw(st.integers(min_value=current_month + 1, max_value=12))
    # The deepest-discount window's index.
    min_index = draw(
        st.floats(
            min_value=0.30, max_value=0.80, allow_nan=False, allow_infinity=False
        )
    )
    # Place the current month materially above the deepest window (>> 0.03).
    delta = draw(
        st.floats(
            min_value=0.10, max_value=0.50, allow_nan=False, allow_infinity=False
        )
    )
    current_index = min_index + delta
    # Every other month sits strictly above the deepest window, keeping it the
    # unique global minimum (so find_best_window resolves to best_month).
    other_index = st.floats(
        min_value=min_index + 0.02,
        max_value=min_index + 2.0,
        allow_nan=False,
        allow_infinity=False,
    )

    indices_by_month = {}
    for month in range(1, 13):
        if month == best_month:
            indices_by_month[month] = min_index
        elif month == current_month:
            indices_by_month[month] = current_index
        else:
            indices_by_month[month] = draw(other_index)
    return indices_by_month, current_month, best_month, min_index


def _run_recommendation(indices_by_month, category, current_month):
    """Seed a fresh in-memory DB with the profile and return the service result.

    One ``category_seasonality`` row is inserted per month; ``is_best_window`` is
    flagged on the lowest-index month(s) to mirror how the profile builder
    persists data (the service derives the best window itself, so this is only
    for realistic seeding).
    """

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        min_index = min(indices_by_month.values())
        for month, rel_index in indices_by_month.items():
            session.add(
                CategorySeasonality(
                    category=category,
                    month=month,
                    relative_price_index=rel_index,
                    is_best_window=(rel_index == min_index),
                    sale_event=None,
                )
            )
        session.commit()
        return recommend_buy_timing(session, category, current_month=current_month)
    finally:
        session.close()
        engine.dispose()


# Feature: price-truth-platform, Property 16: Buy-timing output is category-level, bounded, and disclosed
@settings(max_examples=200, deadline=None)
@given(_seasonal_profiles())
def test_buy_timing_output_is_category_level_bounded_and_disclosed(profile):
    """Validates: Requirements 6.1, 6.3, 6.4, 10.1

    For any category with a seasonal profile, the recommendation is one of
    ``buy_now`` or ``wait``; the payload is scoped to the category
    (``level == "category"``, with no single-product single-date field); and it
    always includes the disclosure that it is category-level and derived from
    snapshot data.
    """

    indices_by_month, current_month = profile
    result = _run_recommendation(indices_by_month, _CATEGORY, current_month)

    # Req 6.1: a bounded recommendation - exactly buy_now or wait for a category
    # that has a seasonal profile.
    assert result["available"] is True
    assert result["recommendation"] in {"buy_now", "wait"}

    # Req 6.3: the payload is scoped to the category, not to a single product on
    # a single future date.
    assert result["level"] == "category"
    assert result["category"] == _CATEGORY
    single_product_or_date_fields = {
        "product_id",
        "product",
        "sku",
        "date",
        "target_date",
        "predicted_date",
    }
    assert single_product_or_date_fields.isdisjoint(result.keys())

    window = result["best_window"]
    assert window is not None
    assert single_product_or_date_fields.isdisjoint(window.keys())
    # The window is identified by a category-level month (1-12), never a
    # concrete single calendar date.
    assert isinstance(window["month"], int)
    assert 1 <= window["month"] <= 12

    # Req 6.4 / 10.1: the category-level + snapshot-data disclosure is always
    # present on the result.
    assert result["disclosure"] == DISCLOSURE
    assert "category-level" in result["disclosure"].lower()
    assert "snapshot" in result["disclosure"].lower()


# Feature: price-truth-platform, Property 17: A "wait" recommendation points to the deepest-discount window
@settings(max_examples=200, deadline=None)
@given(_wait_profiles())
def test_wait_recommendation_points_to_deepest_discount_window(profile):
    """Validates: Requirements 6.2

    Whenever the result is ``wait``, the displayed window is the profile's
    window with the largest historical reduction - the lowest
    ``relative_price_index`` in the seeded profile.
    """

    indices_by_month, current_month, best_month, min_index = profile
    result = _run_recommendation(indices_by_month, _CATEGORY, current_month)

    # The profile is constructed so the deepest window is a future, materially
    # cheaper month, so the service must recommend waiting.
    assert result["recommendation"] == "wait"

    window = result["best_window"]
    seeded_indices = list(indices_by_month.values())

    # Req 6.2: the displayed window is the one with the largest historical
    # reduction, i.e. the minimum relative_price_index in the seeded profile,
    # and no seeded month has a lower index.
    assert window["relative_price_index"] == min(seeded_indices)
    assert all(window["relative_price_index"] <= idx for idx in seeded_indices)
    assert window["month"] == best_month
