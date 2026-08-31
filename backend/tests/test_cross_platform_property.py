"""Property-based tests for the Cross-Platform Aggregator service (Task 10.2).

These tests exercise
``app.services.cross_platform_service.aggregate_cross_platform`` with
Hypothesis, implementing two of the design's Correctness Properties:

* Property 18 - Cross-platform entries mirror available data (Req 7.1, 7.3, 7.4)
* Property 19 - Best deal is the minimum platform price       (Req 7.2)

Each property is implemented by exactly one property-based test carrying its
traceability comment and running at least 100 generated examples.

Rather than mock the data layer, each example provisions a fresh in-memory
SQLite database from ``app.db.models.Base.metadata`` (``StaticPool`` so the
``:memory:`` connection survives across queries within one example), seeds a
product plus the generated ``platform_prices`` rows, and drives the service
end-to-end through the real repository helper.

Coverage of the 0 / 1 / 2+ platform cases is split across the two generators:
the Property 18 generator draws 0..5 rows (distinct Supported Platforms), so it
exercises the no-data, single-platform, and multi-platform branches; the
Property 19 generator draws 2..8 rows (duplicate platforms allowed) so the
best-deal branch always has a genuine comparison to make.
"""

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, PlatformPrice, Product
from app.services.cross_platform_service import (
    SUPPORTED_PLATFORMS,
    aggregate_cross_platform,
)

# The single product every example aggregates. A constant id keeps the foreign
# key valid; the properties do not depend on the id varying.
_PRODUCT_ID = "prod-under-test"


# --- Shared strategies -----------------------------------------------------

# Platforms are drawn from the aggregator's own Supported-Platform list so the
# generator stays in lock-step with the service under test.
_platform = st.sampled_from(SUPPORTED_PLATFORMS)

# A listing price: strictly positive and finite (positivity is guaranteed by
# the ingestion pipeline; the aggregator does not itself reject prices).
_price = st.floats(
    min_value=1e-2, max_value=1e6, allow_nan=False, allow_infinity=False
)

# A product link. The ``product_url`` column is non-nullable, so every listing
# has one; the "https://" prefix guarantees a clearly non-empty link regardless
# of the generated path segment (Req 7.3).
_product_url = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=40
).map(lambda path: "https://example.com/" + path)

# Req 7.4: the score is nullable, so a listing either carries an integer in
# [0, 100] or has none at all.
_genuineness_score = st.one_of(
    st.none(), st.integers(min_value=0, max_value=100)
)

# One generated platform_prices row.
_row = st.fixed_dictionaries(
    {
        "platform": _platform,
        "price": _price,
        "product_url": _product_url,
        "genuineness_score": _genuineness_score,
    }
)


def _run_aggregate(rows):
    """Seed a fresh in-memory DB with ``rows`` and return the service result.

    Each row becomes a ``platform_prices`` listing for the single product under
    test; the service is then driven through the real repository helper.
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
            )
        )
        for row in rows:
            session.add(
                PlatformPrice(
                    product_id=_PRODUCT_ID,
                    platform=row["platform"],
                    price=row["price"],
                    product_url=row["product_url"],
                    genuineness_score=row["genuineness_score"],
                )
            )
        session.commit()
        return aggregate_cross_platform(session, _PRODUCT_ID)
    finally:
        session.close()
        engine.dispose()


# Feature: price-truth-platform, Property 18: Cross-platform entries mirror available data
@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        _row,
        min_size=0,
        max_size=len(SUPPORTED_PLATFORMS),
        unique_by=lambda row: row["platform"],
    )
)
def test_cross_platform_entries_mirror_available_data(rows):
    """Validates: Requirements 7.1, 7.3, 7.4

    For any set of platform prices, the returned entries correspond exactly to
    the platforms that have data; each entry carries a non-empty product link;
    and an entry shows a genuineness score if and only if the underlying
    listing has one, matching its value.
    """
    result = _run_aggregate(rows)
    entries = result["platforms"]

    # Req 7.1: exactly one entry per listing with data - no more, no fewer, and
    # for the same platforms that were present.
    assert sorted(e["platform"] for e in entries) == sorted(
        r["platform"] for r in rows
    )
    # 'available' mirrors whether any platform has data (empty vs non-empty).
    assert result["available"] == (len(rows) > 0)

    rows_by_platform = {row["platform"]: row for row in rows}
    for entry in entries:
        source = rows_by_platform[entry["platform"]]

        # Req 7.3: every entry carries a non-empty product link, mirroring data.
        assert entry["product_url"]
        assert entry["product_url"] == source["product_url"]

        # Req 7.4: a score is present if and only if the listing has one, and
        # equals the stored value when present.
        if source["genuineness_score"] is None:
            assert "genuineness_score" not in entry
        else:
            assert entry["genuineness_score"] == source["genuineness_score"]


# Feature: price-truth-platform, Property 19: Best deal is the minimum platform price
@settings(max_examples=200, deadline=None)
@given(st.lists(_row, min_size=2, max_size=8))
def test_best_deal_is_minimum_platform_price(rows):
    """Validates: Requirements 7.2

    For any set of two or more platform prices, exactly one entry is marked the
    best deal, its price equals the minimum price present, and no entry has a
    lower price.
    """
    result = _run_aggregate(rows)
    entries = result["platforms"]

    # Two or more platforms means a comparison exists and a best deal is marked.
    assert result["comparison_available"] is True
    best_entries = [entry for entry in entries if entry.get("best_deal")]
    assert len(best_entries) == 1
    best = best_entries[0]

    prices = [entry["price"] for entry in entries]
    # The marked entry holds the minimum price and nothing undercuts it.
    assert best["price"] == min(prices)
    assert all(price >= best["price"] for price in prices)
    # The reported best-deal platform names that same winning entry.
    assert result["best_deal_platform"] == best["platform"]
