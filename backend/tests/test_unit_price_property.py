"""Property-based tests for the Unit Price Comparator service (Task 6.2).

These tests exercise ``app.services.unit_price_service.compare_units`` with
Hypothesis, implementing four of the design's Correctness Properties:

* Property 12 - Unit price identity per variant            (Req 5.1)
* Property 13 - Best value is the minimum unit price        (Req 5.2)
* Property 14 - Unit-price comparison is invariant to scale (Req 5.4)
* Property 15 - Invalid-quantity variants are excluded      (Req 5.5)

Each property is implemented by exactly one property-based test carrying its
traceability comment and running at least 100 generated examples.
"""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services.unit_price_service import (
    REASON_MISSING_QUANTITY,
    REASON_NON_POSITIVE_QUANTITY,
    compare_units,
)

# --- Shared strategies -----------------------------------------------------

# A price is a positive, finite float (positivity is guaranteed upstream at the
# API boundary; the service does not itself reject prices).
prices = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)

# The four supported units, spanning both measure families (mass, volume).
units = st.sampled_from(["g", "kg", "ml", "l"])

# A quantity that will always be *included*: strictly positive and finite.
valid_quantities = st.floats(
    min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False
)

# A quantity that must be *excluded* per Req 5.5: missing (None) or non-positive.
invalid_quantities = st.one_of(
    st.none(),
    st.just(0.0),
    st.floats(min_value=-1e6, max_value=-1e-3, allow_nan=False, allow_infinity=False),
)

# A fully-valid variant (valid unit, positive finite quantity) -> always included.
valid_variants = st.fixed_dictionaries(
    {
        "label": st.text(min_size=1, max_size=8),
        "price": prices,
        "quantity": valid_quantities,
        "unit": units,
    }
)


# Feature: price-truth-platform, Property 12: Unit price identity per variant
@settings(max_examples=200, deadline=None)
@given(st.lists(valid_variants, min_size=1, max_size=8))
def test_unit_price_identity_per_variant(variants):
    """Validates: Requirements 5.1

    For any set of variants with positive standardized quantities, each
    included variant's unit price equals price / quantity_std.
    """
    result = compare_units(variants)

    # Every valid variant is included; none is excluded.
    assert result["excluded"] == []
    assert len(result["comparison"]) == len(variants)

    for entry in result["comparison"]:
        assert entry["quantity_std"] > 0
        assert math.isclose(
            entry["unit_price"],
            entry["price"] / entry["quantity_std"],
            rel_tol=1e-9,
            abs_tol=0.0,
        )


# Feature: price-truth-platform, Property 13: Best value is the minimum unit price
@settings(max_examples=200, deadline=None)
@given(st.lists(valid_variants, min_size=2, max_size=8))
def test_best_value_is_minimum_unit_price(variants):
    """Validates: Requirements 5.2

    For any two or more included variants, exactly one is marked best value,
    its unit price is the minimum, and no included variant is lower.
    """
    included = compare_units(variants)["comparison"]
    assert len(included) == len(variants)  # all valid -> all included

    best_entries = [entry for entry in included if entry.get("best_value")]
    assert len(best_entries) == 1
    best = best_entries[0]

    unit_prices = [entry["unit_price"] for entry in included]
    assert best["unit_price"] == min(unit_prices)
    assert all(up >= best["unit_price"] for up in unit_prices)


# Feature: price-truth-platform, Property 14: Unit-price comparison is invariant to unit scale
@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(prices, valid_quantities, st.booleans()),
        min_size=1,
        max_size=6,
    )
)
def test_unit_price_invariant_to_unit_scale(params):
    """Validates: Requirements 5.4

    Expressing a variant's quantity in kg vs g (or l vs ml) yields the same
    unit price and the same best-value selection; converting a quantity to the
    standard unit and back recovers the original within tolerance.
    """
    # Each param is (price, quantity_in_large_unit, is_mass). Build two
    # representations of the same variants: one in the large unit (kg/l) and one
    # in the small unit (g/ml), where large = small * 1000.
    large, small = [], []
    for index, (price, quantity, is_mass) in enumerate(params):
        label = f"v{index}"
        large_unit, small_unit = ("kg", "g") if is_mass else ("l", "ml")
        large.append({"label": label, "price": price, "quantity": quantity, "unit": large_unit})
        small.append(
            {"label": label, "price": price, "quantity": quantity * 1000, "unit": small_unit}
        )

    result_large = compare_units(large)
    result_small = compare_units(small)

    up_large = {e["label"]: e["unit_price"] for e in result_large["comparison"]}
    up_small = {e["label"]: e["unit_price"] for e in result_small["comparison"]}
    std_large = {e["label"]: e["quantity_std"] for e in result_large["comparison"]}

    expected_labels = {f"v{i}" for i in range(len(params))}
    assert set(up_large) == set(up_small) == expected_labels

    for index, (price, quantity, _is_mass) in enumerate(params):
        label = f"v{index}"
        # Same unit price regardless of the unit used to express the quantity.
        assert math.isclose(up_large[label], up_small[label], rel_tol=1e-9, abs_tol=0.0)
        # Standardize (x 1000) then convert back (/ 1000) recovers the original.
        recovered = std_large[label] / 1000.0
        assert math.isclose(recovered, quantity, rel_tol=1e-9, abs_tol=0.0)

    # Best-value selection is unchanged by the unit scale.
    best_large = next(e["label"] for e in result_large["comparison"] if e.get("best_value"))
    best_small = next(e["label"] for e in result_small["comparison"] if e.get("best_value"))
    assert best_large == best_small


# Feature: price-truth-platform, Property 15: Invalid-quantity variants are excluded, valid ones included once
@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(prices, st.one_of(valid_quantities, invalid_quantities), units),
        min_size=1,
        max_size=10,
    )
)
def test_invalid_quantities_excluded_valid_included_once(params):
    """Validates: Requirements 5.5

    Every variant with a missing or non-positive quantity appears in the
    excluded list with a reason and never in the comparison; every valid-quantity
    variant appears in the comparison exactly once.
    """
    variants = [
        {"label": f"v{index}", "price": price, "quantity": quantity, "unit": unit}
        for index, (price, quantity, unit) in enumerate(params)
    ]

    result = compare_units(variants)
    comparison_labels = [e["label"] for e in result["comparison"]]
    excluded_by_label = {e["label"]: e["reason"] for e in result["excluded"]}

    for index, (_price, quantity, _unit) in enumerate(params):
        label = f"v{index}"
        is_valid = quantity is not None and quantity > 0
        if is_valid:
            assert comparison_labels.count(label) == 1
            assert label not in excluded_by_label
        else:
            assert label not in comparison_labels
            assert label in excluded_by_label
            reason = excluded_by_label[label]
            assert isinstance(reason, str) and reason
            if quantity is None:
                assert reason == REASON_MISSING_QUANTITY
            else:
                assert reason == REASON_NON_POSITIVE_QUANTITY

    # No valid variant is duplicated in the comparison.
    assert len(comparison_labels) == len(set(comparison_labels))
