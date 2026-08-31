"""Property-based tests for the Data Service (Task 7.4).

These tests exercise ``app.services.data_service`` with Hypothesis, implementing
four of the design's Correctness Properties. Each property is implemented by
exactly one property-based test carrying its traceability tag and running at
least 100 generated examples:

* Property 20 - Missing OFF fields degrade gracefully               (Req 9.1)
* Property 21 - Cache returns results identical to fresh computation (Req 9.4, 12.3)
* Property 22 - External and input values are validated before use   (Req 9.5, 15.4, 18.1)
* Property 24 - OFF-derived results disclose their crowd-sourced origin (Req 10.3)

Generator notes
---------------
* Numeric magnitudes are constrained to the realistic OFF value space (well
  within IEEE-754 double range). A pack quantity is never an astronomically
  large integer, so the generators exercise the intended input domain rather
  than pathological values.
* Property 21 replaces the module-level ``cache_get`` / ``cache_set`` with an
  in-memory dict-backed fake *inside the test* (assigning the attributes on the
  ``data_service`` module and restoring them in a ``finally``) rather than using
  a function-scoped ``pytest`` fixture, so Hypothesis does not raise the
  function-scoped-fixture health check. A cached JSON value uses ``None`` as its
  miss sentinel, so the generated top-level result is never ``None`` (a
  top-level null is, by design, indistinguishable from a cache miss).
"""

from __future__ import annotations

import json
import math

from hypothesis import given, settings
from hypothesis import strategies as st

from app.services import data_service
from app.services.data_service import (
    OFF_PRODUCT_FIELDS,
    SOURCE_OPEN_FOOD_FACTS,
    cached_or_compute,
    validate_off_product,
)

# Every property runs comfortably more than the required 100 examples.
MAX_EXAMPLES = 200


# ===========================================================================
# Property 20: Missing OFF fields degrade gracefully
# ===========================================================================

# Free text with no surrounding whitespace, so a present value survives
# validation unchanged (the validator strips text and rejects blanks).
_clean_text = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126),
    min_size=1,
    max_size=24,
)

# A positive, finite pack quantity inside the accepted range (0, 1e6].
_valid_quantity = st.floats(
    min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False
)

# Valid values for every used field; a per-example subset of these is included.
_full_off_values = st.fixed_dictionaries(
    {
        "product_name": _clean_text,
        "brands": _clean_text,
        "quantity": _valid_quantity,
        "categories": _clean_text,
    }
)

# An arbitrary subset (possibly empty, possibly all) of the used fields.
_present_fields = st.sets(st.sampled_from(OFF_PRODUCT_FIELDS))


# Feature: price-truth-platform, Property 20: Missing OFF fields degrade gracefully
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(values=_full_off_values, present=_present_fields)
def test_missing_off_fields_degrade_gracefully(values, present):
    """Validates: Requirements 9.1

    For an OFF payload with an arbitrary subset of the used fields omitted,
    ``validate_off_product`` returns every present field unchanged, marks every
    omitted field unavailable, and never raises.
    """
    raw = {field: values[field] for field in present}

    shaped = validate_off_product(raw)  # must never raise

    # Exactly the omitted fields are reported unavailable.
    expected_missing = set(OFF_PRODUCT_FIELDS) - set(present)
    assert set(shaped["unavailable_fields"]) == expected_missing

    for field in OFF_PRODUCT_FIELDS:
        if field in present:
            # Present field is returned unchanged (quantity normalised to float).
            expected = float(values[field]) if field == "quantity" else values[field]
            assert shaped[field] == expected
            assert field not in shaped["unavailable_fields"]
        else:
            # A missing field degrades to unavailable rather than raising.
            assert shaped[field] is None
            assert field in shaped["unavailable_fields"]


# ===========================================================================
# Property 21: Cache returns results identical to fresh computation
# ===========================================================================

# JSON leaves whose top-level form is never ``None`` (the cache treats a
# top-level null as a miss). Magnitudes are bounded to keep JSON round-trips
# exact and fast.
_json_leaf = st.one_of(
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12),
    st.text(max_size=32),
)

# Nested values *may* contain null; only the top-level value must be non-null.
_json_nested = st.recursive(
    st.none() | _json_leaf,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=8), children, max_size=4),
    max_leaves=15,
)

# A cacheable compute result: a non-null leaf, or a (possibly nested) list/dict.
_cacheable_value = st.one_of(
    _json_leaf,
    st.lists(_json_nested, max_size=4),
    st.dictionaries(st.text(max_size=8), _json_nested, max_size=4),
)


# Feature: price-truth-platform, Property 21: Cache returns results identical to fresh computation
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(
    key=st.text(min_size=1, max_size=48),
    ttl=st.integers(min_value=1, max_value=86_400),
    value=_cacheable_value,
)
def test_cache_returns_results_identical_to_fresh_computation(key, ttl, value):
    """Validates: Requirements 9.4, 12.3

    For a cacheable request, the value served from the cache within its validity
    period equals the value produced by computing from scratch for the same key:
    ``cached_or_compute`` computes exactly once, then serves an identical value
    from the cache without recomputing.
    """
    store: dict[str, str] = {}

    def fake_get(cache_key):
        return store.get(cache_key)

    def fake_set(cache_key, serialized, ttl_seconds):
        store[cache_key] = serialized

    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return value

    def must_not_compute():
        raise AssertionError("compute_fn must not run on a cache hit")

    # Swap the module-level cache helpers for the in-memory fake for the
    # duration of this example only (no function-scoped fixture -> no health
    # check), then restore them.
    original_get = data_service.cache_get
    original_set = data_service.cache_set
    data_service.cache_get = fake_get
    data_service.cache_set = fake_set
    try:
        # 1. Cold cache -> computed from scratch and cached.
        fresh = cached_or_compute(key, ttl, compute)
        assert calls["count"] == 1

        # 2. Same key within validity -> served from cache, no recompute.
        served = cached_or_compute(key, ttl, must_not_compute)
        assert calls["count"] == 1  # compute_fn was not called again
        # Cache hit equals the fresh computation (Req 9.4, 12.3).
        assert served == fresh

        # 3. Independently recomputing from an empty cache yields the same value
        #    for the same key.
        store.clear()
        recomputed = cached_or_compute(key, ttl, compute)
        assert calls["count"] == 2
        assert recomputed == served

        # The served value is the deterministic JSON normalisation of the result.
        assert served == json.loads(json.dumps(value, sort_keys=True))
    finally:
        data_service.cache_get = original_get
        data_service.cache_set = original_set


# ===========================================================================
# Property 22: External and input values are validated before use
# ===========================================================================
#
# Each case is a ``(raw_value, expected_valid, expected_number)`` triple whose
# expectation is fixed by construction, so the test does not merely re-implement
# the validator. Magnitudes stay within the double range.

_valid_number = st.one_of(
    st.integers(min_value=1, max_value=1_000_000),
    st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
)

# Valid: the number itself, its plain string form, and a whitespace-padded
# string form (the validator strips before parsing).
_valid_numeric_case = _valid_number.map(lambda n: (n, True, float(n)))
_valid_numeric_string_case = _valid_number.map(lambda n: (str(n), True, float(n)))
_valid_padded_string_case = st.tuples(
    _valid_number,
    st.sampled_from(["", " ", "  ", "\t", "\n"]),
    st.sampled_from(["", " ", "  "]),
).map(lambda t: (f"{t[1]}{t[0]}{t[2]}", True, float(t[0])))

# Invalid: None, booleans, zero, negatives, non-finite, out-of-range magnitudes,
# garbage strings, and wrong container types - all must be rejected.
_invalid_case = st.one_of(
    st.just((None, False, None)),
    st.booleans().map(lambda b: (b, False, None)),
    st.sampled_from([0, 0.0, "0", "0.0", " 0 "]).map(lambda v: (v, False, None)),
    st.integers(min_value=-1_000_000, max_value=-1).map(lambda n: (n, False, None)),
    st.floats(min_value=-1e6, max_value=-1e-3, allow_nan=False, allow_infinity=False).map(
        lambda n: (n, False, None)
    ),
    st.sampled_from([float("nan"), float("inf"), float("-inf")]).map(
        lambda v: (v, False, None)
    ),
    st.sampled_from(["nan", "inf", "-inf", "Infinity"]).map(lambda v: (v, False, None)),
    st.sampled_from(
        ["abc", "750 g", "", "   ", "12,000", "1.2.3", "$5", "5%", "one", "e10"]
    ).map(lambda v: (v, False, None)),
    # Absurd magnitude above the accepted ceiling (still float-safe).
    st.floats(min_value=1e6 + 1, max_value=1e12, allow_nan=False, allow_infinity=False).map(
        lambda n: (n, False, None)
    ),
    st.integers(min_value=1_000_001, max_value=10**12).map(lambda n: (n, False, None)),
    # Wrong container / structural types.
    st.sampled_from([[], {}, [500], {"value": 1}, ("500",)]).map(
        lambda v: (v, False, None)
    ),
)

_quantity_case = st.one_of(
    _valid_numeric_case,
    _valid_numeric_string_case,
    _valid_padded_string_case,
    _invalid_case,
)


# Feature: price-truth-platform, Property 22: External and input values are validated before use
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(case=_quantity_case)
def test_external_and_input_values_are_validated_before_use(case):
    """Validates: Requirements 9.5, 15.4, 18.1

    For any quantity value, ``validate_off_product`` accepts valid positive,
    finite numbers and numeric strings (normalising them to ``float``) and
    rejects invalid ones (None, non-positive, non-finite, out-of-range, garbage
    strings, booleans, wrong types) by marking the field unavailable - never
    raising and never letting a bad value reach a feature module.
    """
    raw_value, expected_valid, expected_number = case
    raw = {
        "product_name": "Valid Name",
        "brands": "Valid Brand",
        "quantity": raw_value,
        "categories": "Valid Category",
    }

    shaped = validate_off_product(raw)  # must never raise

    if expected_valid:
        assert shaped["quantity"] == expected_number
        assert math.isfinite(shaped["quantity"]) and shaped["quantity"] > 0
        assert "quantity" not in shaped["unavailable_fields"]
    else:
        assert shaped["quantity"] is None
        assert "quantity" in shaped["unavailable_fields"]

    # A rejected external value never contaminates the sibling valid fields.
    assert shaped["product_name"] == "Valid Name"
    assert shaped["brands"] == "Valid Brand"
    assert shaped["categories"] == "Valid Category"


# ===========================================================================
# Property 24: OFF-derived results disclose their crowd-sourced origin
# ===========================================================================

# Arbitrary scalars, including non-finite floats. Integer/float magnitudes are
# bounded to the double range so validation never overflows.
_payload_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12),
    st.sampled_from([float("nan"), float("inf"), float("-inf")]),
    st.text(max_size=24),
)

_payload_nested = st.recursive(
    _payload_scalar,
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(st.text(max_size=8), children, max_size=5),
    max_leaves=15,
)

# Realistic OFF-ish product dicts that draw on the used keys plus noise keys.
_off_like_dict = st.dictionaries(
    keys=st.sampled_from(
        list(OFF_PRODUCT_FIELDS) + ["product_quantity", "category", "status", "_id"]
    ),
    values=_payload_nested,
    max_size=8,
)

# Any payload at all: well-formed mappings and malformed non-mappings.
_arbitrary_payload = st.one_of(
    st.none(),
    st.integers(min_value=-(10**12), max_value=10**12),
    st.text(max_size=24),
    st.lists(_payload_nested, max_size=5),
    _off_like_dict,
    _payload_nested,
)


# Feature: price-truth-platform, Property 24: OFF-derived results disclose their crowd-sourced origin
@settings(max_examples=MAX_EXAMPLES, deadline=None)
@given(payload=_arbitrary_payload)
def test_off_results_disclose_crowd_sourced_origin(payload):
    """Validates: Requirements 10.3

    For any payload - a well-formed product mapping or a malformed non-mapping -
    the shaped result always discloses its crowd-sourced Open Food Facts origin
    via ``crowd_sourced=True`` and ``source == SOURCE_OPEN_FOOD_FACTS``, and
    never raises.
    """
    shaped = validate_off_product(payload)  # must never raise

    assert shaped["crowd_sourced"] is True
    assert shaped["source"] == SOURCE_OPEN_FOOD_FACTS
    assert SOURCE_OPEN_FOOD_FACTS == "open_food_facts"

    # The disclosure is always accompanied by the stable, complete field shape.
    assert "unavailable_fields" in shaped
    assert isinstance(shaped["unavailable_fields"], list)
    for field in OFF_PRODUCT_FIELDS:
        assert field in shaped
