"""Property-based tests for the SHAP explainability breakdown (Task 8.4).

These tests exercise ``app.ml.explainer.explain`` - the feature-by-feature
contribution breakdown produced for a True Discount Checker result - with
Hypothesis, implementing three of the design's Correctness Properties:

* Property 6 - SHAP breakdown is complete and plainly labelled (Req 3.1, 3.5)
* Property 7 - SHAP contribution direction matches its sign     (Req 3.2)
* Property 8 - SHAP contributions reconcile to the result       (Req 3.3)

Each property is implemented by exactly one property-based test carrying its
traceability comment and running at least 100 generated examples.

Rather than mock the model, every example builds a genuine engineered feature
mapping the same way inference does: it draws a positive displayed and reference
price plus a plausible per-category price-distribution ``category_stats`` dict,
then runs ``app.ml.discount_model.engineer_features`` and explains the result
with the real, process-wide SHAP explainer built from the trained model
artifact (``get_explainer()``). If that artifact is unavailable the whole module
skips, so the suite still passes on a model-less checkout.
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.ml import discount_model
from app.ml.discount_model import FEATURE_NAMES, engineer_features
from app.ml.explainer import (
    RECONCILIATION_TOLERANCE,
    TOWARD_GENUINE,
    TOWARD_INFLATED,
    explain,
    get_explainer,
)
from app.ml.feature_labels import FEATURE_LABELS

# Build the real model and its SHAP explainer once (both are process-wide and
# memoized). Reusing them across every generated example keeps each explanation
# fast - the per-example cost is a single-row SHAP call, not an explainer build -
# and mirrors how the running service explains a score (Req 3.4). When the model
# artifact is missing both are ``None`` and the whole module skips.
_MODEL = discount_model.get_model()
_EXPLAINER = get_explainer()

pytestmark = pytest.mark.skipif(
    _EXPLAINER is None or _MODEL is None,
    reason="Trained discount model / SHAP explainer artifact is unavailable.",
)

# The valid label vocabulary (Req 3.5) and the raw identifiers a label must
# never be (Req 3.1/3.5). Computed once from the source of truth.
_PLAIN_LABELS = frozenset(FEATURE_LABELS.values())
_RAW_IDENTIFIERS = frozenset(FEATURE_NAMES)


# --- Shared strategies -----------------------------------------------------

# Positive, finite prices (selling price, reference price, and the category's
# price-distribution statistics). Kept strictly positive so the engineered
# ratios and z-scores are well-defined the same way they are at inference.
_price = st.floats(
    min_value=1.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False
)
# A strictly positive category price spread; a real category always has one.
_std_price = st.floats(
    min_value=1.0, max_value=500_000.0, allow_nan=False, allow_infinity=False
)
# Category discount statistics are stored as percentages in [0, 100]; keep them
# strictly positive and plausibly bounded.
_mean_discount_pct = st.floats(
    min_value=1.0, max_value=90.0, allow_nan=False, allow_infinity=False
)
_std_discount_pct = st.floats(
    min_value=1.0, max_value=40.0, allow_nan=False, allow_infinity=False
)
# Star ratings live on a 0-5 scale; the category mean rating stays in [1, 5].
_rating = st.floats(
    min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False
)
_mean_rating = st.floats(
    min_value=1.0, max_value=5.0, allow_nan=False, allow_infinity=False
)
# Review counts: a non-negative observed count and a positive category mean.
_rating_count = st.integers(min_value=0, max_value=1_000_000)
_mean_rating_count = st.floats(
    min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
)


@st.composite
def _scored_features(draw):
    """Draw a realistic engineered feature mapping for a single listing.

    Mirrors the inference path: random positive prices and a plausible positive
    ``category_stats`` dict are fed through the real
    :func:`app.ml.discount_model.engineer_features`, so the features are exactly
    what the model (and therefore SHAP) would see for a real request.
    """

    displayed_price = draw(_price)
    reference_price = draw(_price)
    category_stats = {
        "mean_price": draw(_price),
        "median_price": draw(_price),
        "std_price": draw(_std_price),
        "p25_price": draw(_price),
        "p75_price": draw(_price),
        "mean_discount_pct": draw(_mean_discount_pct),
        "std_discount_pct": draw(_std_discount_pct),
        "mean_rating": draw(_mean_rating),
        "mean_rating_count": draw(_mean_rating_count),
    }
    rating = draw(_rating)
    rating_count = draw(_rating_count)

    return engineer_features(
        displayed_price,
        reference_price,
        category_stats,
        rating,
        rating_count,
    )


# Feature: price-truth-platform, Property 6: SHAP breakdown is complete and plainly labelled
@settings(max_examples=100, deadline=None)
@given(_scored_features())
def test_shap_breakdown_is_complete_and_plainly_labelled(features):
    """Validates: Requirements 3.1, 3.5

    For any scored input, the breakdown contains exactly one entry per model
    feature, and every entry's ``feature`` label is a plain-language label from
    the label map - never a raw FEATURE_NAMES identifier.
    """
    result = explain(features, explainer=_EXPLAINER, model=_MODEL)
    contributions = result["contributions"]

    # Exactly one contribution per model feature (Req 3.1).
    assert len(contributions) == len(FEATURE_NAMES)

    for contribution in contributions:
        label = contribution["feature"]
        # The label is a plain-language name (Req 3.5)...
        assert label in _PLAIN_LABELS
        # ...and never a raw model feature identifier.
        assert label not in _RAW_IDENTIFIERS


# Feature: price-truth-platform, Property 7: SHAP contribution direction matches its sign
@settings(max_examples=100, deadline=None)
@given(_scored_features())
def test_shap_direction_matches_sign(features):
    """Validates: Requirements 3.2

    For every contribution, ``direction`` is ``toward_genuine`` when its impact
    is >= 0 and ``toward_inflated`` otherwise.
    """
    result = explain(features, explainer=_EXPLAINER, model=_MODEL)

    for contribution in result["contributions"]:
        impact = contribution["impact"]
        expected = TOWARD_GENUINE if impact >= 0.0 else TOWARD_INFLATED
        assert contribution["direction"] == expected


# Feature: price-truth-platform, Property 8: SHAP contributions reconcile to the result
@settings(max_examples=100, deadline=None)
@given(_scored_features())
def test_shap_contributions_reconcile_to_result(features):
    """Validates: Requirements 3.3

    ``base_value + sum(impacts)`` equals ``final_score`` within
    ``RECONCILIATION_TOLERANCE`` - the additivity guarantee the breakdown is
    built around.
    """
    result = explain(features, explainer=_EXPLAINER, model=_MODEL)

    reconstructed = result["base_value"] + sum(
        contribution["impact"] for contribution in result["contributions"]
    )
    assert abs(reconstructed - result["final_score"]) <= RECONCILIATION_TOLERANCE
