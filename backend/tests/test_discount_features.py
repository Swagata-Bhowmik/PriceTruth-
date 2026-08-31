"""Verification for the snapshot-aware discount feature transform (Task 4.1, Req 2.3).

These are example-based unit tests for
``app.ml.discount_model.engineer_features`` - the pure, category-relative
feature transform that later tasks (labeling 4.2, training 4.3, inference 4.4,
SHAP 4.5) build on. They confirm the behaviour the task calls out:

* an **inflated** listing (reference far above the category mean, displayed near
  the mean) produces a high positive ``reference_price_z`` and a high
  ``claimed_discount_pct`` - the signature of a manufactured discount;
* a **genuine** listing (reference near the category norm, modest markdown)
  produces only modest values;
* **degenerate stats** (``std == 0``) never raise - the guarded division makes
  the transform total, collapsing z-scores to ``0.0``.

Two contract checks are included because 4.3/4.4/4.5 depend on them: the feature
dict is keyed by exactly ``FEATURE_NAMES``, and missing review inputs fall back
to the neutral category mean.

The numbered property-based tests (Correctness Properties, task 4.6) are
separate; these are concrete anchors that also print the computed feature
vectors for the task report (visible under ``pytest -s``).
"""

import math

from app.ml.discount_model import (
    FEATURE_NAMES,
    MAX_STAR_RATING,
    engineer_features,
    engineer_features_frame,
    features_to_vector,
)

# A representative category distribution. ``mean_discount_pct`` / ``std`` are
# expressed as fractions to match ``claimed_discount_pct`` (see the module's
# unit-consistency note).
CATEGORY_STATS = {
    "mean_price": 1000.0,
    "median_price": 1000.0,
    "std_price": 300.0,
    "p25_price": 800.0,
    "p75_price": 1200.0,
    "mean_discount_pct": 0.15,
    "std_discount_pct": 0.10,
    "mean_rating": 4.0,
    "mean_rating_count": 500.0,
}


def test_inflated_listing_flags_high_reference_z_and_discount():
    """(a) Inflated 'original' price -> high reference_price_z & claimed_discount_pct."""

    inflated = engineer_features(
        displayed_price=1000.0,  # near the category mean
        reference_price=5000.0,  # far above it: an inflated "original"
        category_stats=CATEGORY_STATS,
        rating=4.2,
        rating_count=750,
    )
    print("INFLATED features:", inflated)

    # A steep *claimed* discount driven by the inflated reference.
    assert inflated["claimed_discount_pct"] > 0.5
    # The reference price sits many category std-devs above the mean.
    assert inflated["reference_price_z"] > 3.0
    # (5000 - 1000) / 300 == 13.33..., (5000 - 1000) / 5000 == 0.8
    assert math.isclose(inflated["reference_price_z"], 4000.0 / 300.0, rel_tol=1e-9)
    assert math.isclose(inflated["claimed_discount_pct"], 0.8, rel_tol=1e-9)
    # The claimed discount is far above the category norm.
    assert inflated["discount_vs_category_z"] > 3.0


def test_genuine_listing_yields_modest_values():
    """(b) A real, modest markdown near the category norm -> modest features."""

    genuine = engineer_features(
        displayed_price=850.0,
        reference_price=1000.0,  # reference right at the category mean
        category_stats=CATEGORY_STATS,
        rating=4.1,
        rating_count=600,
    )
    print("GENUINE features:", genuine)

    # Modest claimed discount, in line with the category.
    assert genuine["claimed_discount_pct"] < 0.3
    assert math.isclose(genuine["claimed_discount_pct"], 0.15, rel_tol=1e-9)
    # The reference is not inflated: it sits at the mean, so z ~ 0.
    assert abs(genuine["reference_price_z"]) < 1.0
    assert math.isclose(genuine["reference_price_z"], 0.0, abs_tol=1e-9)
    # Discount is right at the category norm -> z ~ 0.
    assert math.isclose(genuine["discount_vs_category_z"], 0.0, abs_tol=1e-9)


def test_inflated_dominates_genuine_on_key_signals():
    """The inflated listing scores strictly higher on the two headline signals."""

    inflated = engineer_features(1000.0, 5000.0, CATEGORY_STATS)
    genuine = engineer_features(850.0, 1000.0, CATEGORY_STATS)

    assert inflated["reference_price_z"] > genuine["reference_price_z"]
    assert inflated["claimed_discount_pct"] > genuine["claimed_discount_pct"]


def test_zero_std_stats_do_not_raise_and_zero_out_z_scores():
    """(c) Degenerate (std == 0) stats keep the transform total, not raising."""

    degenerate = {
        "mean_price": 1000.0,
        "median_price": 0.0,  # also exercise a zero ratio denominator
        "std_price": 0.0,
        "p25_price": 0.0,
        "p75_price": 0.0,
        "mean_discount_pct": 0.15,
        "std_discount_pct": 0.0,
        "mean_rating": 0.0,
        "mean_rating_count": 0.0,
    }

    # Must not raise despite every denominator being zero.
    features = engineer_features(1200.0, 6000.0, degenerate)
    print("ZERO-STD features:", features)

    assert features["displayed_price_z"] == 0.0
    assert features["reference_price_z"] == 0.0
    assert features["discount_vs_category_z"] == 0.0
    assert features["displayed_vs_median"] == 0.0
    assert features["reference_vs_p75"] == 0.0
    assert features["rating_vs_category"] == 0.0
    # Every value is a finite float.
    assert all(math.isfinite(v) for v in features.values())


def test_feature_dict_matches_the_frozen_contract():
    """The transform emits exactly FEATURE_NAMES, and features_to_vector orders them."""

    features = engineer_features(850.0, 1000.0, CATEGORY_STATS)

    assert tuple(features.keys()) == FEATURE_NAMES
    assert set(features.keys()) == set(FEATURE_NAMES)

    vector = features_to_vector(features)
    assert vector == [features[name] for name in FEATURE_NAMES]
    assert len(vector) == len(FEATURE_NAMES)


def test_missing_review_inputs_fall_back_to_neutral_category_mean():
    """Absent rating / rating_count default to the category mean (neutral)."""

    features = engineer_features(850.0, 1000.0, CATEGORY_STATS)

    # rating_norm = mean_rating / 5, rating_vs_category = mean/mean = 1.0,
    # rating_count_log = log1p(mean) - log1p(mean) = 0.0
    assert math.isclose(
        features["rating_norm"],
        CATEGORY_STATS["mean_rating"] / MAX_STAR_RATING,
        rel_tol=1e-9,
    )
    assert math.isclose(features["rating_vs_category"], 1.0, rel_tol=1e-9)
    assert math.isclose(features["rating_count_log"], 0.0, abs_tol=1e-9)


def test_dataframe_transform_matches_scalar(capsys):
    """The vectorized training helper reproduces the scalar transform per row."""

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "category": "electronics",
                "displayed_price": 1000.0,
                "reference_price": 5000.0,
                "rating": 4.2,
                "rating_count": 750,
            },
            {
                "category": "electronics",
                "displayed_price": 850.0,
                "reference_price": 1000.0,
                "rating": 4.1,
                "rating_count": 600,
            },
        ]
    )
    stats_by_category = {"electronics": CATEGORY_STATS}

    frame = engineer_features_frame(df, stats_by_category)

    assert list(frame.columns) == list(FEATURE_NAMES)
    assert len(frame) == 2

    for i, row in enumerate(df.itertuples(index=False)):
        expected = engineer_features(
            row.displayed_price,
            row.reference_price,
            CATEGORY_STATS,
            rating=row.rating,
            rating_count=row.rating_count,
        )
        for name in FEATURE_NAMES:
            assert math.isclose(
                frame.iloc[i][name], expected[name], rel_tol=1e-9, abs_tol=1e-12
            )
