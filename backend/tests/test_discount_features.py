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
    DISPLAYED_NORM_BAND_Z,
    FEATURE_NAMES,
    LABEL_GENUINE,
    LABEL_INFLATED,
    MAX_STAR_RATING,
    REFERENCE_OUTLIER_Z,
    engineer_features,
    engineer_features_frame,
    features_to_vector,
    label_frame,
    label_row,
)

# A representative category distribution. ``mean_discount_pct`` / ``std`` are
# expressed as **percentages** in [0, 100] to match how ingestion (task 3.3)
# stores them - reduced from the ``discount_pct`` column. ``claimed_discount_pct``
# is a fraction and the transform scales it by 100 before standardising, so
# ``discount_vs_category_z`` is unit-consistent (see the module docstring).
CATEGORY_STATS = {
    "mean_price": 1000.0,
    "median_price": 1000.0,
    "std_price": 300.0,
    "p25_price": 800.0,
    "p75_price": 1200.0,
    "mean_discount_pct": 15.0,
    "std_discount_pct": 10.0,
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
    # The claimed discount is far above the category norm. Unit-consistent:
    # claimed_discount_pct is scaled to a percentage (0.8 -> 80) before it is
    # compared against the percentage category stats, so z = (80 - 15) / 10 = 6.5.
    assert inflated["discount_vs_category_z"] > 3.0
    assert math.isclose(inflated["discount_vs_category_z"], 6.5, rel_tol=1e-9)


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
    # Discount is right at the category norm -> z ~ 0. Unit-consistent: the 15%
    # claimed discount (0.15 -> 15) matches the 15.0% category mean, so
    # z = (15 - 15) / 10 = 0.0. The corrected formula keeps a genuine row's
    # discount z at a sensible magnitude (|z| well under ~5), where the old
    # fraction-vs-percentage mismatch produced a meaningless value.
    assert math.isclose(genuine["discount_vs_category_z"], 0.0, abs_tol=1e-9)
    assert abs(genuine["discount_vs_category_z"]) < 5.0


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
        "mean_discount_pct": 15.0,  # percentage, matching ingestion stats
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


# ---------------------------------------------------------------------------
# Transparent weak-supervision labeling (Task 4.2, Req 2.3 / 10.1)
# ---------------------------------------------------------------------------
# These anchor the disclosed labeling heuristic in ``label_row`` / ``label_frame``:
# a manufactured discount (an outlier "original" price with the selling price
# still sitting at the category norm) is labeled ``inflated`` (0), while a real
# markdown is labeled ``genuine`` (1).


def test_label_row_flags_manufactured_discount_as_inflated():
    """A huge outlier reference with the displayed price at the norm -> inflated (0)."""

    label = label_row(
        displayed_price=1000.0,  # sits right at the category mean (not a real markdown)
        reference_price=5000.0,  # a fabricated "original" far above the category
        category_stats=CATEGORY_STATS,
    )

    assert label == LABEL_INFLATED
    assert label == 0


def test_label_row_marks_modest_markdown_as_genuine():
    """A modest markdown with a reference inside the normal band -> genuine (1)."""

    label = label_row(
        displayed_price=850.0,
        reference_price=1000.0,  # reference at the category mean, not inflated
        category_stats=CATEGORY_STATS,
    )

    assert label == LABEL_GENUINE
    assert label == 1


def test_label_row_treats_deep_real_markdown_as_genuine_despite_high_reference():
    """A price genuinely far below the norm is genuine even if the reference is high.

    The signature of a *manufactured* discount is that the discounted price still
    sits near the norm; when the shopper actually pays well below the category
    norm the markdown is real, so the row is genuine.
    """

    features = engineer_features(500.0, 5000.0, CATEGORY_STATS)
    # Precondition: the displayed price is genuinely below the norm and the
    # reference is an outlier above it.
    assert features["displayed_price_z"] < -DISPLAYED_NORM_BAND_Z
    assert features["reference_price_z"] >= REFERENCE_OUTLIER_Z

    assert label_row(500.0, 5000.0, CATEGORY_STATS) == LABEL_GENUINE


def test_label_row_degenerate_stats_default_to_genuine():
    """With no category spread (std == 0) z-scores collapse to 0 -> genuine, not a raise."""

    degenerate = {
        "mean_price": 1000.0,
        "median_price": 0.0,
        "std_price": 0.0,
        "p25_price": 0.0,
        "p75_price": 0.0,
        "mean_discount_pct": 15.0,
        "std_discount_pct": 0.0,
        "mean_rating": 0.0,
        "mean_rating_count": 0.0,
    }

    assert label_row(1200.0, 6000.0, degenerate) == LABEL_GENUINE


def test_label_frame_matches_label_row_per_row():
    """The vectorized labeler reproduces ``label_row`` for every row and defaults genuine."""

    import pandas as pd

    df = pd.DataFrame(
        [
            {  # manufactured discount -> inflated (0)
                "category": "electronics",
                "displayed_price": 1000.0,
                "reference_price": 5000.0,
            },
            {  # modest real markdown -> genuine (1)
                "category": "electronics",
                "displayed_price": 850.0,
                "reference_price": 1000.0,
            },
            {  # deep real markdown, high reference -> genuine (1)
                "category": "electronics",
                "displayed_price": 500.0,
                "reference_price": 5000.0,
            },
        ]
    )
    stats_by_category = {"electronics": CATEGORY_STATS}

    labels = label_frame(df, stats_by_category)
    print("LABELS:", list(labels))

    assert list(labels) == [LABEL_INFLATED, LABEL_GENUINE, LABEL_GENUINE]
    assert labels.name == "label"
    assert list(labels.index) == list(df.index)

    # Identical to the scalar rule, row by row.
    for i, row in enumerate(df.itertuples(index=False)):
        assert labels.iloc[i] == label_row(
            row.displayed_price, row.reference_price, CATEGORY_STATS
        )
