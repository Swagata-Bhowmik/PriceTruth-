"""True Discount Checker model layer - snapshot-aware feature engineering (Req 2.3).

The single most important constraint on the discount model is that the public
data is **snapshot-level** (point-in-time records, not a per-product daily price
series). There is therefore no per-product price history to learn from; the only
statistically sound baseline is the **per-category price distribution** captured
in ``category_price_stats``. Every feature in this module is expressed *relative
to that category distribution* rather than to a product's own past.

This file is built up across several tasks; **tasks 4.1 and 4.2 are implemented
here** (the feature-engineering transform and the labeling rule). The later
stages hook in here without changing the feature contract:

* **4.2 - labeling** *(implemented)*: a transparent weak-supervision rule lives
  in :func:`label_row` (and the vectorized :func:`label_frame`). A row is labeled
  ``inflated`` when its reference/"original" price is a statistical outlier
  *above* the category distribution while its discounted price sits near the
  category norm (the discount is manufactured by inflating the reference), and
  ``genuine`` otherwise. Because no public dataset labels fake discounts, this is
  a **disclosed weak-supervision heuristic**, not ground truth; the data-sources
  panel surfaces it as a stated limitation (Req 10.1, 10.2). The rule lives in one
  documented function with named, tunable thresholds precisely so it can be
  disclosed and audited.
* **4.3 - training**: an XGBoost binary classifier trained on engineered
  features (labeled by :func:`label_frame`), serialized with joblib to
  :data:`MODEL_PATH`.
* **4.4 - inference**: the model loaded once per process into FastAPI app state,
  exposed through a reusable ``predict_proba`` here.
* **4.5 - SHAP**: a single ``shap.TreeExplainer`` built from the same loaded
  model instance (see ``app/ml/explainer.py``).

To keep those stages consistent, the raw feature order is frozen once in
:data:`FEATURE_NAMES` and every consumer (training, inference, SHAP) must order
its feature vector through :func:`features_to_vector`.

The documented features (design "True Discount Checker - Feature engineering"),
given displayed price ``P_d``, reference price ``P_r`` and a category's stats
``S``:

* ``claimed_discount_pct   = (P_r - P_d) / P_r``  (a **fraction** in [0, 1])
* ``discount_vs_category_z = (claimed_discount_pct*100 - S.mean_discount_pct) / S.std_discount_pct``
  (``claimed_discount_pct`` is scaled to a percentage so it shares the unit of
  the stored category discount stats - see the unit-consistency note below)
* ``displayed_price_z      = (P_d - S.mean_price) / S.std_price``
* ``reference_price_z      = (P_r - S.mean_price) / S.std_price``  (high == an inflated "original")
* ``displayed_vs_median    = P_d / S.median_price``
* ``reference_vs_p75       = P_r / S.p75_price``
* review signals: ``rating_norm``, ``rating_count_log``, ``rating_vs_category``

Unit-consistency convention (resolved): the category discount statistics
``S.mean_discount_pct`` / ``S.std_discount_pct`` are stored as **percentages** in
[0, 100]. This matches ingestion (task 3.3), which reduces them from the cleaned
``discount_pct`` column - a percentage clamped to [0, 100] (and, for Flipkart,
derived as ``(reference - displayed) / reference * 100``). ``claimed_discount_pct``
is instead a **fraction** in [0, 1]. To keep ``discount_vs_category_z`` on a
single scale, ``claimed_discount_pct`` is multiplied by 100 (converting it to a
percentage) before the mean is subtracted and the result is divided by the
percentage standard deviation, so numerator and denominator share the percentage
unit. ``claimed_discount_pct`` itself stays a fraction as its own feature (the
documented design). Earlier this z-score mixed a fraction with percentages and
was meaningless; scaling the fraction here removes that mismatch without touching
the ingestion stats or reordering the feature contract.

Every computation is guarded so the transform is **total**: a zero (or missing)
denominator yields ``0.0`` rather than raising, missing review inputs fall back
to the category mean (a neutral assumption), and non-finite results collapse to
``0.0``. This mirrors the side-effect-free, import-safe style of
:mod:`app.ml.seasonality`; the scalar path depends only on the standard library
(``math``) so it stays cheap to import in the request path, while the optional
DataFrame helper imports pandas lazily for the offline training path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard pandas import
    import pandas as pd

__all__ = [
    "DISPLAYED_NORM_BAND_Z",
    "FEATURE_NAMES",
    "LABEL_GENUINE",
    "LABEL_INFLATED",
    "MAX_STAR_RATING",
    "MODEL_PATH",
    "REFERENCE_OUTLIER_Z",
    "REFERENCE_P75_OUTLIER_RATIO",
    "engineer_features",
    "engineer_features_frame",
    "features_to_vector",
    "label_frame",
    "label_row",
]

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
# The frozen order of raw model features. This exact order is the contract
# reused by training (4.3), inference (4.4) and the SHAP explainer (4.5); never
# reorder in place - append only, and only with a corresponding retrain.
FEATURE_NAMES: tuple[str, ...] = (
    "claimed_discount_pct",
    "discount_vs_category_z",
    "displayed_price_z",
    "reference_price_z",
    "displayed_vs_median",
    "reference_vs_p75",
    "rating_norm",
    "rating_count_log",
    "rating_vs_category",
)

# The maximum star rating on the source marketplaces (Amazon/Flipkart use a
# 5-star scale); ``rating_norm`` normalises an absolute rating onto [0, 1].
MAX_STAR_RATING: float = 5.0

# ---------------------------------------------------------------------------
# Weak-supervision labeling contract (task 4.2, Req 2.3 / 10.1)
# ---------------------------------------------------------------------------
# The two label classes. ``genuine`` is the positive class (1) and ``inflated``
# the negative (0), matching the discount model's ``p(genuine)`` output (4.3/4.4).
LABEL_GENUINE: int = 1
LABEL_INFLATED: int = 0

# Named, tunable thresholds for the disclosed labeling heuristic. They are module
# constants (not magic numbers buried in the function) precisely so the rule can
# be disclosed and audited as a limitation (Req 10.1). All are expressed on the
# same category-relative scales the feature transform emits.
#
# A reference/"original" price counts as a statistical outlier *above* the
# category distribution when it is at least this many category std-devs above the
# mean (``reference_price_z``)...
REFERENCE_OUTLIER_Z: float = 2.0
# ...or at least this multiple of the category's 75th-percentile price
# (``reference_vs_p75``) - a second, spread-robust view of "far above the norm".
REFERENCE_P75_OUTLIER_RATIO: float = 1.5
# The discounted price "sits near the category norm" (i.e. is *not* genuinely
# below it) when its ``displayed_price_z`` is at or above ``-DISPLAYED_NORM_BAND_Z``.
# A displayed price further below the mean than this is treated as a real markdown,
# which points to ``genuine`` even when the reference looks high.
DISPLAYED_NORM_BAND_Z: float = 0.5

# Extension point (tasks 4.3 persist / 4.4 load): the trained model is
# serialized here with joblib. Resolved relative to the repo root
# (``backend/app/ml/discount_model.py`` -> parents[3] == repo root) so it does
# not depend on the process working directory. No load/train logic is
# implemented in task 4.1.
MODEL_PATH: Path = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "discount_model.pkl"
)

# The category-statistic attributes this transform reads. ``category_stats`` may
# be a mapping (dict / DataFrame row) or an object (the ``CategoryPriceStats``
# ORM row) - both are supported via :func:`_stat`.
_REQUIRED_STAT_FIELDS: tuple[str, ...] = (
    "mean_price",
    "median_price",
    "std_price",
    "p75_price",
    "mean_discount_pct",
    "std_discount_pct",
    "mean_rating",
    "mean_rating_count",
)


# ---------------------------------------------------------------------------
# Small total-arithmetic helpers (keep the transform side-effect free & total)
# ---------------------------------------------------------------------------
def _as_float(value: Any, default: float = 0.0) -> float:
    """Coerce ``value`` to a finite float, returning ``default`` otherwise.

    Handles ``None``, non-numeric strings and NaN/inf uniformly so no caller
    can push a non-finite value into the feature vector.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide, returning ``default`` for a zero / non-finite denominator.

    This is what makes z-scores and ratios total: e.g. ``std == 0`` (a category
    with no price spread) yields a ``0.0`` z-score instead of raising, as the
    design requires.
    """

    if denominator == 0.0 or not math.isfinite(denominator):
        return default
    result = numerator / denominator
    if not math.isfinite(result):
        return default
    return result


def _stat(category_stats: Any, name: str) -> Any:
    """Read attribute ``name`` from a mapping or an object (ORM row).

    Returning the raw value (possibly ``None``) is intentional; callers pass it
    through :func:`_as_float`, which supplies the neutral fallback.
    """

    if isinstance(category_stats, Mapping):
        return category_stats.get(name)
    return getattr(category_stats, name, None)


# ---------------------------------------------------------------------------
# The feature transform (task 4.1)
# ---------------------------------------------------------------------------
def engineer_features(
    displayed_price: Any,
    reference_price: Any,
    category_stats: Any,
    rating: Optional[float] = None,
    rating_count: Optional[float] = None,
) -> dict[str, float]:
    """Engineer the snapshot-aware discount features for a single listing (Req 2.3).

    Args:
        displayed_price: The current/selling price ``P_d``.
        reference_price: The claimed "original"/reference price ``P_r`` (the
            value a fake discount inflates). May be missing; the transform stays
            total (the not-evaluable pre-condition of Req 2.5 is enforced by the
            service layer, not here).
        category_stats: The listing's :class:`~app.db.models.CategoryPriceStats`
            row, or any mapping exposing the same fields
            (``mean_price``, ``median_price``, ``std_price``, ``p75_price``,
            ``mean_discount_pct``, ``std_discount_pct``, ``mean_rating``,
            ``mean_rating_count``). Missing fields are treated as ``0.0``.
            **Unit convention:** ``mean_discount_pct`` / ``std_discount_pct`` are
            **percentages** in [0, 100] (as produced by ingestion from the
            ``discount_pct`` column). ``claimed_discount_pct`` is a fraction and
            is scaled by 100 inside this transform before it is compared against
            those percentage stats, so ``discount_vs_category_z`` is
            unit-consistent.
        rating: Optional product star rating (0-5). When absent it defaults to
            the category mean, a neutral assumption that keeps a review-less
            product from being pushed toward either class.
        rating_count: Optional number of ratings. When absent it defaults to the
            category mean review volume.

    Returns:
        A ``dict[str, float]`` keyed by exactly :data:`FEATURE_NAMES`. Use
        :func:`features_to_vector` to obtain the ordered vector a model expects.
    """

    p_d = _as_float(displayed_price)
    p_r = _as_float(reference_price)

    mean_price = _as_float(_stat(category_stats, "mean_price"))
    median_price = _as_float(_stat(category_stats, "median_price"))
    std_price = _as_float(_stat(category_stats, "std_price"))
    p75_price = _as_float(_stat(category_stats, "p75_price"))
    mean_discount_pct = _as_float(_stat(category_stats, "mean_discount_pct"))
    std_discount_pct = _as_float(_stat(category_stats, "std_discount_pct"))
    mean_rating = _as_float(_stat(category_stats, "mean_rating"))
    mean_rating_count = _as_float(_stat(category_stats, "mean_rating_count"))

    # Price / discount features, all relative to the category distribution.
    # ``claimed_discount_pct`` is a fraction in [0, 1]; the stored category
    # discount stats are percentages in [0, 100] (see the module unit-consistency
    # note), so scale the fraction to a percentage before standardising so both
    # operands of the z-score share the percentage unit.
    claimed_discount_pct = _safe_div(p_r - p_d, p_r)
    discount_vs_category_z = _safe_div(
        claimed_discount_pct * 100.0 - mean_discount_pct, std_discount_pct
    )
    displayed_price_z = _safe_div(p_d - mean_price, std_price)
    reference_price_z = _safe_div(p_r - mean_price, std_price)
    displayed_vs_median = _safe_div(p_d, median_price)
    reference_vs_p75 = _safe_div(p_r, p75_price)

    # Review-signal features. Missing rating / count fall back to the category
    # mean (neutral), so ``rating_vs_category`` -> 1.0 and ``rating_count_log``
    # -> 0.0 for a review-less product. Counts are clamped to >= 0 so log1p is
    # always defined.
    rating_value = _as_float(rating, default=mean_rating)
    count_value = max(_as_float(rating_count, default=mean_rating_count), 0.0)
    mean_count = max(mean_rating_count, 0.0)

    rating_norm = _safe_div(rating_value, MAX_STAR_RATING)
    # Category-relative log review volume: log((1 + count) / (1 + mean_count)).
    # Positive == more reviews than the category norm, negative == fewer.
    rating_count_log = math.log1p(count_value) - math.log1p(mean_count)
    rating_vs_category = _safe_div(rating_value, mean_rating)

    return {
        "claimed_discount_pct": claimed_discount_pct,
        "discount_vs_category_z": discount_vs_category_z,
        "displayed_price_z": displayed_price_z,
        "reference_price_z": reference_price_z,
        "displayed_vs_median": displayed_vs_median,
        "reference_vs_p75": reference_vs_p75,
        "rating_norm": rating_norm,
        "rating_count_log": rating_count_log,
        "rating_vs_category": rating_vs_category,
    }


def features_to_vector(features: Mapping[str, float]) -> list[float]:
    """Order a feature mapping into the model input vector (contract helper).

    Training (4.3), inference (4.4) and SHAP (4.5) all convert the feature dict
    into a row through this one function so the column order can never drift
    from :data:`FEATURE_NAMES`. Missing keys default to ``0.0``.
    """

    return [_as_float(features.get(name)) for name in FEATURE_NAMES]


def engineer_features_frame(
    df: "pd.DataFrame",
    stats_by_category: Mapping[str, Any],
    *,
    category_col: str = "category",
    displayed_col: str = "displayed_price",
    reference_col: str = "reference_price",
    rating_col: str = "rating",
    rating_count_col: str = "rating_count",
) -> "pd.DataFrame":
    """Vectorized feature transform over a DataFrame of listings (training helper).

    Provided as the extension point task 4.3 uses to engineer features for the
    whole ingested corpus. It applies :func:`engineer_features` row-by-row so
    the training features are, by construction, identical to what inference
    (4.4) computes for a single listing - correctness/consistency over raw
    speed, which is ample for the ~O(20k)-row Kaggle datasets.

    Args:
        df: Rows to transform. Must contain ``category_col``, ``displayed_col``
            and ``reference_col``; ``rating_col`` / ``rating_count_col`` are
            optional (missing columns are treated as absent per-row).
        stats_by_category: Mapping of category label ->
            :class:`~app.db.models.CategoryPriceStats` row (or dict). A row whose
            category is absent from the mapping gets an empty stats object, so
            its category-relative features degrade to the total-transform
            defaults rather than raising.

    Returns:
        A new DataFrame indexed like ``df`` with exactly the
        :data:`FEATURE_NAMES` columns, in order.
    """

    import pandas as pd  # lazy: keeps the request-path import free of pandas

    empty_stats: dict[str, Any] = {}
    has_rating = rating_col in df.columns
    has_rating_count = rating_count_col in df.columns

    records: list[dict[str, float]] = []
    for row in df.itertuples(index=False):
        row_map = row._asdict()
        stats = stats_by_category.get(row_map.get(category_col), empty_stats)
        records.append(
            engineer_features(
                displayed_price=row_map.get(displayed_col),
                reference_price=row_map.get(reference_col),
                category_stats=stats,
                rating=row_map.get(rating_col) if has_rating else None,
                rating_count=(
                    row_map.get(rating_count_col) if has_rating_count else None
                ),
            )
        )

    return pd.DataFrame(records, columns=list(FEATURE_NAMES), index=df.index)


# ---------------------------------------------------------------------------
# Transparent weak-supervision labeling (task 4.2, Req 2.3 / 10.1)
# ---------------------------------------------------------------------------
def label_row(
    displayed_price: Any,
    reference_price: Any,
    category_stats: Any,
    *,
    features: Optional[Mapping[str, float]] = None,
) -> int:
    """Weak-supervision label for one listing: ``genuine`` (1) or ``inflated`` (0).

    This is the project's **disclosed weak-supervision heuristic** (Req 10.1).
    The public marketplace datasets carry no ground-truth "this discount is fake"
    label, so training labels (task 4.3) are produced by this transparent,
    category-relative rule rather than by hand. It is intentionally simple and
    lives in one documented place so the data-sources panel can surface it as a
    stated limitation (Req 10.1, 10.2).

    The rule follows the design directly. A listing is labeled **inflated** when
    *both* of the following hold, which is the signature of a *manufactured*
    discount - a real-ish selling price dressed up with a fabricated "original":

    1. the reference/"original" price is a statistical outlier **above** the
       category distribution - ``reference_price_z >= REFERENCE_OUTLIER_Z`` (many
       category std-devs above the mean) *or* ``reference_vs_p75 >=
       REFERENCE_P75_OUTLIER_RATIO`` (well above the 75th-percentile price); and
    2. the discounted price **sits near the category norm** rather than genuinely
       below it - ``displayed_price_z >= -DISPLAYED_NORM_BAND_Z``.

    Every other listing is labeled **genuine** - in particular a real markdown
    whose discounted price is meaningfully below the category norm
    (``displayed_price_z < -DISPLAYED_NORM_BAND_Z``) is genuine *even if* its
    reference looks high, because the shopper is actually paying below the norm.
    A modest markdown with a reference inside the normal band is likewise genuine.

    The decision reads only category-relative signals from
    :func:`engineer_features` (``reference_price_z``, ``reference_vs_p75``,
    ``displayed_price_z``), so it inherits the transform's total, guarded
    arithmetic: degenerate stats (``std == 0``) collapse the z-scores to ``0.0``,
    which is not an outlier, so a listing with no usable category spread defaults
    to ``genuine`` rather than raising. Thresholds are the module-level
    ``REFERENCE_OUTLIER_Z`` / ``REFERENCE_P75_OUTLIER_RATIO`` /
    ``DISPLAYED_NORM_BAND_Z`` constants so the heuristic stays tunable and
    auditable.

    Args:
        displayed_price: The current/selling price ``P_d``.
        reference_price: The claimed "original"/reference price ``P_r``.
        category_stats: The listing's category statistics (see
            :func:`engineer_features`).
        features: Optional pre-computed feature mapping for this listing. When
            supplied it is used as-is (so a caller that already engineered the
            features does not pay for the transform twice); otherwise the
            features are engineered from the three price/stat arguments.

    Returns:
        :data:`LABEL_GENUINE` (``1``) or :data:`LABEL_INFLATED` (``0``).
    """

    if features is None:
        features = engineer_features(displayed_price, reference_price, category_stats)

    reference_price_z = _as_float(features.get("reference_price_z"))
    reference_vs_p75 = _as_float(features.get("reference_vs_p75"))
    displayed_price_z = _as_float(features.get("displayed_price_z"))

    reference_is_outlier_above = (
        reference_price_z >= REFERENCE_OUTLIER_Z
        or reference_vs_p75 >= REFERENCE_P75_OUTLIER_RATIO
    )
    discounted_price_near_norm = displayed_price_z >= -DISPLAYED_NORM_BAND_Z

    if reference_is_outlier_above and discounted_price_near_norm:
        return LABEL_INFLATED
    return LABEL_GENUINE


def label_frame(
    df: "pd.DataFrame",
    stats_by_category: Mapping[str, Any],
    *,
    category_col: str = "category",
    displayed_col: str = "displayed_price",
    reference_col: str = "reference_price",
    rating_col: str = "rating",
    rating_count_col: str = "rating_count",
) -> "pd.Series":
    """Vectorized weak-supervision labels for a DataFrame of listings (task 4.3).

    The training entry point (task 4.3) engineers features for the whole ingested
    corpus and needs a matching label vector. This applies exactly the
    :func:`label_row` rule across every row, but vectorized over the engineered
    feature frame so it stays cheap on the ~O(20k)-row Kaggle datasets. Because it
    engineers features through :func:`engineer_features_frame` - which in turn
    calls :func:`engineer_features` per row - the labels are, by construction,
    identical to calling :func:`label_row` on each listing.

    Args:
        df: Rows to label; the same columns :func:`engineer_features_frame`
            expects (``category``/``displayed_price``/``reference_price`` and the
            optional review columns), overridable via the ``*_col`` arguments.
        stats_by_category: Mapping of category label -> category statistics row
            (or dict), as consumed by :func:`engineer_features_frame`.

    Returns:
        A ``pandas.Series`` named ``"label"`` and indexed like ``df``, holding
        :data:`LABEL_GENUINE` (``1``) / :data:`LABEL_INFLATED` (``0``) integers.
    """

    features = engineer_features_frame(
        df,
        stats_by_category,
        category_col=category_col,
        displayed_col=displayed_col,
        reference_col=reference_col,
        rating_col=rating_col,
        rating_count_col=rating_count_col,
    )

    reference_is_outlier_above = (
        features["reference_price_z"] >= REFERENCE_OUTLIER_Z
    ) | (features["reference_vs_p75"] >= REFERENCE_P75_OUTLIER_RATIO)
    discounted_price_near_norm = (
        features["displayed_price_z"] >= -DISPLAYED_NORM_BAND_Z
    )

    inflated = reference_is_outlier_above & discounted_price_near_norm
    # genuine (1) is the default; inflated (0) only where both conditions hold.
    labels = (~inflated).astype(int)
    labels.name = "label"
    return labels
