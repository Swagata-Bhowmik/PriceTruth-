"""Plain-language labels for the discount model's raw features (task 4.5, Req 3.5).

Requirement 3.5 says the SHAP breakdown must label every displayed feature with
a plain-language name rather than a raw model identifier such as
``reference_price_z``. This module is the single source of truth for that
mapping: it turns each entry of
:data:`app.ml.discount_model.FEATURE_NAMES` into a short shopper-facing phrase,
and :func:`label_for` is the one helper the SHAP explainer
(``app/ml/explainer.py``) uses to render a contribution's label.

Keeping the map here (and nowhere else) means the explainer never has to know
how a feature is phrased, and Property 6 ("every entry's displayed label is
drawn from the plain-language label map and is never a raw model feature
identifier") has exactly one place to hold. The labels are written from the
shopper's point of view - each answers "what about this listing pushed the
verdict?" - and deliberately avoid statistics jargon (z-scores, percentiles).

The module is import-cheap and side-effect free: it depends only on the frozen
feature contract in :mod:`app.ml.discount_model` (which itself imports only the
standard library), so importing it never pulls in pandas, xgboost, or shap.
"""

from __future__ import annotations

from app.ml.discount_model import FEATURE_NAMES

__all__ = ["FEATURE_LABELS", "label_for", "unlabeled_features"]


# Plain-language label per raw feature (Req 3.5). Every key is an entry of
# FEATURE_NAMES; the phrasing mirrors the design's examples (e.g.
# ``reference_price_z`` -> "How inflated the 'original' price looks vs. the
# category"). Phrased for a shopper, not a data scientist.
FEATURE_LABELS: dict[str, str] = {
    "claimed_discount_pct": "Size of the claimed discount",
    "discount_vs_category_z": "Size of the claimed discount vs. the category norm",
    "displayed_price_z": "How the selling price compares to the category",
    "reference_price_z": "How inflated the 'original' price looks vs. the category",
    "displayed_vs_median": "Selling price vs. the typical price in the category",
    "reference_vs_p75": "How high the 'original' price is vs. the category's premium prices",
    "rating_norm": "Product rating",
    "rating_count_log": "Number of reviews vs. the category norm",
    "rating_vs_category": "Product rating vs. the category average",
}


def label_for(feature_name: str) -> str:
    """Return the plain-language label for a raw model feature name (Req 3.5).

    Args:
        feature_name: A raw feature identifier, normally an entry of
            :data:`~app.ml.discount_model.FEATURE_NAMES`.

    Returns:
        The mapped shopper-facing label. For an unmapped name (which should not
        occur for the frozen feature set) the identifier is humanized -
        underscores become spaces and the phrase is title-cased - so the result
        is still never the bare raw identifier, preserving the Req 3.5 guarantee.
    """

    label = FEATURE_LABELS.get(feature_name)
    if label is not None:
        return label
    # Defensive fallback: humanize rather than leak a raw identifier.
    return feature_name.replace("_", " ").strip().title()


def unlabeled_features() -> tuple[str, ...]:
    """Return any :data:`FEATURE_NAMES` entries missing a plain-language label.

    A maintenance guard for tests: appending a feature to the model contract
    without adding its label here shows up as a non-empty tuple, so the label
    map is kept in lock-step with the feature set without raising at import time.
    """

    return tuple(name for name in FEATURE_NAMES if name not in FEATURE_LABELS)
