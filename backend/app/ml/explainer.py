"""SHAP explainability for the True Discount Checker (task 4.5, Req 3.1-3.5).

This module turns a single genuineness prediction into a feature-by-feature
contribution breakdown that reconciles exactly to the model's output - the
trust guarantee the whole project is built around (Req 3.3). It has two public
pieces:

* :func:`get_explainer` - builds one :class:`shap.TreeExplainer` from **the same
  loaded model instance** that produces the score
  (:func:`app.ml.discount_model.get_model`) and memoizes it for the process, so
  the explanation always derives from the scoring model (Req 3.4) and the
  explainer is constructed once rather than per request (Req 11.3, 12.4).
* :func:`explain` - returns ``{base_value, final_score, contributions}`` for one
  engineered feature mapping.

**Margin (log-odds) space.** SHAP is *additive*: for a tree model the base
(expected) value plus the per-feature SHAP contributions equal the model output.
For an XGBoost ``binary:logistic`` classifier that additivity is exact in the
raw **margin** (log-odds) space, the space ``TreeExplainer`` explains by
default and the one where ``base_value + sum(contributions)`` equals
``model.predict(..., output_margin=True)``. All values this module returns are
therefore in margin space, which is what makes the reconciliation in Property 8
hold to a tiny numeric tolerance. Mapping the probability to the shopper-facing
0-100 genuineness score is the service layer's job (task 8); this module stays
in the space where the arithmetic is exact.

**Direction.** Because a larger margin means a larger ``p(genuine)`` (the
sigmoid is monotonically increasing), a positive contribution pushes the verdict
``toward_genuine`` and a negative one ``toward_inflated`` (Req 3.2, Property 7).

**Labels.** Each contribution is labelled through
:func:`app.ml.feature_labels.label_for`, never with a raw feature identifier
(Req 3.5, Property 6), and there is exactly one contribution per model feature,
in the frozen :data:`~app.ml.discount_model.FEATURE_NAMES` order (Req 3.1).

The heavy ``shap`` import is deferred into :func:`get_explainer` so that merely
importing this module (for example at app startup before the model is loaded, or
in unrelated tests) does not pay the shap import cost; numpy and pandas are
needed by :func:`explain` for the reconciliation and the feature-named model
input, so they are imported at module load.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.ml.discount_model import FEATURE_NAMES, features_to_vector, get_model
from app.ml.feature_labels import label_for

__all__ = [
    "RECONCILIATION_TOLERANCE",
    "TOWARD_GENUINE",
    "TOWARD_INFLATED",
    "explain",
    "get_explainer",
]

# Contribution direction labels (Req 3.2). A contribution that raises the margin
# raises p(genuine); one that lowers it lowers p(genuine).
TOWARD_GENUINE = "toward_genuine"
TOWARD_INFLATED = "toward_inflated"

# Margin-space tolerance within which ``base_value + sum(contributions)`` is
# expected to equal the model's margin output. SHAP's own additivity check runs
# far tighter than this; the loose bound absorbs float32/float64 rounding across
# platforms while still being a meaningful reconciliation guarantee (Property 8).
RECONCILIATION_TOLERANCE = 1e-3


@lru_cache(maxsize=1)
def get_explainer() -> Optional[Any]:
    """Return the process-wide SHAP explainer, building it at most once (Req 3.4).

    The explainer is a :class:`shap.TreeExplainer` wrapping the single model from
    :func:`app.ml.discount_model.get_model`, so the explanation and the score
    come from the same fitted model. It is memoized for the process lifetime so
    the (non-trivial) construction happens once rather than per request
    (Req 11.3, 12.4).

    Returns:
        The explainer, or ``None`` when no model is available or the explainer
        cannot be built. A ``None`` is cached and logged, so a model-less
        deployment still starts and serves the rest of the API (Req 15.1); the
        discount path simply reports explanations as unavailable.

    Notes:
        Tests that swap the underlying model can reset the one-shot cache with
        ``get_explainer.cache_clear()`` (and usually ``get_model.cache_clear()``).
    """

    from app.core.logging import get_logger

    logger = get_logger(__name__)

    model = get_model()
    if model is None:
        logger.warning(
            "Discount model unavailable; SHAP explanations are disabled."
        )
        return None

    try:
        import shap

        explainer = shap.TreeExplainer(model)
    except Exception as exc:  # noqa: BLE001 - any build failure must degrade, not crash
        logger.warning(
            "Failed to build the SHAP explainer; explanations are disabled.",
            extra={"error": repr(exc)},
        )
        return None

    logger.info("Built SHAP TreeExplainer for the discount model.")
    return explainer


def _feature_frame(features: Mapping[str, float]) -> "pd.DataFrame":
    """Build the one-row, feature-named model input for a single listing.

    The model was trained on a frame whose columns are exactly
    :data:`FEATURE_NAMES`; feeding SHAP and the margin call the same named frame
    keeps XGBoost from warning about feature-name mismatches and guarantees each
    SHAP value lines up with the right feature.
    """

    return pd.DataFrame([features_to_vector(features)], columns=list(FEATURE_NAMES))


def _row_contributions(explanation: Any) -> np.ndarray:
    """Extract the 1-D per-feature SHAP contributions for a single-row explanation.

    ``TreeExplainer`` on a binary XGBoost model returns per-sample values shaped
    ``(1, n_features)``. This is defensive about the shape: it drops the sample
    axis and, if a trailing per-class axis is present, selects the positive
    (genuine) class column so the returned vector is always length
    ``len(FEATURE_NAMES)``.
    """

    values = np.asarray(explanation.values)
    row = np.asarray(values[0])
    if row.ndim > 1:
        # A trailing multi-output axis: take the positive (last) class column.
        row = row[..., -1] if row.shape[-1] > 1 else row[..., 0]
    return np.asarray(row).ravel()


def _row_base_value(explanation: Any) -> float:
    """Extract the scalar base (expected) value for a single-row explanation.

    ``base_values`` is one value per sample (shape ``(1,)``) for a binary model;
    a trailing per-class axis, if present, resolves to the positive class.
    """

    base = np.asarray(explanation.base_values)
    if base.ndim >= 2 and base.shape[-1] > 1:
        return float(base[0, -1])
    return float(base.ravel()[0])


def _direction(impact: float) -> str:
    """Map a contribution's sign to its verdict direction (Req 3.2, Property 7)."""

    return TOWARD_GENUINE if impact >= 0.0 else TOWARD_INFLATED


def explain(
    features: Mapping[str, float],
    explainer: Optional[Any] = None,
    model: Optional[Any] = None,
) -> dict:
    """Explain one genuineness prediction as a reconciling contribution breakdown.

    Args:
        features: Engineered feature mapping from
            :func:`app.ml.discount_model.engineer_features` (ordered internally
            through :func:`features_to_vector`, so extra/missing keys are handled
            like inference).
        explainer: An already-built SHAP explainer to reuse; defaults to the
            process-wide one from :func:`get_explainer`.
        model: The model to compute the reconciling margin output from; defaults
            to the process-wide one from :func:`app.ml.discount_model.get_model`
            (the same instance the explainer wraps, Req 3.4).

    Returns:
        A dict with:

        * ``base_value`` (float) - the explainer's expected value in margin
          (log-odds) space;
        * ``final_score`` (float) - the model's margin output for this input,
          the value the contributions reconcile to
          (``base_value + sum(impact) == final_score`` within
          :data:`RECONCILIATION_TOLERANCE`; Req 3.3, Property 8);
        * ``contributions`` - one entry per model feature, in
          :data:`FEATURE_NAMES` order (Req 3.1), each
          ``{"feature": <plain-language label>, "impact": <float, margin space>,
          "direction": "toward_genuine" | "toward_inflated"}`` (Req 3.2, 3.5).

    Raises:
        RuntimeError: if no explainer is available (none supplied and none
            built). Callers that must degrade gracefully should check
            availability first; the missing-model/explainer path is handled at
            startup/service level.
    """

    if explainer is None:
        explainer = get_explainer()
    if explainer is None:
        raise RuntimeError(
            "The SHAP explainer is unavailable; cannot explain a discount result."
        )
    if model is None:
        model = get_model()

    frame = _feature_frame(features)

    # Callable API: returns an Explanation whose base_values and values are
    # internally consistent (they reconcile to the model margin). Mixing this
    # with the legacy shap_values()/expected_value attributes is unsafe because
    # calling the object can update expected_value, so we use one API only.
    explanation = explainer(frame)
    contributions_raw = _row_contributions(explanation)
    base_value = _row_base_value(explanation)

    # The reconciliation target is the model's *own* margin output, computed
    # independently of the SHAP decomposition so the reconciliation is a real
    # check rather than a tautology (Property 8). Fall back to the additive sum
    # only if a margin call is somehow unavailable.
    if model is not None:
        final_score = float(np.asarray(model.predict(frame, output_margin=True)).ravel()[0])
    else:  # pragma: no cover - explainer implies a model in normal operation
        final_score = base_value + float(np.sum(contributions_raw))

    contributions = []
    for name, impact in zip(FEATURE_NAMES, contributions_raw):
        impact_value = float(impact)
        contributions.append(
            {
                "feature": label_for(name),
                "impact": impact_value,
                "direction": _direction(impact_value),
            }
        )

    return {
        "base_value": base_value,
        "final_score": final_score,
        "contributions": contributions,
    }
