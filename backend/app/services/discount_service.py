"""True Discount Checker service - scoring, banding, and pre-conditions (Req 2).

This module owns the *business logic* of the True Discount Checker: given a
selected product's category and a (displayed, reference) price pair, it decides
whether a discount can be evaluated at all, and - when it can - turns the
trained model's probability of a *genuine* discount into a shopper-facing
0-100 genuineness score, a classification band, and the effective discount
percentage.

It is deliberately a thin service over three collaborators, mirroring the
side-effect-free style of the other feature services (e.g.
:mod:`app.services.shrinkflation_service`):

* the per-category price distribution is read through the parameterised helper
  :func:`app.db.repositories.get_category_price_stats` (Req 18.2) - these
  category statistics are the model's only sound baseline because the public
  data is snapshot-level (Req 2.3);
* feature engineering and inference live in :mod:`app.ml.discount_model`
  (:func:`~app.ml.discount_model.engineer_features`,
  :func:`~app.ml.discount_model.predict_genuineness`,
  :func:`~app.ml.discount_model.get_model`);
* the structured error contract lives in :mod:`app.core.errors`.

No FastAPI/HTTP concerns appear here - the endpoint (task 8.3) wraps this
service, adds the SHAP breakdown, and maps the returned dicts / raised
:class:`~app.core.errors.AppError` onto HTTP responses. Keeping the module HTTP-
free is what makes it unit- and property-testable in isolation (Req 17.5).

Result shapes
-------------
:func:`check_discount` returns a plain ``dict`` in one of three shapes and
raises for the fourth (not-evaluable) case:

* **Scored** (normal path, category stats and model available)::

      {
          "displayed_price": float,
          "reference_price": float,
          "effective_discount_pct": float,   # (P_r - P_d) / P_r * 100  (Req 2.4)
          "genuineness_score": int,          # round(p * 100), in [0, 100] (Req 2.1)
          "classification": "genuine" | "moderate" | "likely_inflated",  # (Req 2.2)
          "features": {<feature name>: float, ...},  # engineered features (Req 2.3)
      }

  The engineered ``features`` dict is included so the endpoint (task 8.3) can
  hand it straight to the SHAP explainer without re-engineering the features -
  the score and its explanation must derive from the *same* feature vector.

* **Limited verification** (Req 2.6, no category statistics for the product)::

      {
          "displayed_price": float,
          "reference_price": float,
          "effective_discount_pct": float,
          "genuineness_score": None,
          "classification": "verification_limited",
          "message": str,
          "price_context": {"displayed_price": float, "reference_price": float},
      }

* **Scoring unavailable** (Req 15.1 resilience, no trained model loaded)::

      {  # same shape as limited verification, with a distinct classification
          ...,
          "genuineness_score": None,
          "classification": "scoring_unavailable",
          "message": str,
          "price_context": {...},
      }

  Returning this instead of raising keeps a model-less deployment serving the
  rest of the API (the discount module simply reports scoring as unavailable)
  rather than surfacing a 500 (Req 15.1).

* **Not evaluable** (Req 2.5): when the reference price is missing or is not
  strictly greater than the displayed price, no score is meaningful, so the
  service raises :class:`~app.core.errors.AppError` with code
  ``DISCOUNT_NOT_EVALUABLE`` (HTTP 422) and a stated reason. A score is *never*
  returned in this case.

Design decisions
----------------
* **Pre-condition first (Req 2.5).** The not-evaluable check runs before any
  data access or scoring, so an un-evaluable discount can never leak a score and
  the more expensive stats read / inference is skipped.
* **Score maps probability to [0, 100] (Req 2.1).** ``genuineness_score =
  round(p * 100)``; :func:`~app.ml.discount_model.predict_genuineness` already
  clamps ``p`` to [0, 1], and the score is clamped again defensively so the
  integer range holds exactly.
* **Banding is a total function of the score (Req 2.2).** :func:`_classify`
  maps every integer score to exactly one band using the documented thresholds
  (``>= 90`` genuine, ``60-89`` moderate, ``< 60`` likely inflated).
* **Effective discount is unrounded (Req 2.4).** ``(P_r - P_d) / P_r * 100`` is
  returned at full precision so the identity holds exactly; display rounding is
  the endpoint/UI's concern, matching the convention in the other services.
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.db.repositories import get_category_price_stats
from app.ml import discount_model

__all__ = [
    "check_discount",
    "CLASSIFICATION_GENUINE",
    "CLASSIFICATION_MODERATE",
    "CLASSIFICATION_LIKELY_INFLATED",
    "CLASSIFICATION_VERIFICATION_LIMITED",
    "CLASSIFICATION_SCORING_UNAVAILABLE",
    "GENUINE_THRESHOLD",
    "MODERATE_THRESHOLD",
    "NOT_EVALUABLE_CODE",
    "SCORE_MIN",
    "SCORE_MAX",
]

# ---------------------------------------------------------------------------
# Classification vocabulary (Req 2.2 / 2.6 / 15.1)
# ---------------------------------------------------------------------------
#: The three scored bands (Req 2.2).
CLASSIFICATION_GENUINE = "genuine"
CLASSIFICATION_MODERATE = "moderate"
CLASSIFICATION_LIKELY_INFLATED = "likely_inflated"
#: Returned with a ``None`` score when category statistics are missing (Req 2.6).
CLASSIFICATION_VERIFICATION_LIMITED = "verification_limited"
#: Returned with a ``None`` score when no trained model is available (Req 15.1).
CLASSIFICATION_SCORING_UNAVAILABLE = "scoring_unavailable"

# ---------------------------------------------------------------------------
# Band thresholds and score range (Req 2.1 / 2.2)
# ---------------------------------------------------------------------------
#: Inclusive lower bound of the ``genuine`` band (score >= 90).
GENUINE_THRESHOLD = 90
#: Inclusive lower bound of the ``moderate`` band (60 <= score < 90).
MODERATE_THRESHOLD = 60
#: The genuineness score is always an integer in this closed interval (Req 2.1).
SCORE_MIN = 0
SCORE_MAX = 100

#: Stable machine-readable error code for the not-evaluable pre-condition (Req 2.5).
NOT_EVALUABLE_CODE = "DISCOUNT_NOT_EVALUABLE"

#: Human-readable message for the not-evaluable error (Req 2.5).
_NOT_EVALUABLE_MESSAGE = (
    "A discount cannot be evaluated because the reference price is missing or is "
    "not greater than the displayed price."
)

#: Message returned when category price statistics are unavailable (Req 2.6).
_VERIFICATION_LIMITED_MESSAGE = (
    "Category price statistics are unavailable for this product; showing "
    "available price context only."
)

#: Message returned when no trained model is available for scoring (Req 15.1).
_SCORING_UNAVAILABLE_MESSAGE = (
    "Genuineness scoring is temporarily unavailable; showing available price "
    "context only."
)


def _classify(score: int) -> str:
    """Map an integer genuineness score to its classification band (Req 2.2).

    A total function over the score: ``score >= 90`` -> ``genuine``,
    ``60 <= score < 90`` -> ``moderate``, ``score < 60`` -> ``likely_inflated``.
    The boundary scores 90, 89, 60, and 59 therefore fall in ``genuine``,
    ``moderate``, ``moderate``, and ``likely_inflated`` respectively.
    """

    if score >= GENUINE_THRESHOLD:
        return CLASSIFICATION_GENUINE
    if score >= MODERATE_THRESHOLD:
        return CLASSIFICATION_MODERATE
    return CLASSIFICATION_LIKELY_INFLATED


def _effective_discount_pct(displayed_price: float, reference_price: float) -> float:
    """Return the effective discount percentage ``(P_r - P_d) / P_r * 100`` (Req 2.4).

    Only called after the not-evaluable pre-condition has passed, so
    ``reference_price`` is strictly greater than ``displayed_price`` (and, for
    the validated positive-price input space, strictly positive), making the
    division well defined. The value is returned unrounded so the identity in
    Correctness Property 4 holds exactly; the endpoint/UI rounds for display.
    """

    return (reference_price - displayed_price) / reference_price * 100.0


def _limited_result(
    *,
    classification: str,
    message: str,
    displayed_price: float,
    reference_price: float,
    effective_discount_pct: float,
) -> dict[str, Any]:
    """Build a no-score result carrying available price context.

    Shared shape for the two graceful-degradation paths - missing category
    statistics (Req 2.6, ``verification_limited``) and a missing trained model
    (Req 15.1, ``scoring_unavailable``). Both echo the prices and the effective
    discount, set ``genuineness_score`` to ``None``, and surface a clear message
    plus a ``price_context`` block so the UI can still show what is known.
    """

    return {
        "displayed_price": displayed_price,
        "reference_price": reference_price,
        "effective_discount_pct": effective_discount_pct,
        "genuineness_score": None,
        "classification": classification,
        "message": message,
        "price_context": {
            "displayed_price": displayed_price,
            "reference_price": reference_price,
        },
    }


def check_discount(
    db: Session,
    category: str,
    displayed_price: float,
    reference_price: Optional[float],
    rating: Optional[float] = None,
    rating_count: Optional[float] = None,
    model: Optional[Any] = None,
) -> dict[str, Any]:
    """Evaluate a displayed discount and return its genuineness verdict (Req 2.1-2.6).

    Pipeline:

    1. **Pre-condition (Req 2.5).** If ``reference_price`` is missing or is not
       strictly greater than ``displayed_price``, raise
       :class:`~app.core.errors.AppError` (code ``DISCOUNT_NOT_EVALUABLE``, HTTP
       422) with a stated reason. A score is never produced in this case.
    2. **Category statistics (Req 2.6).** Read the product's category price
       distribution via :func:`app.db.repositories.get_category_price_stats`.
       If none exists, return a ``verification_limited`` result (no score, with
       price context).
    3. **Model availability (Req 15.1).** Resolve the trained model (the passed
       ``model`` or :func:`app.ml.discount_model.get_model`). If none is loaded,
       return a ``scoring_unavailable`` result rather than raising a 500.
    4. **Score and band (Req 2.1-2.4).** Engineer features from the prices,
       stats, and review signals; compute ``p(genuine)``; map it to
       ``genuineness_score = round(p * 100)`` clamped to [0, 100]; classify into
       a band; and compute the effective discount percentage. The engineered
       ``features`` dict is returned so the endpoint can reuse it for the SHAP
       breakdown without re-engineering.

    Args:
        db: An open SQLAlchemy session (injected; no session is opened here).
        category: The selected product's category label, used to look up its
            price-distribution statistics.
        displayed_price: The current/selling price ``P_d``.
        reference_price: The claimed "original"/reference price ``P_r``; may be
            ``None`` (triggers the not-evaluable pre-condition).
        rating: Optional product star rating (0-5) forwarded to feature
            engineering; defaults to the category mean when absent.
        rating_count: Optional number of ratings forwarded to feature
            engineering; defaults to the category mean when absent.
        model: Optional already-loaded model to score with. Defaults to the
            process-wide instance from :func:`app.ml.discount_model.get_model`,
            so the common request path reuses the single loaded model (Req 12.4).

    Returns:
        One of the result dicts documented in the module docstring: the scored
        result, a ``verification_limited`` result (Req 2.6), or a
        ``scoring_unavailable`` result (Req 15.1).

    Raises:
        AppError: with code ``DISCOUNT_NOT_EVALUABLE`` and status 422 when the
            reference price is missing or not strictly greater than the
            displayed price (Req 2.5).
    """

    # 1. Req 2.5 - pre-condition. Checked first so an un-evaluable discount can
    #    never leak a score and no data access / inference happens needlessly.
    if reference_price is None:
        raise AppError(
            code=NOT_EVALUABLE_CODE,
            message=_NOT_EVALUABLE_MESSAGE,
            status=422,
            details={"reason": "The reference price is missing."},
        )
    if reference_price <= displayed_price:
        raise AppError(
            code=NOT_EVALUABLE_CODE,
            message=_NOT_EVALUABLE_MESSAGE,
            status=422,
            details={
                "reason": (
                    "The reference price is not greater than the displayed "
                    "price, so there is no discount to evaluate."
                )
            },
        )

    # The discount is evaluable, so the effective discount is well defined and
    # shared by every downstream result (Req 2.4).
    effective_discount_pct = _effective_discount_pct(displayed_price, reference_price)

    # 2. Req 2.6 - no category statistics means we cannot score; return price
    #    context with no genuineness score.
    stats = get_category_price_stats(db, category)
    if stats is None:
        return _limited_result(
            classification=CLASSIFICATION_VERIFICATION_LIMITED,
            message=_VERIFICATION_LIMITED_MESSAGE,
            displayed_price=displayed_price,
            reference_price=reference_price,
            effective_discount_pct=effective_discount_pct,
        )

    # 3. Req 15.1 - resilience: with no trained model we degrade to a limited
    #    result instead of raising a 500. Resolve the model once and reuse it.
    if model is None:
        model = discount_model.get_model()
    if model is None:
        return _limited_result(
            classification=CLASSIFICATION_SCORING_UNAVAILABLE,
            message=_SCORING_UNAVAILABLE_MESSAGE,
            displayed_price=displayed_price,
            reference_price=reference_price,
            effective_discount_pct=effective_discount_pct,
        )

    # 4. Req 2.1-2.4 - normal scored path.
    features = discount_model.engineer_features(
        displayed_price,
        reference_price,
        stats,
        rating,
        rating_count,
    )
    p_genuine = discount_model.predict_genuineness(features, model)

    # Req 2.1 - map p(genuine) in [0, 1] to an integer score in [0, 100]. p is
    # already clamped upstream; clamp the score again so the range is exact.
    genuineness_score = max(SCORE_MIN, min(SCORE_MAX, round(p_genuine * 100)))
    classification = _classify(genuineness_score)

    return {
        "displayed_price": displayed_price,
        "reference_price": reference_price,
        "effective_discount_pct": effective_discount_pct,
        "genuineness_score": genuineness_score,
        "classification": classification,
        # Included so the endpoint (task 8.3) reuses the exact same feature
        # vector for the SHAP breakdown rather than re-engineering it (Req 2.3).
        "features": features,
    }
