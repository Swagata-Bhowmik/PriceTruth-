"""True Discount Checker API endpoint with the SHAP breakdown (Task 8.3).

Exposes ``POST /api/v1/discount-check`` - the thin HTTP boundary in front of the
pure ``app.services.discount_service.check_discount`` function (Requirement 2)
and the ``app.ml.explainer.explain`` breakdown (Requirement 3). The endpoint's
jobs are to:

* validate the request body at the API boundary (Req 18.1),
* resolve a request-scoped database session via the ``get_db`` dependency,
* delegate scoring/banding/pre-conditions to the discount service,
* attach the SHAP explanation to a *scored* result (Req 3.1-3.5), and
* serve repeated identical requests from Redis, degrading gracefully when Redis
  is unavailable (Req 11.3, 12.3).

Division of responsibility
---------------------------
All discount business logic lives in the service and is covered by its own unit
and property tests (``tests/test_discount_service.py``); the SHAP decomposition
lives in :mod:`app.ml.explainer`. This module adds no scoring logic of its own -
only request validation, dependency wiring, explanation attachment, and caching.

The three response shapes (Req 2, 3)
------------------------------------
1. **Scored** (category stats + model available): the service returns a dict
   carrying the engineered ``features``. This endpoint hands that exact feature
   vector to :func:`app.ml.explainer.explain` so the explanation derives from
   the *same* features that produced the score, attaches the breakdown under
   ``explanation``, and drops the internal ``features`` key so the response is
   the clean set ``{displayed_price, reference_price, effective_discount_pct,
   genuineness_score, classification, explanation}`` (Req 3.1).
2. **Limited verification / scoring unavailable** (``genuineness_score`` is
   ``None``): returned verbatim with a 200 and **no** ``explanation`` block
   (Req 2.6, 15.1). There is nothing to explain without a score.
3. **Not evaluable** (missing reference, or reference not greater than the
   displayed price): raised by the service as
   :class:`~app.core.errors.AppError` (code ``DISCOUNT_NOT_EVALUABLE``, HTTP
   422) and rendered by the central exception handler in ``app.main`` (Req 2.5,
   15.3). This endpoint deliberately does **not** catch it.

Margin-space explanation (Req 3.3)
----------------------------------
The SHAP breakdown is expressed in the model's **margin (log-odds) space**,
where SHAP additivity is exact: ``base_value + sum(contribution.impact)``
reconciles to ``explanation.final_score`` (the model's margin output) within a
tiny tolerance. That margin ``final_score`` is intentionally distinct from the
top-level ``genuineness_score``, which is the shopper-facing 0-100 score the
service maps from the probability. Keeping the explanation in margin space is
what preserves the exact reconciliation guarantee the project is built around.

Caching (Req 11.3, 12.3)
------------------------
The fully-assembled response (score, banding, and the attached SHAP
explanation) is cached under ``discount:{category}:{displayed}:{reference}`` for
``DISCOUNT_CACHE_TTL_SECONDS`` (1h) via
:func:`app.services.data_service.cached_or_compute`. Caching the *whole* response
dict - explanation included - means a cache hit returns a result identical to a
fresh computation without re-running inference or the explainer (Req 11.3).
The cache layer is best-effort: :func:`cached_or_compute` reads through
``cache_get_json`` (a backend outage degrades to a miss) and writes through
``cache_set_json`` (a write failure is skipped), so with no Redis the endpoint
transparently falls back to computing the result directly.

The not-evaluable pre-condition raises inside the cached computation, so it
propagates cleanly to the central handler and is never cached; the cache key is
only built for an evaluable request (a present reference price), which also
keeps a ``None`` reference out of the key.

The router is mounted under the ``/api/v1`` prefix by ``app.main``; it carries
the ``/discount-check`` path itself, so the resolved path is
``/api/v1/discount-check``.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.ml.explainer import explain
from app.services.data_service import (
    DISCOUNT_CACHE_TTL_SECONDS,
    cached_or_compute,
    discount_cache_key,
)
from app.services.discount_service import check_discount

# The router owns the ``/discount-check`` path; ``app.main`` includes it with
# ``prefix="/api/v1"`` so the resolved path is ``/api/v1/discount-check``.
router = APIRouter()


class DiscountCheckRequest(BaseModel):
    """Discount-check request body, validated at the boundary (Req 18.1).

    Every field is validated for type, length, and range before the service
    runs (Req 18.1). Note the split of responsibility on the reference price:

    * ``reference_price`` is optional and, when present, must be strictly
      positive (``gt=0``); a zero or negative value is a malformed request and
      fails here as a ``422`` ``VALIDATION_ERROR``.
    * A *missing* reference, or a positive reference that is not greater than the
      displayed price, is **not** a request error: it is the not-evaluable
      pre-condition (Req 2.5), which the service raises as
      ``DISCOUNT_NOT_EVALUABLE``. Encoding that here would wrongly conflate the
      two 422 causes, so it is left to the service.
    """

    product_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "Optional identifier of the selected product; bounded in length so "
            "an oversized value is rejected at the boundary (Req 18.1). Not used "
            "for scoring (the category statistics are), but accepted so a "
            "selected product can be passed straight through."
        ),
    )
    category: str = Field(
        ...,
        min_length=1,
        description=(
            "The product's category label, used to look up its price-"
            "distribution statistics. Must be non-empty (Req 18.1)."
        ),
    )
    displayed_price: float = Field(
        ...,
        gt=0,
        description="The current/selling price P_d; strictly positive (Req 18.1).",
    )
    reference_price: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "The claimed 'original'/reference price P_r. Optional; when present "
            "it must be strictly positive (Req 18.1). A missing reference, or "
            "one not greater than the displayed price, yields the not-evaluable "
            "result from the service (Req 2.5), not a validation error."
        ),
    )


def _attach_explanation(result: dict[str, Any]) -> dict[str, Any]:
    """Attach the SHAP breakdown to a scored result and drop the raw features.

    A *scored* result is one the service produced with a genuineness score - it
    carries a non-``None`` ``genuineness_score`` and the engineered ``features``
    dict. For such a result this hands the exact engineered feature vector to
    :func:`app.ml.explainer.explain` (so the explanation derives from the same
    features that produced the score, Req 3.1, 3.4), attaches the breakdown
    under ``explanation`` (base value, margin final score, and one plain-language
    contribution per feature with its direction; Req 3.2, 3.3, 3.5), and removes
    the internal ``features`` key so the response stays the clean shopper-facing
    set.

    A limited-verification or scoring-unavailable result (``genuineness_score``
    is ``None``) has no score to explain and is returned unchanged, with no
    ``explanation`` block (Req 2.6, 15.1).
    """

    features = result.get("features")
    if result.get("genuineness_score") is None or not isinstance(features, dict):
        # Limited-verification / scoring-unavailable: nothing to explain.
        return result

    # Scored path: explain the same feature vector that produced the score, then
    # drop the internal features so the response is clean (Req 3.1).
    explanation = explain(features)
    cleaned = {key: value for key, value in result.items() if key != "features"}
    cleaned["explanation"] = explanation
    return cleaned


@router.post("/discount-check")
def check_discount_endpoint(
    request: DiscountCheckRequest, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Evaluate a displayed discount and return its verdict + SHAP breakdown.

    Delegates scoring, banding, and the pre-conditions to
    :func:`app.services.discount_service.check_discount`, then - for a scored
    result - attaches the SHAP explanation from
    :func:`app.ml.explainer.explain`. The assembled response is cached under
    ``discount:{category}:{displayed}:{reference}`` so a repeated identical
    request is served from Redis without re-running inference or the explainer
    (Req 11.3, 12.3); the cache degrades to direct computation when Redis is
    unavailable.

    Response shapes (Req 2, 3):

    * **Scored** (200)::

          {
              "displayed_price": float,
              "reference_price": float,
              "effective_discount_pct": float,   # (P_r - P_d)/P_r*100 (Req 2.4)
              "genuineness_score": int,           # 0-100 (Req 2.1)
              "classification": "genuine" | "moderate" | "likely_inflated",
              "explanation": {                    # Req 3.1-3.5, margin space
                  "base_value": float,
                  "final_score": float,           # model margin output (Req 3.3)
                  "contributions": [
                      {"feature": <plain-language label>,
                       "impact": float,
                       "direction": "toward_genuine" | "toward_inflated"},
                      ...                          # one per model feature
                  ],
              },
          }

    * **Limited verification / scoring unavailable** (200): the service dict with
      ``genuineness_score: null`` and price context, returned verbatim with no
      ``explanation`` (Req 2.6, 15.1).

    Args:
        request: The validated request body (Req 18.1).
        db: Request-scoped SQLAlchemy session from the ``get_db`` dependency,
            closed automatically once the response is produced.

    Returns:
        The response ``dict``, serialised to JSON by FastAPI (Req 14.4).

    Raises:
        AppError: with code ``DISCOUNT_NOT_EVALUABLE`` (HTTP 422) when the
            reference price is missing or not greater than the displayed price
            (Req 2.5). Raised by the service and rendered by the central handler;
            deliberately not caught here.
    """

    def _compute() -> dict[str, Any]:
        # check_discount raises AppError(DISCOUNT_NOT_EVALUABLE) for the
        # not-evaluable pre-condition (Req 2.5); that propagates uncaught so the
        # central handler renders the 422 and nothing is cached.
        result = check_discount(
            db,
            request.category,
            request.displayed_price,
            request.reference_price,
        )
        return _attach_explanation(result)

    # Only an evaluable request (a present reference price) yields a cacheable
    # result. When the reference is missing, compute directly so the service can
    # raise the not-evaluable error and no ``None`` leaks into the cache key.
    if request.reference_price is None:
        return _compute()

    return cached_or_compute(
        discount_cache_key(
            request.category, request.displayed_price, request.reference_price
        ),
        DISCOUNT_CACHE_TTL_SECONDS,
        _compute,
    )
