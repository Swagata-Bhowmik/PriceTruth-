"""Unit Price Comparator API endpoint (Task 6.3).

Exposes ``POST /api/v1/unit-price/compare`` - the thin HTTP boundary in front of
the pure ``app.services.unit_price_service.compare_units`` function
(Requirement 5). The endpoint's only jobs are to:

* validate the incoming variants at the API boundary (Req 18.1), and
* hand well-formed variants to the service and return its result verbatim as
  JSON (Req 5.3, 14.4).

Boundary validation (Req 18.1) vs. graceful exclusion (Req 5.5)
---------------------------------------------------------------
The two constraints are deliberately split between this module and the service:

* ``price`` must be positive and ``unit`` must be one of ``g|kg|ml|l``. These are
  structural preconditions the service assumes, so an unsupported unit or a
  non-positive price is rejected here as a ``422`` validation error (handled by
  the central ``RequestValidationError`` handler in ``app.main``, Req 15.3).
* ``quantity`` is intentionally left unconstrained. A missing, zero, or negative
  quantity is *not* a request error: Requirement 5.5 requires such a variant to
  flow through to the service and be reported in the ``excluded`` list with a
  machine-readable reason. Adding ``gt=0`` here would wrongly turn that into a
  422 and break the design's ``"Broken"`` example.

The router is mounted under the ``/api/v1`` prefix by ``app.main``; it only
carries the ``/unit-price`` path segment itself.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.unit_price_service import compare_units

# The router owns the ``/unit-price`` segment only; ``app.main`` includes it with
# ``prefix="/api/v1"`` so the resolved path is ``/api/v1/unit-price/compare``.
router = APIRouter(prefix="/unit-price")

# Supported measure units, kept in step with the service's conversion table
# (mass: g/kg, volume: ml/l). Anchored so only an exact match is accepted; any
# other token (e.g. "oz", "L", "kilograms") fails boundary validation (Req 18.1).
_UNIT_PATTERN = "^(g|kg|ml|l)$"


class VariantIn(BaseModel):
    """A single product variant to compare, validated at the boundary (Req 18.1).

    ``label``, ``price`` and ``unit`` carry structural constraints the service
    relies on. ``quantity`` is intentionally unconstrained so a missing/zero/
    negative value reaches the service and is excluded with a reason (Req 5.5)
    rather than rejected as a request error.
    """

    label: str = Field(
        ...,
        min_length=1,
        description="Human-readable variant name (e.g. 'Family pack').",
    )
    price: float = Field(
        ...,
        gt=0,
        description="Selling price of the pack; must be strictly positive.",
    )
    quantity: float = Field(
        ...,
        description=(
            "Pack quantity expressed in `unit`. Deliberately unconstrained: a "
            "missing, zero, or negative quantity is excluded by the service "
            "with a reason (Req 5.5), not rejected here."
        ),
    )
    unit: str = Field(
        ...,
        pattern=_UNIT_PATTERN,
        description="Measure unit; one of g, kg, ml, l (Req 5.4 standardisation).",
    )


class CompareRequest(BaseModel):
    """Request body wrapping the set of variants to compare.

    Mirrors the design's example body ``{"variants": [ ... ]}``. At least one
    variant is required; an empty comparison is a malformed request (Req 18.1).
    """

    variants: list[VariantIn] = Field(
        ...,
        min_length=1,
        description="The product variants to compare by unit price (Req 5.1).",
    )


@router.post("/compare")
def compare_unit_prices(request: CompareRequest) -> dict[str, Any]:
    """Compare variants by unit price and return the service result (Req 5.3, 14.4).

    Each validated variant is passed to
    :func:`app.services.unit_price_service.compare_units`, whose result is
    returned unchanged as JSON. The response therefore has the service's shape::

        {
            "standard_unit": "g" | "ml",
            "comparison": [
                {"label", "price", "quantity_std", "unit_price"[, "best_value"]},
                ...
            ],
            "excluded": [{"label", "reason"}, ...],
        }

    ``comparison`` preserves request order and lists every variant with a valid
    quantity; the single lowest-unit-price entry carries ``"best_value": true``
    (Req 5.2). ``excluded`` lists every variant with a missing/non-positive
    quantity alongside a machine-readable reason (Req 5.5).

    Pydantic models are converted to plain dicts because ``compare_units``
    consumes mappings (it reads fields via ``.get``), keeping the service free
    of any framework coupling.
    """

    return compare_units([variant.model_dump() for variant in request.variants])
