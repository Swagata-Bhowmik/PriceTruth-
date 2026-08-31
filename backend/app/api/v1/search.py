"""Product Search API endpoints (Task 12.3).

Exposes the two HTTP entry points that get a product into the five feature
modules (Requirement 1):

* ``GET /api/v1/search?q={query}`` - the thin, cached boundary over
  ``app.services.search_service.search_products`` (Req 1.1, 1.2, 1.4, 1.5).
* ``POST /api/v1/manual-entry`` - the boundary over
  ``app.services.search_service.create_manual_entry`` (Req 1.6).

All result shaping - the matched-products list carrying each product's name,
brand, and category (Req 1.1, 1.2), the empty-query prompt (Req 1.4), the
no-results message plus manual-entry affordance (Req 1.5), and the
:class:`~app.services.search_service.SelectedProduct` a manual entry yields
(Req 1.6) - lives in the service and is covered by its own unit and property
tests. This module owns only dependency wiring, result caching, boundary
validation of the manual-entry body (Req 18.1), and error translation.

Search caching (Req 12.3, 14.4)
------------------------------
The search result is cached under ``search:{sha1(query)}`` for
``SEARCH_CACHE_TTL_SECONDS`` (1h) via
:func:`app.services.data_service.cached_or_compute`, so repeated queries for the
same text are served from Redis without re-querying the database. The cache
layer is *best-effort*: :func:`cached_or_compute` reads through
``cache_get_json`` (a backend outage degrades to a miss) and writes through
``cache_set_json`` (a write failure is skipped), so when Redis is unavailable
the endpoint transparently falls back to computing the result directly. The
computed value is JSON-normalised, so a cache hit returns a value identical to a
fresh computation (Req 9.4, 12.3).

Manual-entry validation (Req 18.1) vs. domain validation (Req 1.6)
------------------------------------------------------------------
Structural constraints the request must satisfy - a non-empty name, a required
strictly-positive displayed price, and strictly-positive optional reference
price / pack quantity - are declared on the Pydantic body so a malformed request
is rejected as a ``422`` validation error at the boundary (handled by the
central ``RequestValidationError`` handler in ``app.main``, Req 15.3, 18.1). The
service applies the same minimal validation as the single source of truth and
raises :class:`~app.services.search_service.ManualEntryError` for anything the
boundary does not catch (e.g. a name that is only whitespace); this endpoint
maps that domain error onto the structured ``422`` payload, preserving the
offending ``field`` for the client (Req 15.3).

Both routes are registered on a single unprefixed router that ``app.main``
includes with ``prefix="/api/v1"``, so the resolved paths are
``/api/v1/search`` and ``/api/v1/manual-entry``.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.db.session import get_db
from app.services.data_service import (
    SEARCH_CACHE_TTL_SECONDS,
    cached_or_compute,
    search_cache_key,
)
from app.services.search_service import ManualEntryError, create_manual_entry, search_products

# Both paths (``/search`` and ``/manual-entry``) are distinct top-level routes,
# so the router carries no shared sub-prefix; ``app.main`` includes it with
# ``prefix="/api/v1"``.
router = APIRouter()


class ManualEntryRequest(BaseModel):
    """Manually entered product details, validated at the boundary (Req 1.6, 18.1).

    ``name`` and ``displayed_price`` are required; ``reference_price`` and
    ``pack_quantity`` are optional but, when supplied, must be strictly
    positive. ``pack_unit``, ``category`` and ``brand`` are optional free-text
    context. The same rules are re-checked by the service (the single source of
    truth), which additionally rejects a whitespace-only name.
    """

    name: str = Field(
        ...,
        min_length=1,
        description="Product name entered by the shopper (Req 1.6).",
    )
    displayed_price: float = Field(
        ...,
        gt=0,
        description="The price shown to the shopper; must be strictly positive.",
    )
    reference_price: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional 'original' price; strictly positive when supplied.",
    )
    pack_quantity: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional pack quantity; strictly positive when supplied.",
    )
    pack_unit: Optional[str] = Field(
        default=None, description="Optional pack unit (e.g. 'g', 'ml')."
    )
    category: Optional[str] = Field(
        default=None, description="Optional product category."
    )
    brand: Optional[str] = Field(default=None, description="Optional brand name.")


@router.get("/search")
def read_search(
    q: str = Query(
        default="",
        description="The product query text; empty yields an enter-a-name prompt.",
    ),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Search products by name and return the shaped outcome as JSON (Req 1.1-1.5).

    Delegates to :func:`app.services.search_service.search_products`, wrapping
    the call in :func:`app.services.data_service.cached_or_compute` so repeated
    queries for the same text are served from Redis within the cache validity
    period (Req 12.3). The service result is returned unchanged and therefore
    has its uniform shape::

        {
            "status": "ok" | "empty_query" | "no_results",
            "query": str,
            "results": [{"id", "name", "brand", "category"}, ...],
            "message": str | None,
            "manual_entry": bool,
        }

    An empty / whitespace / punctuation-only query yields the enter-a-name
    prompt (Req 1.4); a query that matches nothing yields the no-results message
    with ``manual_entry=True`` (Req 1.5); otherwise each result carries the
    product's id, name, brand, and category (Req 1.1, 1.2). All three are normal
    200 responses.

    Args:
        q: The query text (query-string parameter). Defaults to empty so a
            missing or blank ``q`` maps to the enter-a-name prompt (Req 1.4).
        db: Request-scoped SQLAlchemy session provided by ``get_db``.

    Returns:
        The service result ``dict``, serialised to JSON by FastAPI (Req 14.4).
    """

    return cached_or_compute(
        search_cache_key(q),
        SEARCH_CACHE_TTL_SECONDS,
        lambda: search_products(db, q),
    )


@router.post("/manual-entry")
def create_manual_product_entry(body: ManualEntryRequest) -> dict[str, Any]:
    """Accept a manually entered product and return the SelectedProduct (Req 1.6).

    Hands the validated body to
    :func:`app.services.search_service.create_manual_entry`, which produces the
    same :class:`~app.services.search_service.SelectedProduct` a searched
    selection yields (with ``source == "manual"``) so every feature module
    consumes it identically. The result is returned as JSON via ``.to_dict()``.

    A :class:`~app.services.search_service.ManualEntryError` from the service -
    raised for a value the Pydantic boundary does not reject, such as a
    whitespace-only name - is translated into the structured 422 error payload,
    carrying the offending field so the client can highlight it (Req 15.3).

    Args:
        body: The validated manual-entry request body.

    Returns:
        The :class:`SelectedProduct` as a plain dict, serialised to JSON by
        FastAPI (Req 14.4).

    Raises:
        AppError: With code ``INVALID_MANUAL_ENTRY`` and HTTP 422 when the
            service rejects the entry.
    """

    try:
        selected = create_manual_entry(
            name=body.name,
            displayed_price=body.displayed_price,
            reference_price=body.reference_price,
            pack_quantity=body.pack_quantity,
            pack_unit=body.pack_unit,
            category=body.category,
            brand=body.brand,
        )
    except ManualEntryError as err:
        raise AppError(
            code="INVALID_MANUAL_ENTRY",
            message=str(err),
            status=422,
            details={"field": err.field},
        ) from err

    return selected.to_dict()
