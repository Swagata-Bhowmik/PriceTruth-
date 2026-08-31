"""Cross-Platform Aggregator API endpoint (Task 10.3).

Exposes ``GET /api/v1/cross-platform/{product_id}`` - the thin HTTP boundary in
front of the pure ``app.services.cross_platform_service.aggregate_cross_platform``
function (Requirement 7). The endpoint's only jobs are to:

* resolve a request-scoped database session via the ``get_db`` dependency,
* serve repeated requests for the same product from Redis by wrapping the
  computation in ``app.services.data_service.cached_or_compute`` under the
  ``crossplatform:{product_id}`` key (Req 12.3), and
* return the service result verbatim as JSON (Req 7.1, 7.3, 14.4).

All response shaping - one entry per Supported Platform that has data (Req 7.1),
the product link on every entry (Req 7.3), a genuineness score only where the
listing has one (Req 7.4), best-deal marking when two or more platforms exist
(Req 7.2), the single-platform "no comparison" message (Req 7.5), and the
"unavailable" result for a product with no platform data (Req 7.6) - lives in
the service and is exercised by its own unit and property tests. This module
adds no logic of its own beyond dependency wiring and caching.

Caching (Req 12.3)
------------------
The result is cached under ``crossplatform:{product_id}`` for
``CROSS_PLATFORM_CACHE_TTL_SECONDS`` (6h) via
:func:`app.services.data_service.cached_or_compute`, so a repeated request for
the same product is served from Redis without re-reading the database. The
cache layer is *best-effort*: :func:`cached_or_compute` reads through
``cache_get_json`` (a backend outage degrades to a miss) and writes through
``cache_set_json`` (a write failure is skipped), so when Redis is unavailable
the endpoint transparently falls back to computing the result directly. The
computed value is JSON-normalised, so a cache hit returns a value identical to
a fresh computation (Req 9.4, 12.3).

A product with no platform data is *not* an error: the service reports it with
``available=False`` and a message, so this endpoint responds 200 in both the
available and unavailable cases; FastAPI serialises the service ``dict`` to
JSON at the boundary (Req 14.4).

The router is mounted under the ``/api/v1`` prefix by ``app.main``; it only
carries the ``/cross-platform`` path segment itself, so the resolved path is
``/api/v1/cross-platform/{product_id}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.cross_platform_service import aggregate_cross_platform
from app.services.data_service import (
    CROSS_PLATFORM_CACHE_TTL_SECONDS,
    cached_or_compute,
    cross_platform_cache_key,
)

# The router owns the ``/cross-platform`` segment only; ``app.main`` includes it
# with ``prefix="/api/v1"`` so the resolved path is
# ``/api/v1/cross-platform/{product_id}``.
router = APIRouter(prefix="/cross-platform")


@router.get("/{product_id}")
def read_cross_platform_comparison(
    product_id: str, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Return the cross-platform price comparison for ``product_id`` as JSON.

    Delegates to
    :func:`app.services.cross_platform_service.aggregate_cross_platform`,
    wrapping the call in :func:`app.services.data_service.cached_or_compute` so
    repeated requests for the same product are served from Redis within the
    cache validity period (Req 12.3). The service result is returned unchanged
    and therefore has its shape::

        {
            "product_id": str,
            "available": bool,             # any platform has data (Req 7.6)
            "comparison_available": bool,  # two or more platforms (Req 7.5)
            "best_deal_platform": str | None,
            "platforms": [
                {"platform", "price", "product_url",
                 "genuineness_score"?,     # present only when the listing has one
                 "best_deal"?},            # present only on the winning entry
                ...
            ],
            "message": str | None,
        }

    Entries mirror the stored data: one per Supported Platform that has data,
    each with its product link (Req 7.1, 7.3), a genuineness score only where
    one exists (Req 7.4), and - when two or more platforms have data - the
    cheapest marked as the best deal (Req 7.2). A single platform yields a
    no-comparison message (Req 7.5) and no platform data yields an unavailable
    message (Req 7.6); both are normal 200 responses, not errors.

    Args:
        product_id: The product whose comparison is requested (path parameter).
        db: Request-scoped SQLAlchemy session provided by the ``get_db``
            dependency and closed automatically once the response is produced.

    Returns:
        The service result ``dict``, serialised to JSON by FastAPI (Req 14.4).
    """

    return cached_or_compute(
        cross_platform_cache_key(product_id),
        CROSS_PLATFORM_CACHE_TTL_SECONDS,
        lambda: aggregate_cross_platform(db, product_id),
    )
