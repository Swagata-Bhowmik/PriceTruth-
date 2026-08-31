"""Shrinkflation Timeline API endpoint (Task 9.3).

Exposes ``GET /api/v1/shrinkflation/{product_id}`` - the thin HTTP boundary in
front of the pure ``app.services.shrinkflation_service.get_shrinkflation_timeline``
function (Requirement 4). The endpoint's only jobs are to:

* resolve a request-scoped database session via the ``get_db`` dependency, and
* hand the session and path ``product_id`` to the service and return its result
  verbatim as JSON (Req 4.1, 4.4, 14.4).

All response shaping - chronological ordering (Req 4.1), per-point unit price
(Req 4.2), total percentage change (Req 4.3), source attribution (Req 4.4), and
the graceful "unavailable" result for a product with no recorded history
(Req 4.5) - lives in the service and is exercised by its own unit and property
tests. This module therefore adds no logic of its own beyond dependency wiring.

The service always returns a plain ``dict`` with a stable key set, so it is
returned unchanged; FastAPI serialises it to JSON (Req 14.4), converting each
point's ``observed_at`` :class:`datetime.date` to an ISO-8601 string at the
boundary. A product that has no history is *not* an error: the service reports
it with ``status="unavailable"`` and a message, so this endpoint responds 200
in both the available and unavailable cases.

The router is mounted under the ``/api/v1`` prefix by ``app.main``; it only
carries the ``/shrinkflation`` path segment itself, so the resolved path is
``/api/v1/shrinkflation/{product_id}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.shrinkflation_service import get_shrinkflation_timeline

# The router owns the ``/shrinkflation`` segment only; ``app.main`` includes it
# with ``prefix="/api/v1"`` so the resolved path is
# ``/api/v1/shrinkflation/{product_id}``.
router = APIRouter(prefix="/shrinkflation")


@router.get("/{product_id}")
def read_shrinkflation_timeline(
    product_id: str, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Return the pack-size timeline for ``product_id`` as JSON (Req 4.1, 4.4, 14.4).

    Delegates to
    :func:`app.services.shrinkflation_service.get_shrinkflation_timeline`, whose
    result is returned unchanged. The response therefore has the service's
    shape::

        {
            "status": "ok" | "unavailable",
            "product_id": str,
            "points": [
                {"observed_at", "pack_quantity", "pack_unit", "selling_price",
                 "unit_price", "source_type", "source_citation"},
                ...
            ],
            "total_change": {
                "period_start", "period_end",
                "pack_quantity_pct", "unit_price_pct"
            } | None,
            "message": str | None,
        }

    ``points`` are ordered chronologically and each carries its source
    attribution (Req 4.1, 4.4). A product with no recorded pack-size history is
    reported with ``status="unavailable"`` and a message rather than an error,
    so this endpoint returns 200 in both cases (Req 4.5).

    Args:
        product_id: The product whose timeline is requested (path parameter).
        db: Request-scoped SQLAlchemy session provided by the ``get_db``
            dependency and closed automatically once the response is produced.

    Returns:
        The service result ``dict``, serialised to JSON by FastAPI (Req 14.4).
    """

    return get_shrinkflation_timeline(db, product_id)
