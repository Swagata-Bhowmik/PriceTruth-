"""Buy Timing Signal API endpoint (Task 11.3).

Exposes ``GET /api/v1/buy-timing/{category}`` - the thin HTTP boundary in front
of the pure ``app.services.buy_timing_service.recommend_buy_timing`` function
(Requirement 6). The endpoint's only jobs are to:

* resolve a request-scoped database session via the ``get_db`` dependency, and
* return the service result verbatim as JSON (Req 6.1, 6.4, 14.4).

All response shaping - the ``buy_now`` / ``wait`` recommendation (Req 6.1), the
deepest-discount window on a ``wait`` result (Req 6.2), the category-level +
snapshot-data disclosure attached to every result (Req 6.4, 10.1), and the
"unavailable" result for a category with no seasonal profile (Req 6.6) - lives
in the service and is exercised by its own unit and property tests. This module
adds no logic of its own beyond dependency wiring.

``current_month`` is intentionally *not* exposed as a request parameter: the
service defaults it to the current calendar month, which is the only sensible
value for a real shopper. The parameter exists purely so the service's own
tests can evaluate a deterministic month.

A category with no seasonal profile is *not* an error: the service reports it
with ``available=False`` and a message, so this endpoint responds 200 in both
the available and unavailable cases; FastAPI serialises the service ``dict`` to
JSON at the boundary (Req 14.4).

The path parameter uses the ``:path`` converter so a category label that itself
contains slashes (the platform stores categories like
``"electronics/headphones"``) is captured whole rather than truncated at the
first slash. The router is mounted under the ``/api/v1`` prefix by ``app.main``;
it only carries the ``/buy-timing`` path segment itself, so the resolved path is
``/api/v1/buy-timing/{category}``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.buy_timing_service import recommend_buy_timing

# The router owns the ``/buy-timing`` segment only; ``app.main`` includes it with
# ``prefix="/api/v1"`` so the resolved path is ``/api/v1/buy-timing/{category}``.
router = APIRouter(prefix="/buy-timing")


@router.get("/{category:path}")
def read_buy_timing(category: str, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Return the category-level buy-timing recommendation as JSON (Req 6.1, 6.4).

    Delegates to
    :func:`app.services.buy_timing_service.recommend_buy_timing`, letting
    ``current_month`` default to the current calendar month. The service result
    is returned unchanged and therefore has its shape::

        {
            "category": str,
            "available": bool,            # a seasonal profile exists (Req 6.6)
            "level": "category",          # Req 6.3 scoping
            "current_month": int,         # 1-12
            "recommendation": "buy_now" | "wait" | None,
            "best_window": { ... } | None,
            "disclosure": str,            # always present (Req 6.4, 10.1)
            "message": str,
        }

    A ``wait`` recommendation carries the deepest-discount window (Req 6.2), and
    every result - available or not - carries the category-level / snapshot-data
    disclosure statement (Req 6.4, 10.1). A category with no stored seasonal
    profile yields an ``available=False`` result with an unavailable message
    (Req 6.6); that is a normal 200 response, not an error.

    Args:
        category: The product category to evaluate (path parameter).
        db: Request-scoped SQLAlchemy session provided by the ``get_db``
            dependency and closed automatically once the response is produced.

    Returns:
        The service result ``dict``, serialised to JSON by FastAPI (Req 14.4).
    """

    return recommend_buy_timing(db, category)
