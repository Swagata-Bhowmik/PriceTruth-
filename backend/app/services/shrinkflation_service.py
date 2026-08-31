"""Shrinkflation Timeline service - business logic for Requirement 4.

This module builds the pack-size *timeline* the Shrinkflation Timeline feature
presents for a selected FMCG product: how the pack quantity and price have
moved over time, and the hidden unit-price increase that a steady shelf price
can conceal (classic shrinkflation).

It is a thin service over the data layer:

* the ORM schema lives in :mod:`app.db.models`,
* the parameterised read lives in
  :func:`app.db.repositories.list_pack_size_history` (Req 18.2),

so this module only *reads* the recorded points and *shapes* them into a
response the endpoint (task 9.3) can serialise directly. It performs no I/O of
its own beyond the injected :class:`~sqlalchemy.orm.Session`, which keeps it
unit- and property-testable in isolation (Req 17.5).

Response shape
--------------
:func:`get_shrinkflation_timeline` always returns a plain ``dict`` with a
stable set of keys so the endpoint can wrap it in a fixed Pydantic model::

    {
        "status": "ok" | "unavailable",
        "product_id": str,
        "points": [
            {
                "observed_at": date,
                "pack_quantity": float,
                "pack_unit": str,
                "selling_price": float,
                "unit_price": float | None,   # computed selling_price / pack_quantity
                "source_type": str,           # "off" | "cited_public_record"
                "source_citation": str | None,
            },
            ...
        ],
        "total_change": {                     # None unless >= 2 points exist
            "period_start": date,
            "period_end": date,
            "pack_quantity_pct": float | None,
            "unit_price_pct": float | None,
        } | None,
        "message": str | None,                # unavailable message, else None
    }

Design decisions
----------------
* **Chronological order (Req 4.1).** The repository already orders points by
  ``observed_at``; the service additionally sorts defensively so the ordering
  invariant (Correctness Property 9) holds at the service boundary regardless
  of how the rows arrive. The sort is stable, so points sharing a date keep
  their stored order and the result is *non-decreasing* in time.

* **Unit price is computed, not trusted (Req 4.2).** Each point's unit price is
  recomputed here as ``selling_price / pack_quantity`` rather than read from the
  precomputed ``pack_size_history.unit_price`` column, so the identity in
  Correctness Property 10 holds by construction and cannot drift from a stale
  stored value. No rounding is applied: the endpoint/UI rounds for display.

* **Total change across the full period (Req 4.3).** When two or more points
  exist, the totals compare the first (earliest) and last (latest) points:
  ``(last - first) / first * 100`` for both pack quantity and unit price
  (Correctness Property 11). With fewer than two points there is no period to
  measure, so ``total_change`` is ``None``.

* **Attribution on every point (Req 4.4).** Each point carries its
  ``source_type`` (always present) and ``source_citation`` (optional for
  OFF-sourced points) so the timeline can attribute every data point.

* **Graceful absence (Req 4.5).** A product with no recorded history yields an
  ``"unavailable"`` status with an empty ``points`` list and a clear message,
  never an error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.db.models import PackSizeHistory
from app.db.repositories import list_pack_size_history

__all__ = [
    "get_shrinkflation_timeline",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "UNAVAILABLE_MESSAGE",
]

#: Result status vocabulary, mirroring the wider data-layer convention.
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"

#: Message returned when a product has no recorded pack-size history (Req 4.5).
UNAVAILABLE_MESSAGE = "Pack-size history is unavailable for this product."


def _unit_price(selling_price: float, pack_quantity: float) -> Optional[float]:
    """Return ``selling_price / pack_quantity`` (Req 4.2).

    Guards the domain edge where a stored pack quantity is missing or
    non-positive: no meaningful per-unit price exists, so ``None`` is returned
    instead of raising. Real ingested rows always carry a positive pack
    quantity, so this guard is defensive rather than a normal path.
    """

    if pack_quantity is None or pack_quantity <= 0:
        return None
    return selling_price / pack_quantity


def _pct_change(first: Optional[float], last: Optional[float]) -> Optional[float]:
    """Return the total percentage change from ``first`` to ``last`` (Req 4.3).

    Computes ``(last - first) / first * 100``. Returns ``None`` when either
    endpoint is unknown or the baseline is zero (a percentage change against a
    zero baseline is undefined), so the caller can surface "not computable"
    rather than crash on a division by zero.
    """

    if first is None or last is None or first == 0:
        return None
    return (last - first) / first * 100.0


def _build_point(row: PackSizeHistory) -> dict[str, Any]:
    """Shape one recorded pack-size row into a timeline point.

    Carries the pack quantity and selling price (Req 4.1), the freshly computed
    unit price (Req 4.2), the observation date, and the source attribution
    (Req 4.4). ``observed_at`` is left as a :class:`datetime.date`; FastAPI/
    Pydantic serialise it to an ISO-8601 string at the endpoint boundary.
    """

    return {
        "observed_at": row.observed_at,
        "pack_quantity": row.pack_quantity,
        "pack_unit": row.pack_unit,
        "selling_price": row.selling_price,
        "unit_price": _unit_price(row.selling_price, row.pack_quantity),
        "source_type": row.source_type,
        "source_citation": row.source_citation,
    }


def get_shrinkflation_timeline(db: Session, product_id: str) -> dict[str, Any]:
    """Build the pack-size timeline for a product (Req 4.1-4.5).

    Reads the product's recorded pack-size points through the parameterised
    repository helper, returns them in chronological order with a computed unit
    price and source attribution per point, and - when two or more points exist
    - the total percentage change in pack quantity and unit price across the
    full recorded period. A product with no recorded history yields an
    ``"unavailable"`` result with a clear message.

    Args:
        db: An open SQLAlchemy session (injected; no session is opened here).
        product_id: The primary key of the product whose timeline is requested.

    Returns:
        The response ``dict`` documented in the module docstring. ``status`` is
        ``"ok"`` when at least one point exists, otherwise ``"unavailable"``.
    """

    rows: Sequence[PackSizeHistory] = list_pack_size_history(db, product_id)

    # Req 4.5: no recorded history -> a clearly-labelled unavailable result.
    if not rows:
        return {
            "status": STATUS_UNAVAILABLE,
            "product_id": product_id,
            "points": [],
            "total_change": None,
            "message": UNAVAILABLE_MESSAGE,
        }

    # Req 4.1: guarantee non-decreasing chronological order at the service
    # boundary. ``sorted`` is stable, so equal dates keep repository order.
    ordered = sorted(rows, key=lambda row: row.observed_at)
    points = [_build_point(row) for row in ordered]

    # Req 4.3: total change is only defined across a period of two or more
    # points (first -> last).
    total_change: Optional[dict[str, Any]] = None
    if len(points) >= 2:
        first, last = points[0], points[-1]
        total_change = {
            "period_start": first["observed_at"],
            "period_end": last["observed_at"],
            "pack_quantity_pct": _pct_change(
                first["pack_quantity"], last["pack_quantity"]
            ),
            "unit_price_pct": _pct_change(first["unit_price"], last["unit_price"]),
        }

    return {
        "status": STATUS_OK,
        "product_id": product_id,
        "points": points,
        "total_change": total_change,
        "message": None,
    }
