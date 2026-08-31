"""Buy Timing Signal service - business logic for Requirement 6.

Given a category, this service turns the stored category-level seasonal profile
into a ``buy_now`` / ``wait`` recommendation and shapes it for the buy-timing
endpoint (task 11.3). Like the platform's other feature services it is a thin,
side-effect-free reducer over the data layer:

* it reads the profile with
  :func:`app.db.repositories.list_category_seasonality` (a parameter-bound
  query, Req 18.2) and never issues SQL itself,
* it delegates the seasonal maths (deepest-discount window, sale-calendar
  anchoring) to the pure :mod:`app.ml.seasonality` module, and
* it returns a plain ``dict`` with no framework or HTTP concerns, so the
  property tests (task 11.2, Correctness Properties 16 and 17) and the FastAPI
  endpoint can both call it directly.

Requirement mapping
-------------------
* **Req 6.1** - returns ``buy_now`` or ``wait`` for the category.
* **Req 6.2** - a ``wait`` result carries the seasonal window with the largest
  historical reduction (the deepest-discount window).
* **Req 6.3** - every recommendation is scoped to the category (``level:
  "category"``), never to a single product on a single future date.
* **Req 6.4 / 10.1** - every result carries the :data:`DISCLOSURE` statement
  that the recommendation is category-level and derived from snapshot data.
* **Req 6.5** - the relevant (best) window references the Indian sale-calendar
  event via ``sale_event``.
* **Req 6.6** - a category with no stored profile yields an unavailable result.

Decision rule
-------------
Let ``best`` be the deepest-discount window (lowest ``relative_price_index``)
in the profile and ``current_index`` the index of the current month. The
service recommends ``wait`` only when the best window is *both* still ahead of
the current month *and* materially cheaper than the current month
(``current_index - best_index > MATERIALITY_MARGIN``); otherwise it recommends
``buy_now`` (the shopper is at/near the best window, or the year's deepest
window has already passed and nothing materially cheaper lies ahead).

Restricting ``wait`` to a best window that lies ahead keeps the displayed
window consistent with Correctness Property 17: whenever the service says
``wait``, the window it shows is the profile's largest historical reduction and
that window is a *future* month, never one already behind the shopper.

Result shape
------------
::

    {
        "category": str,
        "available": bool,            # a seasonal profile exists (Req 6.6)
        "level": "category",          # Req 6.3 scoping
        "current_month": int,         # 1-12, the month the call evaluated
        "recommendation": "buy_now" | "wait" | None,
        "best_window": {              # None when unavailable
            "month": int,
            "month_name": str,
            "relative_price_index": float,
            "expected_reduction_pct": float,   # vs the category average
            "sale_event": str | None,          # Indian sale calendar (Req 6.5)
        } | None,
        "disclosure": str,            # always present (Req 6.4, 10.1)
        "message": str,               # human-readable summary / unavailable note
    }
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.db.repositories import list_category_seasonality
from app.ml.seasonality import (
    NEUTRAL_INDEX,
    find_best_window,
    sale_event_for_month,
)

__all__ = [
    "recommend_buy_timing",
    "DISCLOSURE",
    "MESSAGE_UNAVAILABLE",
    "MATERIALITY_MARGIN",
]

# Req 6.4 / 6.3 / 10.1: the honest-limitations statement attached to every
# result. It states both that the recommendation is category-level (not for a
# single product on a single future date) and that it is derived from
# snapshot data rather than a continuous price history.
DISCLOSURE = (
    "This buy-timing recommendation is category-level - it applies to the "
    "product category as a whole, not to an individual product on a single "
    "future date - and is derived from point-in-time snapshot data rather than "
    "a continuous per-product price history."
)

# Req 6.6: shown when a category has no stored seasonal profile.
MESSAGE_UNAVAILABLE = "A timing recommendation is unavailable for this category."

# How much cheaper (in relative-price-index units, where 1.0 == the category
# average) a future window must be than the current month before waiting is
# worthwhile. Reductions at or below this margin are treated as "at/near" the
# current price, so the service recommends buying now.
MATERIALITY_MARGIN = 0.03


def _best_window_payload(best: Any) -> dict[str, Any]:
    """Shape the deepest-discount window for the response (Req 6.2, 6.5)."""

    month = int(best.month)
    relative_price_index = float(best.relative_price_index)
    # Prefer the stored sale event; fall back to the calendar mapping so the
    # window still references the Indian sale calendar (Req 6.5) even if the
    # persisted row left it blank.
    sale_event = best.sale_event or sale_event_for_month(month)
    # Reduction relative to the category average (index 1.0), never negative.
    expected_reduction_pct = round(max(0.0, 1.0 - relative_price_index) * 100, 1)

    return {
        "month": month,
        "month_name": calendar.month_name[month],
        "relative_price_index": relative_price_index,
        "expected_reduction_pct": expected_reduction_pct,
        "sale_event": sale_event,
    }


def _wait_message(category: str, window: dict[str, Any]) -> str:
    event = f" ({window['sale_event']})" if window["sale_event"] else ""
    return (
        f"Prices in the {category} category have historically been lowest "
        f"around {window['month_name']}{event}. Consider waiting for that "
        f"window rather than buying now."
    )


def _buy_now_message(
    category: str, window: dict[str, Any], current_month: int
) -> str:
    if window["month"] == current_month:
        event = f" ({window['sale_event']})" if window["sale_event"] else ""
        return (
            f"The {category} category is in its historically lowest-price "
            f"window ({window['month_name']}{event}); buying now is reasonable."
        )
    return (
        f"No materially cheaper seasonal window lies ahead for the {category} "
        f"category this year; buying now is reasonable."
    )


def recommend_buy_timing(
    db: Session,
    category: str,
    current_month: Optional[int] = None,
) -> dict[str, Any]:
    """Recommend buying now or waiting for a category (Requirement 6).

    Args:
        db: An open SQLAlchemy session.
        category: The product category to evaluate.
        current_month: The month (1-12) to evaluate against. Defaults to the
            current calendar month; accepted explicitly for testability.

    Returns:
        A plain dict (see the module docstring for the full shape) carrying the
        recommendation, the deepest-discount window, and the category-level /
        snapshot-data disclosure. When no seasonal profile exists for the
        category, an unavailable result is returned instead (Req 6.6).

    Raises:
        ValueError: If ``current_month`` is outside 1-12.
    """

    if current_month is None:
        current_month = date.today().month
    current_month = int(current_month)
    if not 1 <= current_month <= 12:
        raise ValueError(
            f"current_month must be between 1 and 12, got {current_month}"
        )

    rows = list_category_seasonality(db, category)

    # Req 6.6: no seasonal profile exists for the category. The disclosure is
    # still attached - the platform's category-level/snapshot limitation holds
    # regardless of whether a recommendation could be produced.
    if not rows:
        return {
            "category": category,
            "available": False,
            "level": "category",
            "current_month": current_month,
            "recommendation": None,
            "best_window": None,
            "disclosure": DISCLOSURE,
            "message": MESSAGE_UNAVAILABLE,
        }

    # Deepest-discount window across the whole profile (Req 6.2). find_best_window
    # reads the ORM rows directly and returns the winning row.
    best = find_best_window(rows)
    window = _best_window_payload(best)

    # The current month's index; default to the neutral (average) index if the
    # profile happens not to include the current month.
    current_index = next(
        (
            float(row.relative_price_index)
            for row in rows
            if int(row.month) == current_month
        ),
        NEUTRAL_INDEX,
    )

    best_is_ahead = window["month"] > current_month
    materially_cheaper = (
        current_index - window["relative_price_index"]
    ) > MATERIALITY_MARGIN

    if best_is_ahead and materially_cheaper:
        recommendation = "wait"
        message = _wait_message(category, window)
    else:
        recommendation = "buy_now"
        message = _buy_now_message(category, window, current_month)

    return {
        "category": category,
        "available": True,
        "level": "category",
        "current_month": current_month,
        "recommendation": recommendation,
        "best_window": window,
        "disclosure": DISCLOSURE,
        "message": message,
    }
