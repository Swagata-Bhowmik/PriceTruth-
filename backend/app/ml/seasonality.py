"""Category-level seasonality logic for the Buy Timing Signal - Req 6.

Because the public data is *snapshot-level* (point-in-time records, not a
continuous per-product daily series), seasonality cannot be expressed per SKU.
It is therefore expressed at the **category level** (Req 6.3): a category's
twelve-month profile of a ``relative_price_index`` (1.0 == the category's
average price; values below 1.0 mark cheaper, deeper-discount months).

This module owns two things:

* the **Indian sale-calendar** mapping (month -> named sale event) that anchors
  the profile to the events shoppers actually plan around - Big Billion Days,
  Diwali, Republic Day Sale and Prime Day (Req 6.5); and
* the **pure** logic that turns whatever monthly signal is available in the
  ingested data into a persistable monthly profile, falling back to a
  sale-calendar *prior* when no temporal signal exists.

Everything here is side-effect free and importable: task 3.4 calls
:func:`build_category_profile` to populate the ``category_seasonality`` table,
and :mod:`app.services.buy_timing_service` calls :func:`find_best_window` (and
:func:`sale_event_for_month`) to shape a recommendation. Keeping it free of any
DB, HTTP, Prophet/statsmodels or framework dependency lets both the persistence
step and the property/unit tests exercise it directly.

Design note (why a *prior* rather than a fit): a real Prophet/statsmodels
seasonal fit needs a category-level monthly index, which only exists once
enough dated snapshots are ingested. Until then - and for categories that never
accrue a temporal signal - the honest, still-useful behaviour is to lean on the
documented Indian sale calendar. :func:`build_category_profile` *combines* the
two: months that have a real normalized signal use it, and the remaining months
fall back to the calendar prior, so the deepest known sale (Big Billion Days)
remains the default best window when data is silent.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Optional, TypedDict

__all__ = [
    "INDIAN_SALE_CALENDAR",
    "NAMED_SALE_EVENTS",
    "NEUTRAL_INDEX",
    "MonthlySeasonPoint",
    "sale_event_for_month",
    "build_category_profile",
    "find_best_window",
]

# ---------------------------------------------------------------------------
# Indian sale calendar (Req 6.5)
# ---------------------------------------------------------------------------
# Month number (1-12) -> the named sale event that anchors that month. The
# ``category_seasonality.sale_event`` column stores a single string per
# (category, month), so each month maps to at most one event. Diwali spans
# late October/November; October is anchored to Big Billion Days (the deepest
# festive-season sale) and November carries Diwali, so all four named events
# from Req 6.5 are represented exactly once.
INDIAN_SALE_CALENDAR: dict[int, str] = {
    1: "Republic Day Sale",   # January
    7: "Prime Day",           # July
    10: "Big Billion Days",   # October (festive season)
    11: "Diwali",             # October/November
}

# The four events named in Req 6.5, exposed as a set for callers/tests that
# only need to assert coverage rather than the month anchoring.
NAMED_SALE_EVENTS: frozenset[str] = frozenset(INDIAN_SALE_CALENDAR.values())

# A relative price index of 1.0 means "at the category's average price"; a
# month with no sale event and no temporal signal sits here.
NEUTRAL_INDEX: float = 1.0

# The sale-calendar *prior*: the relative price index assumed for each sale
# month when no temporal signal is available. Depths encode the well-known
# ordering of Indian sale depth - Big Billion Days is the deepest, followed by
# Diwali, Prime Day, then the Republic Day Sale. Keys mirror
# ``INDIAN_SALE_CALENDAR`` exactly; this invariant is asserted at import so the
# two never drift apart.
_SALE_PRIOR_INDEX: dict[int, float] = {
    1: 0.90,   # Republic Day Sale
    7: 0.88,   # Prime Day
    10: 0.80,  # Big Billion Days (deepest discount of the year)
    11: 0.85,  # Diwali
}

assert set(_SALE_PRIOR_INDEX) == set(INDIAN_SALE_CALENDAR), (
    "sale-calendar prior months must match the named-event months"
)

# A monthly signal is only trusted to override the prior when at least this
# many months carry a usable value; a single point normalises to 1.0 and
# carries no relative information.
_MIN_SIGNAL_MONTHS: int = 2


class MonthlySeasonPoint(TypedDict):
    """One row of a category's monthly profile.

    The keys match the ``category_seasonality`` columns 1:1 so task 3.4 can
    persist a point directly, e.g.
    ``CategorySeasonality(category=category, **point)``.
    """

    month: int
    relative_price_index: float
    is_best_window: bool
    sale_event: Optional[str]


def sale_event_for_month(month: int) -> Optional[str]:
    """Return the Indian sale event anchored to ``month``, or ``None`` (Req 6.5).

    Used both when building a profile and when a recommendation needs to name
    the sale event on a best window whose stored ``sale_event`` is absent.
    """

    return INDIAN_SALE_CALENDAR.get(month)


def _is_positive_finite(value: Any) -> bool:
    """True when ``value`` coerces to a strictly positive, finite float."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _normalize_signal(
    monthly_signal: Optional[Mapping[int, float]],
) -> dict[int, float]:
    """Turn a raw monthly price signal into a mean-normalized index.

    ``monthly_signal`` maps a month (1-12) to a price *level* for that month
    (e.g. the average observed price, higher == more expensive). Each value is
    divided by the mean across the provided months, yielding an index centred
    on 1.0 that is directly comparable with the sale-calendar prior.

    Months whose value is missing, non-numeric or non-positive are dropped. If
    fewer than :data:`_MIN_SIGNAL_MONTHS` usable months remain, an empty mapping
    is returned so the caller falls back entirely to the prior.
    """

    if not monthly_signal:
        return {}

    valid: dict[int, float] = {
        int(month): float(value)
        for month, value in monthly_signal.items()
        if isinstance(month, int)
        and 1 <= month <= 12
        and _is_positive_finite(value)
    }
    if len(valid) < _MIN_SIGNAL_MONTHS:
        return {}

    mean = sum(valid.values()) / len(valid)
    if mean <= 0:
        return {}

    return {month: value / mean for month, value in valid.items()}


def build_category_profile(
    monthly_signal: Optional[Mapping[int, float]] = None,
) -> list[MonthlySeasonPoint]:
    """Build a category's twelve-month seasonal profile (pure).

    This is the logic task 3.4 calls to persist ``category_seasonality``.

    Args:
        monthly_signal: Optional mapping of month (1-12) to an observed price
            level for that month. When two or more usable months are present it
            is mean-normalized and used for those months; every other month
            (and the whole profile when no usable signal exists) falls back to
            the Indian sale-calendar prior (Req 6.5).

    Returns:
        A list of twelve :class:`MonthlySeasonPoint` dicts ordered January to
        December. Each carries the ``relative_price_index``, the ``sale_event``
        anchored to that month (or ``None``), and ``is_best_window`` set on the
        month(s) with the lowest index (the deepest discount / largest
        historical reduction, Req 6.2).
    """

    signal_index = _normalize_signal(monthly_signal)

    points: list[MonthlySeasonPoint] = []
    for month in range(1, 13):
        if month in signal_index:
            # Real (normalized) signal wins over the prior for this month.
            relative_price_index = round(signal_index[month], 6)
        else:
            # No signal for this month: lean on the sale-calendar prior, which
            # is NEUTRAL_INDEX (1.0) for ordinary months.
            relative_price_index = _SALE_PRIOR_INDEX.get(month, NEUTRAL_INDEX)

        points.append(
            MonthlySeasonPoint(
                month=month,
                relative_price_index=relative_price_index,
                is_best_window=False,
                sale_event=sale_event_for_month(month),
            )
        )

    # Mark the deepest-discount window(s): the lowest relative price index.
    best_index = min(point["relative_price_index"] for point in points)
    for point in points:
        if point["relative_price_index"] == best_index:
            point["is_best_window"] = True

    return points


def _relative_index(point: Any) -> float:
    """Read ``relative_price_index`` from a dict point or an ORM row."""

    if isinstance(point, Mapping):
        return float(point["relative_price_index"])
    return float(point.relative_price_index)


def _month(point: Any) -> int:
    """Read ``month`` from a dict point or an ORM row."""

    if isinstance(point, Mapping):
        return int(point["month"])
    return int(point.month)


def find_best_window(profile: Sequence[Any]) -> Optional[Any]:
    """Return the deepest-discount window in ``profile`` (Req 6.2).

    The best window is the point with the lowest ``relative_price_index``; ties
    are broken by the earliest calendar month so the result is deterministic.
    ``profile`` may be the dict points produced by
    :func:`build_category_profile` *or* the ``CategorySeasonality`` ORM rows
    returned by ``list_category_seasonality`` - the same point object is
    returned unchanged, so the caller keeps full access to its fields.

    Returns ``None`` for an empty profile.
    """

    points = list(profile)
    if not points:
        return None
    return min(points, key=lambda point: (_relative_index(point), _month(point)))
