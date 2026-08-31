"""Cross-Platform Aggregator service - business logic for Requirement 7.

This module compares a product's price across the platform listings stored in
``platform_prices`` and shapes the result for the cross-platform endpoint
(task 10.3). It is a thin, side-effect-free reducer over the data layer:

* it reads rows with :func:`app.db.repositories.list_platform_prices` (a
  parameter-bound query, Req 18.2) and never issues SQL itself, and
* it returns a plain ``dict`` with no framework or HTTP concerns, so the
  property tests (task 10.2, Correctness Properties 18 and 19) can call it
  directly and the FastAPI endpoint can wrap it behind a response schema.

Requirement mapping
-------------------
* **Req 7.1** - one entry per Supported Platform that has data, each carrying
  its available price.
* **Req 7.2** - when two or more platforms have data, the lowest-price entry is
  marked ``best_deal``.
* **Req 7.3** - every entry includes ``product_url``, the link to that
  platform's product page.
* **Req 7.4** - an entry exposes ``genuineness_score`` *only when* the
  underlying listing has one; a listing without a score omits the key entirely
  (the column is nullable), so the presence of the key mirrors the data
  exactly (the "if and only if" half of Property 18).
* **Req 7.5** - when exactly one platform has data, the single price is
  returned with a message stating that no comparison is available and nothing
  is marked as the best deal.
* **Req 7.6** - when no platform has data, an unavailable message is returned
  with an empty platform list.

Result shape
------------
::

    {
        "product_id": str,
        "available": bool,             # any platform has data (Req 7.6)
        "comparison_available": bool,  # two or more platforms have data (Req 7.5)
        "best_deal_platform": str | None,
        "platforms": [
            {
                "platform": str,
                "price": float,
                "product_url": str,
                "genuineness_score": int,  # present only when the listing has one
                "best_deal": True,         # present only on the winning entry
            },
            ...
        ],
        "message": str | None,
    }

``platforms`` is ordered cheapest-first so the best deal reads first for the
endpoint/frontend; the ordering is fully deterministic (ties broken by the
Supported-Platform display order, then platform name). The lone winning entry
carries ``best_deal: True`` (mirroring ``unit_price_service``'s ``best_value``
convention); non-winning entries omit the key, and no entry is marked when a
comparison is not possible.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.db.repositories import list_platform_prices

__all__ = [
    "aggregate_cross_platform",
    "SUPPORTED_PLATFORMS",
    "MESSAGE_NO_DATA",
    "MESSAGE_SINGLE_PLATFORM",
]

# The Supported Platforms compared by the aggregator, in display order
# (requirements.md glossary: "Supported Platform"). Used purely to give ties a
# stable, meaningful order; it does not filter the data, because the ingestion
# pipeline only ever writes Supported-Platform rows into ``platform_prices``.
SUPPORTED_PLATFORMS: tuple[str, ...] = (
    "Amazon",
    "Flipkart",
    "Croma",
    "Tata CLiQ",
    "Reliance Digital",
)

_PLATFORM_ORDER: dict[str, int] = {
    name: index for index, name in enumerate(SUPPORTED_PLATFORMS)
}

# Module-specific unavailable messages (Req 7.5, 7.6). The no-data message
# matches the design's structured-error example so the copy is consistent
# wherever it surfaces.
MESSAGE_NO_DATA = "Cross-platform data is unavailable for this product."
MESSAGE_SINGLE_PLATFORM = (
    "Price data is available on only one platform, "
    "so no cross-platform comparison is available."
)


def _platform_rank(platform: str) -> int:
    """Return the display-order rank of ``platform``.

    Known Supported Platforms sort by their glossary order; any unrecognised
    platform sorts after them so the ordering stays deterministic without
    dropping data.
    """

    return _PLATFORM_ORDER.get(platform, len(SUPPORTED_PLATFORMS))


def aggregate_cross_platform(db: Session, product_id: str) -> dict[str, Any]:
    """Compare a product's price across Supported Platforms (Requirement 7).

    Args:
        db: An open SQLAlchemy session.
        product_id: The primary key of the product to aggregate.

    Returns:
        A plain dict (see the module docstring for the full shape) describing
        the per-platform prices, the best deal when a comparison is possible,
        and an unavailable/no-comparison message otherwise.
    """

    rows = list_platform_prices(db, product_id)

    # Req 7.1/7.3/7.4: one entry per listing, always with a product link, and a
    # genuineness score only when the listing actually has one.
    platforms: list[dict[str, Any]] = []
    for row in rows:
        entry: dict[str, Any] = {
            "platform": row.platform,
            "price": row.price,
            "product_url": row.product_url,
        }
        if row.genuineness_score is not None:
            entry["genuineness_score"] = row.genuineness_score
        platforms.append(entry)

    # Req 7.6: no Supported Platform has data.
    if not platforms:
        return {
            "product_id": product_id,
            "available": False,
            "comparison_available": False,
            "best_deal_platform": None,
            "platforms": [],
            "message": MESSAGE_NO_DATA,
        }

    # Deterministic, cheapest-first ordering: price ascending, ties broken by
    # Supported-Platform display order and then platform name.
    platforms.sort(
        key=lambda item: (
            item["price"],
            _platform_rank(item["platform"]),
            item["platform"],
        )
    )

    # Req 7.5: a single platform cannot be compared.
    if len(platforms) == 1:
        return {
            "product_id": product_id,
            "available": True,
            "comparison_available": False,
            "best_deal_platform": None,
            "platforms": platforms,
            "message": MESSAGE_SINGLE_PLATFORM,
        }

    # Req 7.2: with two or more platforms, the cheapest is the best deal. After
    # the ascending sort the first entry holds the minimum price, so marking it
    # keeps the marked entry and the leading entry the same one.
    best = platforms[0]
    best["best_deal"] = True

    return {
        "product_id": product_id,
        "available": True,
        "comparison_available": True,
        "best_deal_platform": best["platform"],
        "platforms": platforms,
        "message": None,
    }
