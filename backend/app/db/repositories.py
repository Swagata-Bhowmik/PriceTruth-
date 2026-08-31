"""Parameter-bound read helpers for the Price Truth data layer.

This module owns *query helpers* only. The ORM schema lives in
:mod:`app.db.models` and connection plumbing in :mod:`app.db.session`;
isolating the queries here lets the read patterns each feature service needs
be changed without touching model definitions or session management
(Req 17.5).

Security note (Req 18.2): every helper builds its query with SQLAlchemy 2.0
:func:`~sqlalchemy.select` and expresses filters through ORM column
comparisons (``==``, :meth:`~sqlalchemy.orm.InstrumentedAttribute.ilike`).
SQLAlchemy renders the caller-supplied values as *bound parameters*, never as
inline SQL text, so user input is passed to the driver out-of-band and can
never be concatenated into a statement. No helper here interpolates a value
into a query string.

The helpers are intentionally thin and side-effect free: each takes an open
:class:`~sqlalchemy.orm.Session` plus typed arguments and returns ORM
instances (or ``None``). Caching, normalization policy, and response shaping
belong to the service layer, not here.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    CategoryPriceStats,
    CategorySeasonality,
    PackSizeHistory,
    PlatformPrice,
    Product,
)

__all__ = [
    "get_product_by_id",
    "search_products_by_name",
    "get_category_price_stats",
    "list_pack_size_history",
    "list_platform_prices",
    "list_category_seasonality",
]


def get_product_by_id(db: Session, product_id: str) -> Product | None:
    """Return the product with primary key ``product_id``, or ``None``.

    ``product_id`` is bound as a parameter in the ``WHERE products.id = ?``
    clause (Req 18.2). Used by every feature module to resolve a selected
    product before reading its owned data.
    """

    stmt = select(Product).where(Product.id == product_id)
    return db.execute(stmt).scalar_one_or_none()


def search_products_by_name(
    db: Session, query: str, limit: int = 20
) -> Sequence[Product]:
    """Return products whose normalized name contains ``query`` (Req 1.2).

    Matching is a case-insensitive substring test against
    :attr:`Product.normalized_name` via ``ILIKE`` (SQLAlchemy renders this as
    a ``lower(...) LIKE lower(...)`` comparison on backends without a native
    ``ILIKE``, so the same helper works on both PostgreSQL and the SQLite used
    by tests). The search term and the ``LIKE`` wildcards are combined into a
    single value that is passed as a bound parameter, so nothing the caller
    supplies is concatenated into SQL (Req 18.2).

    The query is only stripped of surrounding whitespace here; fuller
    normalization (punctuation, casing) is the search service's concern. Each
    returned :class:`Product` carries the name, brand, and category the search
    results view displays. Results are ordered by normalized name for stable
    output and capped by ``limit``.
    """

    pattern = f"%{query.strip()}%"
    stmt = (
        select(Product)
        .where(Product.normalized_name.ilike(pattern))
        .order_by(Product.normalized_name)
        .limit(limit)
    )
    return db.execute(stmt).scalars().all()


def get_category_price_stats(
    db: Session, category: str
) -> CategoryPriceStats | None:
    """Return the price-distribution statistics for ``category``, or ``None``.

    These per-category statistics are the discount model's input features
    (Req 2.3). ``category`` is bound as a parameter. Returning ``None`` lets
    the discount service fall back to its limited-verification path (Req 2.6).
    """

    stmt = select(CategoryPriceStats).where(
        CategoryPriceStats.category == category
    )
    return db.execute(stmt).scalar_one_or_none()


def list_pack_size_history(
    db: Session, product_id: str
) -> Sequence[PackSizeHistory]:
    """Return a product's pack-size points in chronological order (Req 4.1).

    Ordered by ``observed_at`` ascending so the Shrinkflation Timeline can read
    the sequence directly. ``product_id`` is bound as a parameter. Returns an
    empty sequence when the product has no recorded history (Req 4.5).
    """

    stmt = (
        select(PackSizeHistory)
        .where(PackSizeHistory.product_id == product_id)
        .order_by(PackSizeHistory.observed_at)
    )
    return db.execute(stmt).scalars().all()


def list_platform_prices(
    db: Session, product_id: str
) -> Sequence[PlatformPrice]:
    """Return every stored platform price for ``product_id`` (Req 7.1).

    Feeds the Cross-Platform Aggregator, which picks the best deal and shows
    each listing's link and optional genuineness score. ``product_id`` is
    bound as a parameter; an empty sequence means no platform data exists
    (Req 7.6).
    """

    stmt = select(PlatformPrice).where(PlatformPrice.product_id == product_id)
    return db.execute(stmt).scalars().all()


def list_category_seasonality(
    db: Session, category: str
) -> Sequence[CategorySeasonality]:
    """Return a category's monthly seasonal profile ordered by month (Req 6.x).

    Powers the Buy Timing Signal's category-level recommendation. ``category``
    is bound as a parameter; the rows are ordered by ``month`` so the twelve
    monthly points read in calendar order. An empty sequence means no seasonal
    pattern is available for the category (Req 6.6).
    """

    stmt = (
        select(CategorySeasonality)
        .where(CategorySeasonality.category == category)
        .order_by(CategorySeasonality.month)
    )
    return db.execute(stmt).scalars().all()
