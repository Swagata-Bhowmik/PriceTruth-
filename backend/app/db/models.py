"""SQLAlchemy 2.0 ORM models for the Price Truth data layer.

This module owns the relational schema only: the declarative :class:`Base`
and the six tables from the design's Data Models ER diagram
(``products``, ``category_price_stats``, ``price_snapshots``,
``pack_size_history``, ``platform_prices``, ``category_seasonality``).
Connection plumbing lives in :mod:`app.db.session` and query helpers in
:mod:`app.db.repositories`; keeping models isolated here lets each data-layer
concern be modified independently (Req 17.5).

Design notes:

* Models use the modern SQLAlchemy 2.0 typed style (``Mapped[...]`` +
  :func:`mapped_column`), so column nullability is driven by whether the
  annotation is ``Optional[...]``.
* Column types are the portable SQLAlchemy generics (``String``, ``Text``,
  ``Float``, ``Integer``, ``Date``, ``DateTime``, ``Boolean``) rather than
  Postgres-specific ones, so the same metadata creates cleanly on SQLite
  (used by tests) and PostgreSQL (production).
* ``category`` is stored as a plain indexed string ("FK-ish") rather than a
  hard foreign key: there is no ``categories`` table in scope, and products
  are ingested before per-category statistics are computed, so a real FK
  constraint would impose an ordering the ingestion pipeline does not need.
* ``product_id`` columns *are* real foreign keys onto ``products.id`` and are
  indexed to support the per-product reads the feature services perform.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base shared by every Price Truth ORM model.

    ``Base.metadata`` collects all six tables so a single
    ``Base.metadata.create_all(engine)`` (wired up in task 2.2) provisions the
    whole schema.
    """


class Product(Base):
    """A product resolved from the ingested Kaggle datasets or manual entry.

    The search entry point matches on :attr:`normalized_name`, and every
    feature module keys its data off :attr:`id`. ``category`` links a product
    to its :class:`CategoryPriceStats` and :class:`CategorySeasonality`
    profile by name.
    """

    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String)
    # Lower-cased / punctuation-stripped form used for trigram / ILIKE search.
    normalized_name: Mapped[str] = mapped_column(String, index=True)
    # Crowd-sourced sources frequently omit brand, so it is optional.
    brand: Mapped[Optional[str]] = mapped_column(String)
    # "FK-ish" category label (see module docstring); indexed for per-category
    # joins against stats and seasonality.
    category: Mapped[str] = mapped_column(String(128), index=True)
    # Origin of the row, e.g. "amazon_kaggle", "flipkart_kaggle", or "off".
    source: Mapped[Optional[str]] = mapped_column(String)
    # Identifier in the source system (e.g. ASIN, OFF barcode).
    external_id: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Owned child collections. Relationships are lazy-loaded (the default) so
    # importing a product never eagerly pulls its history.
    price_snapshots: Mapped[List["PriceSnapshot"]] = relationship(
        back_populates="product"
    )
    pack_size_history: Mapped[List["PackSizeHistory"]] = relationship(
        back_populates="product"
    )
    platform_prices: Mapped[List["PlatformPrice"]] = relationship(
        back_populates="product"
    )


class CategoryPriceStats(Base):
    """Per-category price distribution — the backbone of the discount model.

    Because the public data is snapshot-level (no per-product history), the
    discount model consumes these category-level statistics as features
    (Req 2.3). The row is recomputed only when data is re-ingested.
    """

    __tablename__ = "category_price_stats"

    category: Mapped[str] = mapped_column(String(128), primary_key=True)
    mean_price: Mapped[float] = mapped_column(Float)
    median_price: Mapped[float] = mapped_column(Float)
    std_price: Mapped[float] = mapped_column(Float)
    p25_price: Mapped[float] = mapped_column(Float)
    p75_price: Mapped[float] = mapped_column(Float)
    mean_discount_pct: Mapped[float] = mapped_column(Float)
    std_discount_pct: Mapped[float] = mapped_column(Float)
    mean_rating: Mapped[float] = mapped_column(Float)
    mean_rating_count: Mapped[float] = mapped_column(Float)
    sample_size: Mapped[int] = mapped_column(Integer)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class PriceSnapshot(Base):
    """A raw per-product observation ingested from a Kaggle dataset.

    ``captured_at`` is honest about being a point-in-time snapshot date rather
    than one point in a continuous series; several source datasets omit a date
    entirely, which is why it (and the review signals) are optional.
    """

    __tablename__ = "price_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("products.id"), index=True
    )
    platform: Mapped[str] = mapped_column(String)
    reference_price: Mapped[Optional[float]] = mapped_column(Float)
    displayed_price: Mapped[float] = mapped_column(Float)
    discount_pct: Mapped[Optional[float]] = mapped_column(Float)
    rating: Mapped[Optional[float]] = mapped_column(Float)
    rating_count: Mapped[Optional[int]] = mapped_column(Integer)
    captured_at: Mapped[Optional[date]] = mapped_column(Date)
    source_dataset: Mapped[Optional[str]] = mapped_column(String)

    product: Mapped["Product"] = relationship(back_populates="price_snapshots")


class PackSizeHistory(Base):
    """A recorded pack-size point powering the Shrinkflation Timeline (Req 4).

    Every point carries a ``source_type`` (``off`` or ``cited_public_record``)
    so the timeline can display attribution (Req 4.4). ``source_citation`` is
    optional because OFF-sourced points need no external citation string.
    """

    __tablename__ = "pack_size_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("products.id"), index=True
    )
    observed_at: Mapped[date] = mapped_column(Date)
    pack_quantity: Mapped[float] = mapped_column(Float)
    pack_unit: Mapped[str] = mapped_column(String)
    selling_price: Mapped[float] = mapped_column(Float)
    # Unit price at this point (selling_price / pack_quantity), precomputed on
    # ingestion so the timeline reads it directly.
    unit_price: Mapped[float] = mapped_column(Float)
    # "off" | "cited_public_record"
    source_type: Mapped[str] = mapped_column(String)
    source_citation: Mapped[Optional[str]] = mapped_column(Text)

    product: Mapped["Product"] = relationship(back_populates="pack_size_history")


class PlatformPrice(Base):
    """A product's price on one Supported Platform for the aggregator (Req 7)."""

    __tablename__ = "platform_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("products.id"), index=True
    )
    platform: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float)
    # Req 7.3: every entry links to the product page on that platform.
    product_url: Mapped[str] = mapped_column(Text)
    # Req 7.4: nullable so a score is shown only when the listing has one.
    genuineness_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    captured_at: Mapped[Optional[date]] = mapped_column(Date)

    product: Mapped["Product"] = relationship(back_populates="platform_prices")


class CategorySeasonality(Base):
    """A category's monthly seasonal profile for the Buy Timing Signal (Req 6).

    One row per (category, month). ``is_best_window`` flags the month(s) with
    the largest historical reductions, and ``sale_event`` maps a best window to
    an Indian sale-calendar event when one applies.
    """

    __tablename__ = "category_seasonality"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category: Mapped[str] = mapped_column(String(128), index=True)
    month: Mapped[int] = mapped_column(Integer)
    relative_price_index: Mapped[float] = mapped_column(Float)
    is_best_window: Mapped[bool] = mapped_column(Boolean, default=False)
    # Req 6.5: e.g. "Big Billion Days"; nullable because most months map to no
    # named sale event.
    sale_event: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
