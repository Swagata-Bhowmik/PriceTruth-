"""Offline ingestion of the Kaggle / synthetic-fixture CSVs into the core tables.

Task 3.2 of the ``price-truth-platform`` spec. This script loads the Amazon and
Flipkart product CSVs, the cross-platform price CSV, and the curated
pack-size-history CSV, cleans them, and populates four of the six core tables
through the SQLAlchemy models:

* ``amazon_sample.csv`` + ``flipkart_sample.csv`` -> ``products`` + ``price_snapshots``
* ``platform_prices.csv``                          -> ``platform_prices``
* ``pack_size_history.csv``                        -> ``pack_size_history``
  (``unit_price = selling_price / pack_quantity`` is computed per row)

It deliberately does **not** compute ``category_price_stats`` (task 3.3) or
``category_seasonality`` (task 3.4); those are separate tasks that read the rows
this script writes.

Cleaning rules (Req 9.5): currency symbols (``Rs``/``INR``/the rupee sign) and
thousands separators are stripped from prices, ``%`` is stripped from discounts,
every numeric field is coerced to a number, rows whose *selling* price is missing
or non-positive are dropped, discount percentages are clamped to ``[0, 100]``, and
category labels are normalised onto canonical slugs (``electronics/headphones``,
``electronics/tv``, ``grocery/biscuits``, ``grocery/edible-oil``, ``home/kitchen``,
``personal-care``) by a keyword match on the category-tree leaf. ``normalized_name``
is stored as a lower-cased / whitespace-collapsed form of the product name, and
``source`` records the originating dataset (``amazon_kaggle`` / ``flipkart_kaggle``).
The CSV ``product_id`` is used verbatim as ``products.id`` (Req 17.1).

The input directory defaults to the synthetic fixtures under
``data/raw/fixtures/`` and can be pointed at the real Kaggle download with
``--path``. Ingestion is safe to re-run: the four populated tables are cleared
(children first, to respect the foreign keys onto ``products``) before the fresh
rows are written, so re-running yields the same state rather than duplicates.

Usage::

    # default: load the synthetic fixtures into the configured DATABASE_URL
    backend\\venv\\Scripts\\python.exe data\\scripts\\ingest.py

    # point at real data and/or a throwaway database, with a proof report
    backend\\venv\\Scripts\\python.exe data\\scripts\\ingest.py \\
        --path data/raw --database-url sqlite:///./_ingest_test.db --self-check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
# The script lives at ``data/scripts/ingest.py``; the repo root is two levels up
# and the FastAPI application package lives under ``backend/``. Putting
# ``backend/`` on ``sys.path`` lets us import the SQLAlchemy models, session
# factory, and table-creation entry point the app already defines, so the schema
# has a single source of truth (Req 17.5).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

DEFAULT_FIXTURES_DIR = PROJECT_ROOT / "data" / "raw" / "fixtures"

# Source filenames. The synthetic fixtures use these names; the ``_resolve_file``
# fallbacks accept the real Kaggle export names so ``--path data/raw`` works once
# the datasets are downloaded (task 3.5).
AMAZON_CANDIDATES = ("amazon_sample.csv", "amazon.csv")
FLIPKART_CANDIDATES = ("flipkart_sample.csv", "flipkart_com-ecommerce_sample.csv")
PLATFORM_CANDIDATES = ("platform_prices.csv",)
PACK_SIZE_CANDIDATES = ("pack_size_history.csv",)

# Importing the ORM models does *not* open a database connection: models.py only
# defines mapped classes and never imports app.db.session. The engine is created
# lazily when app.db.session is imported, which is deferred until after any
# ``--database-url`` override has been applied (see ``_configure_database``).
from app.db.models import (  # noqa: E402  (import after sys.path setup)
    PackSizeHistory,
    PlatformPrice,
    PriceSnapshot,
    Product,
)


# ---------------------------------------------------------------------------
# Cleaning helpers (Req 9.5)
# ---------------------------------------------------------------------------
_MISSING_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


def to_number(value: object) -> Optional[float]:
    """Coerce a raw cell to a float, or ``None`` when it is not numeric.

    Handles the messy formats present in the Amazon export: the rupee sign, an
    ``Rs``/``Rs.``/``INR`` currency prefix, thousands separators (``1,299``), and
    a trailing ``%``. The strategy is to drop thousands separators and then read
    the first signed numeric token in the string, which naturally ignores any
    surrounding currency or percent decoration. Returns ``None`` for blank or
    unparseable cells so callers can decide whether the field is required.
    """

    if value is None:
        return None
    if isinstance(value, (int, float)):
        # Guard against pandas NaN sentinels arriving as floats.
        return None if pd.isna(value) else float(value)

    text = str(value).strip()
    if text.lower() in _MISSING_TOKENS:
        return None

    # Remove thousands separators first so "1,299" reads as 1299, not 1.
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group())
    except ValueError:  # pragma: no cover - regex guarantees a valid float
        return None


def to_int(value: object) -> Optional[int]:
    """Coerce a raw cell to an int (rounding), or ``None`` when not numeric."""

    number = to_number(value)
    return None if number is None else int(round(number))


def clean_discount_pct(value: object) -> Optional[float]:
    """Parse a discount percentage and clamp it into the closed range [0, 100].

    Strips a trailing ``%`` (via :func:`to_number`) and clamps out-of-range or
    corrupt values rather than dropping the row, since the discount is a
    secondary field (Req 9.5).
    """

    number = to_number(value)
    if number is None:
        return None
    return max(0.0, min(100.0, number))


def normalize_name(name: object) -> str:
    """Return a lower-cased, whitespace-collapsed form of a product name.

    This is what ``products.normalized_name`` stores and what the search service
    later matches against (Req 1.2).
    """

    return re.sub(r"\s+", " ", str(name)).strip().lower()


def parse_date(value: object) -> Optional[date]:
    """Parse an ``observed_at`` cell into a ``date`` (``YYYY-MM-DD``), else ``None``."""

    text = str(value).strip()
    if text.lower() in _MISSING_TOKENS:
        return None
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        parsed = pd.to_datetime(text, errors="coerce")
        return None if pd.isna(parsed) else parsed.date()


# ---------------------------------------------------------------------------
# Category normalisation (Req 9.5)
# ---------------------------------------------------------------------------
# Canonical slugs mapped by a keyword match on the *leaf* of the category tree,
# exactly as documented in data/raw/fixtures/README.md. Rules are ordered
# specific-first; the first slug whose keywords appear in the leaf wins. The leaf
# keywords are disjoint across the six fixture categories, so ordering only
# matters defensively for unusual real-world labels.
_CATEGORY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("electronics/headphones", ("headphone", "earbud", "earphone", "headset")),
    ("electronics/tv", ("smart tv", "smarttv", "television", "tv")),
    ("grocery/biscuits", ("biscuit", "cookie")),
    ("grocery/edible-oil", ("edible oil", "oil")),
    ("home/kitchen", ("kitchen", "cookware")),
    ("personal-care", ("personal care", "personalcare")),
)


def _category_leaf(raw: str) -> str:
    """Extract the leaf label from an Amazon pipe hierarchy or Flipkart tree.

    Amazon categories look like ``Electronics|Audio|Headphones``; Flipkart's
    ``category_tree`` is a JSON-ish list of a ``>>``-delimited path such as
    ``["Electronics >> Audio >> Headphones"]``. Both reduce to their final
    segment (``Headphones``).
    """

    text = str(raw).strip()

    # Flipkart bracketed tree: ["A >> B >> C"].
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list) and parsed:
                text = str(parsed[0])
        except (ValueError, TypeError):
            text = text.strip("[]\"'")

    if ">>" in text:
        segments = [segment.strip() for segment in text.split(">>")]
    elif "|" in text:
        segments = [segment.strip() for segment in text.split("|")]
    else:
        segments = [text.strip()]

    return segments[-1] if segments else text


def normalize_category(raw: object) -> Optional[str]:
    """Map a raw category label onto one of the six canonical slugs.

    Matches keywords against the tree leaf first (the documented rule), then
    falls back to the full label so real Kaggle rows with a differently-named
    leaf still resolve where possible. Returns ``None`` only when nothing
    matches, which never happens for the synthetic fixtures.
    """

    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None

    leaf = _category_leaf(raw).lower().strip()
    for slug, keywords in _CATEGORY_RULES:
        if any(keyword in leaf for keyword in keywords):
            return slug

    full = re.sub(r"[^a-z0-9 ]+", " ", str(raw).lower())
    for slug, keywords in _CATEGORY_RULES:
        if any(keyword in full for keyword in keywords):
            return slug

    return None


# ---------------------------------------------------------------------------
# Ingestion bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class IngestCounts:
    """Row counts and drop reasons captured during a single ingestion run."""

    products: int = 0
    price_snapshots: int = 0
    platform_prices: int = 0
    pack_size_history: int = 0
    # Diagnostics: rows skipped during cleaning.
    dropped_amazon: int = 0
    dropped_flipkart: int = 0
    dropped_platform: int = 0
    dropped_pack: int = 0
    # Child rows referencing an unknown product_id (skipped to preserve the FK).
    orphan_platform: int = 0
    orphan_pack: int = 0
    categories: list[str] = field(default_factory=list)


def _resolve_file(directory: Path, *candidates: str) -> Optional[Path]:
    """Return the first existing candidate filename in ``directory``, else ``None``."""

    for name in candidates:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV as raw strings so the cleaning helpers see the original text.

    ``dtype=str`` + ``keep_default_na=False`` keeps values like ``"1,299"`` and
    ``"₹1,299"`` intact (rather than letting pandas coerce or NaN them) and turns
    empty cells into ``""`` for uniform handling.
    """

    return pd.read_csv(path, dtype=str, keep_default_na=False)


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------
def _load_amazon(
    directory: Path,
    products: dict[str, Product],
    snapshots: list[PriceSnapshot],
    counts: IngestCounts,
) -> None:
    """Load ``amazon_sample.csv`` into products + price snapshots (source Amazon)."""

    path = _resolve_file(directory, *AMAZON_CANDIDATES)
    if path is None:
        print(f"  amazon: no file found in {directory} (skipped)")
        return

    frame = _read_csv(path)
    for row in frame.to_dict("records"):
        product_id = str(row.get("product_id", "")).strip()
        displayed = to_number(row.get("discounted_price"))
        reference = to_number(row.get("actual_price"))

        # Drop rows without a usable selling price (Req 9.5).
        if not product_id or displayed is None or displayed <= 0:
            counts.dropped_amazon += 1
            continue
        # A non-positive reference is not a valid "original" price -> treat as absent.
        if reference is not None and reference <= 0:
            reference = None

        name = str(row.get("product_name", "")).strip()
        if product_id not in products:
            products[product_id] = Product(
                id=product_id,
                name=name,
                normalized_name=normalize_name(name),
                brand=None,  # the Amazon export has no brand column
                category=normalize_category(row.get("category")),
                source="amazon_kaggle",
                external_id=product_id,
            )

        snapshots.append(
            PriceSnapshot(
                product_id=product_id,
                platform="Amazon",
                reference_price=reference,
                displayed_price=displayed,
                discount_pct=clean_discount_pct(row.get("discount_percentage")),
                rating=to_number(row.get("rating")),
                rating_count=to_int(row.get("rating_count")),
                captured_at=None,  # the snapshot export carries no date
                source_dataset="amazon_kaggle",
            )
        )


def _load_flipkart(
    directory: Path,
    products: dict[str, Product],
    snapshots: list[PriceSnapshot],
    counts: IngestCounts,
) -> None:
    """Load ``flipkart_sample.csv`` into products + price snapshots (source Flipkart).

    Flipkart carries no discount column, so the discount percentage is derived
    from the retail (reference) and discounted (selling) prices and clamped to
    ``[0, 100]`` like the Amazon values.
    """

    path = _resolve_file(directory, *FLIPKART_CANDIDATES)
    if path is None:
        print(f"  flipkart: no file found in {directory} (skipped)")
        return

    frame = _read_csv(path)
    for row in frame.to_dict("records"):
        product_id = str(row.get("product_id", "")).strip()
        displayed = to_number(row.get("discounted_price"))
        reference = to_number(row.get("retail_price"))

        if not product_id or displayed is None or displayed <= 0:
            counts.dropped_flipkart += 1
            continue
        if reference is not None and reference <= 0:
            reference = None

        discount_pct: Optional[float] = None
        if reference is not None and reference > 0:
            discount_pct = max(0.0, min(100.0, (reference - displayed) / reference * 100.0))

        name = str(row.get("product_name", "")).strip()
        brand = str(row.get("brand", "")).strip() or None
        if product_id not in products:
            products[product_id] = Product(
                id=product_id,
                name=name,
                normalized_name=normalize_name(name),
                brand=brand,
                category=normalize_category(row.get("category_tree")),
                source="flipkart_kaggle",
                external_id=product_id,
            )

        snapshots.append(
            PriceSnapshot(
                product_id=product_id,
                platform="Flipkart",
                reference_price=reference,
                displayed_price=displayed,
                discount_pct=discount_pct,
                rating=to_number(row.get("overall_rating")),
                rating_count=to_int(row.get("rating_count")),
                captured_at=None,
                source_dataset="flipkart_kaggle",
            )
        )


def _load_platform_prices(
    directory: Path,
    known_product_ids: set[str],
    platform_prices: list[PlatformPrice],
    counts: IngestCounts,
) -> None:
    """Load ``platform_prices.csv`` into the ``platform_prices`` table.

    Rows with a non-positive price are dropped, and rows referencing a product
    that was not ingested are skipped to keep the foreign key onto ``products``
    valid (this never happens for the fixtures, which only reference anchors).
    """

    path = _resolve_file(directory, *PLATFORM_CANDIDATES)
    if path is None:
        print(f"  platform_prices: no file found in {directory} (skipped)")
        return

    frame = _read_csv(path)
    for row in frame.to_dict("records"):
        product_id = str(row.get("product_id", "")).strip()
        price = to_number(row.get("price"))

        if not product_id or price is None or price <= 0:
            counts.dropped_platform += 1
            continue
        if product_id not in known_product_ids:
            counts.orphan_platform += 1
            continue

        platform_prices.append(
            PlatformPrice(
                product_id=product_id,
                platform=str(row.get("platform", "")).strip(),
                price=price,
                product_url=str(row.get("product_url", "")).strip(),
                genuineness_score=None,  # scored later; nullable by design (Req 7.4)
                captured_at=None,
            )
        )


def _load_pack_size_history(
    directory: Path,
    known_product_ids: set[str],
    pack_rows: list[PackSizeHistory],
    counts: IngestCounts,
) -> None:
    """Load ``pack_size_history.csv`` and compute per-row unit price.

    ``unit_price = selling_price / pack_quantity``. Rows are dropped when the
    quantity or price is missing/non-positive (unit price would be undefined) or
    the observed date is unparseable, and skipped when the product is unknown.
    """

    path = _resolve_file(directory, *PACK_SIZE_CANDIDATES)
    if path is None:
        print(f"  pack_size_history: no file found in {directory} (skipped)")
        return

    frame = _read_csv(path)
    for row in frame.to_dict("records"):
        product_id = str(row.get("product_id", "")).strip()
        quantity = to_number(row.get("pack_quantity"))
        selling_price = to_number(row.get("selling_price"))
        observed_at = parse_date(row.get("observed_at"))

        if (
            not product_id
            or quantity is None
            or quantity <= 0
            or selling_price is None
            or selling_price <= 0
            or observed_at is None
        ):
            counts.dropped_pack += 1
            continue
        if product_id not in known_product_ids:
            counts.orphan_pack += 1
            continue

        citation = str(row.get("source_citation", "")).strip() or None
        pack_rows.append(
            PackSizeHistory(
                product_id=product_id,
                observed_at=observed_at,
                pack_quantity=quantity,
                pack_unit=str(row.get("pack_unit", "")).strip(),
                selling_price=selling_price,
                unit_price=selling_price / quantity,
                source_type=str(row.get("source_type", "")).strip(),
                source_citation=citation,
            )
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _clear_core_tables(session) -> None:
    """Delete the four populated tables, children before parents (idempotency).

    ``price_snapshots``, ``platform_prices``, and ``pack_size_history`` all hold a
    foreign key onto ``products``, so they are emptied first. ``category_price_stats``
    and ``category_seasonality`` are intentionally left untouched (tasks 3.3 / 3.4).
    """

    from sqlalchemy import delete

    session.execute(delete(PriceSnapshot))
    session.execute(delete(PlatformPrice))
    session.execute(delete(PackSizeHistory))
    session.execute(delete(Product))
    session.commit()


def run_ingestion(directory: Path, session) -> IngestCounts:
    """Clean and load every source CSV under ``directory`` into ``session``.

    Products are written (and flushed) before the child tables so the foreign
    keys resolve on databases that enforce them (e.g. PostgreSQL).
    """

    counts = IngestCounts()
    _clear_core_tables(session)

    # Products + price snapshots from both marketplaces. A dict keyed by id
    # dedupes products that appear more than once while keeping every snapshot.
    products: dict[str, Product] = {}
    snapshots: list[PriceSnapshot] = []
    _load_amazon(directory, products, snapshots, counts)
    _load_flipkart(directory, products, snapshots, counts)

    session.add_all(products.values())
    session.add_all(snapshots)
    session.flush()  # make product PKs visible before child FKs are inserted

    known_ids = set(products.keys())

    platform_prices: list[PlatformPrice] = []
    _load_platform_prices(directory, known_ids, platform_prices, counts)
    session.add_all(platform_prices)

    pack_rows: list[PackSizeHistory] = []
    _load_pack_size_history(directory, known_ids, pack_rows, counts)
    session.add_all(pack_rows)

    session.commit()

    counts.products = len(products)
    counts.price_snapshots = len(snapshots)
    counts.platform_prices = len(platform_prices)
    counts.pack_size_history = len(pack_rows)
    counts.categories = sorted({p.category for p in products.values() if p.category})
    return counts


def _configure_database(database_url: Optional[str]) -> None:
    """Apply a ``--database-url`` override before the engine is created.

    The engine in :mod:`app.db.session` is built from ``get_settings().DATABASE_URL``
    at import time, so the override must be set in the environment (and the cached
    settings cleared) *before* that module is imported. Callers therefore invoke
    this ahead of importing the session / init_db modules.
    """

    if not database_url:
        return
    os.environ["DATABASE_URL"] = database_url
    try:
        from app.core.config import get_settings

        get_settings.cache_clear()
    except Exception:  # pragma: no cover - config is always importable here
        pass


def ingest(
    path: Optional[os.PathLike | str] = None,
    database_url: Optional[str] = None,
) -> IngestCounts:
    """Programmatic entry point: create tables, then clean-and-load ``path``.

    ``path`` defaults to the synthetic fixtures (``data/raw/fixtures``). Pass a
    directory containing the real Kaggle CSVs for a full-data run. ``database_url``
    optionally overrides ``DATABASE_URL`` for this process.
    """

    directory = Path(path) if path is not None else DEFAULT_FIXTURES_DIR
    _configure_database(database_url)

    # Imported here (not at module top) so any DATABASE_URL override is in effect
    # before the engine is constructed.
    from app.db.init_db import create_all
    from app.db.session import SessionLocal

    create_all()  # idempotent: checkfirst=True skips existing tables

    session = SessionLocal()
    try:
        return run_ingestion(directory, session)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# CLI + reporting
# ---------------------------------------------------------------------------
def _print_report(counts: IngestCounts, directory: Path) -> None:
    """Print inserted row counts per table and any drop/skip diagnostics."""

    print("=" * 60)
    print("PRICE TRUTH - CORE-TABLE INGESTION")
    print("=" * 60)
    print(f"Source directory : {directory}")
    print(f"Database URL     : {os.environ.get('DATABASE_URL', '(default from settings)')}")
    print("-" * 60)
    print("Inserted rows:")
    print(f"  products           : {counts.products}")
    print(f"  price_snapshots    : {counts.price_snapshots}")
    print(f"  platform_prices    : {counts.platform_prices}")
    print(f"  pack_size_history  : {counts.pack_size_history}")
    print("-" * 60)
    print("Normalized categories:")
    for slug in counts.categories:
        print(f"  - {slug}")
    dropped = (
        counts.dropped_amazon
        + counts.dropped_flipkart
        + counts.dropped_platform
        + counts.dropped_pack
    )
    orphans = counts.orphan_platform + counts.orphan_pack
    if dropped or orphans:
        print("-" * 60)
        print("Diagnostics:")
        print(f"  dropped (amazon/flipkart/platform/pack): "
              f"{counts.dropped_amazon}/{counts.dropped_flipkart}/"
              f"{counts.dropped_platform}/{counts.dropped_pack}")
        print(f"  orphan child rows (platform/pack)      : "
              f"{counts.orphan_platform}/{counts.orphan_pack}")
    print("=" * 60)


def _self_check() -> None:
    """Print cleaning proof: messy tokens coerced to numbers + DB categories.

    Demonstrates the Req 9.5 cleaning end-to-end after an ingestion run:
    the pure cleaners on the documented messy tokens, the distinct canonical
    categories persisted to ``products``, and the cleaned numeric fields of the
    known messy Amazon rows (``amz_0003`` = ``₹1,299`` / ``35%`` and
    ``amz_0101`` = ``₹6,930`` / ``75%``) read back from the database.
    """

    print("-" * 60)
    print("SELF-CHECK: cleaning proof (Req 9.5)")
    print("-" * 60)
    samples = ["\u20b91,299", "\u20b96,930", "75%", "35%", "32,210"]
    for token in samples:
        print(f"  to_number({token!r}) -> {to_number(token)}")

    from sqlalchemy import select

    from app.db.session import SessionLocal

    session = SessionLocal()
    try:
        categories = sorted(
            {c for (c,) in session.execute(select(Product.category)).all() if c}
        )
        print(f"  distinct product categories -> {categories}")

        for pid in ("amz_0003", "amz_0101"):
            product = session.get(Product, pid)
            snapshot = session.execute(
                select(PriceSnapshot).where(PriceSnapshot.product_id == pid)
            ).scalars().first()
            if product is None or snapshot is None:
                continue
            print(
                f"  {pid}: category={product.category!r} "
                f"displayed={snapshot.displayed_price} "
                f"reference={snapshot.reference_price} "
                f"discount_pct={snapshot.discount_pct}"
            )
    finally:
        session.close()
    print("-" * 60)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Load and clean the Price Truth source CSVs into the core tables."
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Directory containing the source CSVs (default: data/raw/fixtures).",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="Override DATABASE_URL for this run, e.g. sqlite:///./_ingest_test.db.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="After ingesting, print cleaning proof and persisted categories.",
    )
    args = parser.parse_args(argv)

    directory = Path(args.path) if args.path is not None else DEFAULT_FIXTURES_DIR
    counts = ingest(path=directory, database_url=args.database_url)
    _print_report(counts, directory)
    if args.self_check:
        _self_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
