"""Unit tests for the ORM models and repository helpers (task 2.3).

These tests exercise :mod:`app.db.repositories` against a temporary in-memory
SQLite database seeded from :mod:`app.db.models`. They verify two things:

* **Shape** - each helper returns the expected instance(s) (or ``None`` /
  an empty sequence) for its documented inputs, including nullable columns
  (``platform_prices.genuineness_score``, ``category_seasonality.sale_event``)
  and the two ordered reads (pack-size history by ``observed_at``, seasonality
  by ``month``).
* **Parameter binding (Req 18.2)** - user-supplied values reach the database
  as *bound parameters*, never concatenated into the SQL text. This is proven
  three ways: by compiling a statement and confirming a placeholder (not the
  literal) appears, by capturing the actual SQL the driver executes for a real
  repository call, and by feeding an injection-like string and confirming it is
  treated as inert data (no rows, no error, table intact).

The engine uses a ``StaticPool`` so the whole test shares the single
connection backing the ``sqlite://`` in-memory database.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import (
    Base,
    CategoryPriceStats,
    CategorySeasonality,
    PackSizeHistory,
    PlatformPrice,
    Product,
)
from app.db.repositories import (
    get_category_price_stats,
    get_product_by_id,
    list_category_seasonality,
    list_pack_size_history,
    list_platform_prices,
    search_products_by_name,
)

# An injection-like payload used to prove user input is bound, not concatenated
# (Req 18.2). If it were ever interpolated into the SQL text it would terminate
# the string literal and drop the products table.
INJECTION = "x'; DROP TABLE products;--"


def _seed(session: Session) -> None:
    """Populate every table the repository helpers read."""
    session.add_all(
        [
            Product(
                id="amz_amul_butter",
                name="Amul Butter 500g",
                normalized_name="amul butter 500g",
                brand="Amul",
                category="grocery/dairy",
                source="amazon_kaggle",
            ),
            Product(
                id="amz_nutella",
                name="Nutella Hazelnut Spread 750g",
                normalized_name="nutella hazelnut spread 750g",
                brand="Ferrero",
                category="grocery/spreads",
                source="amazon_kaggle",
            ),
            Product(
                id="fk_boat_rockerz",
                name="boAt Rockerz 450 Bluetooth Headphones",
                normalized_name="boat rockerz 450 bluetooth headphones",
                brand=None,  # exercises the nullable brand column
                category="electronics/headphones",
                source="flipkart_kaggle",
            ),
        ]
    )

    session.add(
        CategoryPriceStats(
            category="grocery/dairy",
            mean_price=260.0,
            median_price=250.0,
            std_price=30.0,
            p25_price=240.0,
            p75_price=280.0,
            mean_discount_pct=12.5,
            std_discount_pct=6.0,
            mean_rating=4.2,
            mean_rating_count=1500.0,
            sample_size=320,
        )
    )

    # Pack-size points inserted OUT of chronological order so the helper's
    # ORDER BY observed_at is actually exercised (not insertion order).
    session.add_all(
        [
            PackSizeHistory(
                product_id="amz_amul_butter",
                observed_at=date(2021, 6, 1),
                pack_quantity=500.0,
                pack_unit="g",
                selling_price=250.0,
                unit_price=0.5,
                source_type="off",
            ),
            PackSizeHistory(
                product_id="amz_amul_butter",
                observed_at=date(2019, 1, 1),
                pack_quantity=550.0,
                pack_unit="g",
                selling_price=250.0,
                unit_price=250.0 / 550.0,
                source_type="cited_public_record",
                source_citation="Company annual report 2019",
            ),
            PackSizeHistory(
                product_id="amz_amul_butter",
                observed_at=date(2023, 3, 15),
                pack_quantity=450.0,
                pack_unit="g",
                selling_price=250.0,
                unit_price=250.0 / 450.0,
                source_type="off",
            ),
        ]
    )

    # Two platform prices for the same product: one carries a genuineness
    # score, the other leaves it NULL (Req 7.4 / nullable column). A third row
    # for a different product guards the product_id filter.
    session.add_all(
        [
            PlatformPrice(
                product_id="amz_amul_butter",
                platform="amazon",
                price=250.0,
                product_url="https://example.com/amazon/amul-butter",
                genuineness_score=88,
            ),
            PlatformPrice(
                product_id="amz_amul_butter",
                platform="flipkart",
                price=245.0,
                product_url="https://example.com/flipkart/amul-butter",
                genuineness_score=None,
            ),
            PlatformPrice(
                product_id="amz_nutella",
                platform="amazon",
                price=699.0,
                product_url="https://example.com/amazon/nutella",
                genuineness_score=95,
            ),
        ]
    )

    # Seasonality rows inserted out of month order; one maps to a named sale
    # event, the others leave sale_event NULL (nullable column).
    session.add_all(
        [
            CategorySeasonality(
                category="grocery/dairy",
                month=10,
                relative_price_index=0.82,
                is_best_window=True,
                sale_event="Big Billion Days",
            ),
            CategorySeasonality(
                category="grocery/dairy",
                month=1,
                relative_price_index=1.05,
                is_best_window=False,
                sale_event=None,
            ),
            CategorySeasonality(
                category="grocery/dairy",
                month=5,
                relative_price_index=0.98,
                is_best_window=False,
                sale_event=None,
            ),
        ]
    )

    session.commit()


@pytest.fixture()
def engine():
    """Yield an in-memory SQLite engine with the full schema created."""
    eng = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture()
def db_session(engine):
    """Yield a session backed by the seeded in-memory database."""
    testing_session = sessionmaker(
        bind=engine, class_=Session, expire_on_commit=False
    )
    session = testing_session()
    _seed(session)
    try:
        yield session
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# get_product_by_id
# --------------------------------------------------------------------------- #
def test_get_product_by_id_returns_seeded_product(db_session):
    product = get_product_by_id(db_session, "amz_amul_butter")

    assert product is not None
    assert isinstance(product, Product)
    assert product.id == "amz_amul_butter"
    assert product.name == "Amul Butter 500g"
    assert product.category == "grocery/dairy"


def test_get_product_by_id_returns_none_for_unknown_id(db_session):
    assert get_product_by_id(db_session, "does_not_exist") is None


# --------------------------------------------------------------------------- #
# search_products_by_name
# --------------------------------------------------------------------------- #
def test_search_matches_case_insensitively_on_normalized_name(db_session):
    """An upper-case query still matches the lower-cased normalized_name."""
    results = search_products_by_name(db_session, "BUTTER")

    assert [p.id for p in results] == ["amz_amul_butter"]


def test_search_returns_rows_bearing_name_brand_category(db_session):
    results = search_products_by_name(db_session, "nutella")

    assert len(results) == 1
    match = results[0]
    assert match.name == "Nutella Hazelnut Spread 750g"
    assert match.brand == "Ferrero"
    assert match.category == "grocery/spreads"


def test_search_returns_row_with_null_brand(db_session):
    """A match with a NULL brand is still returned with name/category."""
    results = search_products_by_name(db_session, "rockerz")

    assert len(results) == 1
    match = results[0]
    assert match.brand is None
    assert match.name == "boAt Rockerz 450 Bluetooth Headphones"
    assert match.category == "electronics/headphones"


def test_search_is_empty_for_no_match(db_session):
    assert list(search_products_by_name(db_session, "zzz-no-such-product")) == []


def test_search_respects_limit(db_session):
    """'a' matches all three seeded products; limit caps the result set."""
    unlimited = search_products_by_name(db_session, "a")
    limited = search_products_by_name(db_session, "a", limit=1)

    assert len(unlimited) == 3
    assert len(limited) == 1
    # Ordered by normalized_name, so the first hit is the alphabetical minimum.
    assert limited[0].id == "amz_amul_butter"


# --------------------------------------------------------------------------- #
# get_category_price_stats
# --------------------------------------------------------------------------- #
def test_get_category_price_stats_returns_seeded_stats(db_session):
    stats = get_category_price_stats(db_session, "grocery/dairy")

    assert stats is not None
    assert isinstance(stats, CategoryPriceStats)
    assert stats.category == "grocery/dairy"
    assert stats.median_price == 250.0
    assert stats.sample_size == 320


def test_get_category_price_stats_returns_none_for_unknown_category(db_session):
    assert get_category_price_stats(db_session, "no/such/category") is None


# --------------------------------------------------------------------------- #
# list_pack_size_history
# --------------------------------------------------------------------------- #
def test_list_pack_size_history_ordered_by_observed_at(db_session):
    history = list_pack_size_history(db_session, "amz_amul_butter")

    observed = [row.observed_at for row in history]
    assert observed == [date(2019, 1, 1), date(2021, 6, 1), date(2023, 3, 15)]
    # Ascending order regardless of insertion order.
    assert observed == sorted(observed)
    # Attribution rides along with each point (Req 4.4).
    assert history[0].source_type == "cited_public_record"
    assert history[0].source_citation == "Company annual report 2019"
    assert history[1].source_type == "off"


def test_list_pack_size_history_empty_for_product_without_history(db_session):
    assert list(list_pack_size_history(db_session, "amz_nutella")) == []


# --------------------------------------------------------------------------- #
# list_platform_prices
# --------------------------------------------------------------------------- #
def test_list_platform_prices_includes_nullable_genuineness_score(db_session):
    prices = list_platform_prices(db_session, "amz_amul_butter")

    # Filtered to this product only (the nutella row is excluded).
    assert len(prices) == 2
    by_platform = {row.platform: row for row in prices}
    assert set(by_platform) == {"amazon", "flipkart"}

    # One listing carries a score, the other leaves it NULL.
    assert by_platform["amazon"].genuineness_score == 88
    assert by_platform["flipkart"].genuineness_score is None
    # Every entry carries its product link (Req 7.3).
    assert all(row.product_url for row in prices)


def test_list_platform_prices_empty_for_product_without_prices(db_session):
    assert list(list_platform_prices(db_session, "fk_boat_rockerz")) == []


# --------------------------------------------------------------------------- #
# list_category_seasonality
# --------------------------------------------------------------------------- #
def test_list_category_seasonality_ordered_by_month_with_nullable_sale_event(
    db_session,
):
    rows = list_category_seasonality(db_session, "grocery/dairy")

    months = [row.month for row in rows]
    assert months == [1, 5, 10]
    assert months == sorted(months)

    by_month = {row.month: row for row in rows}
    # Only the best window maps to a named sale event; the rest are NULL.
    assert by_month[10].sale_event == "Big Billion Days"
    assert by_month[10].is_best_window is True
    assert by_month[1].sale_event is None
    assert by_month[5].sale_event is None


def test_list_category_seasonality_empty_for_unknown_category(db_session):
    assert list(list_category_seasonality(db_session, "no/such/category")) == []


# --------------------------------------------------------------------------- #
# Parameter binding (Req 18.2)
# --------------------------------------------------------------------------- #
def test_equality_query_compiles_to_placeholder_not_inline_literal():
    """Compiling the id lookup shows a bound placeholder, not the raw value."""
    # Mirrors the statement get_product_by_id builds internally.
    stmt = select(Product).where(Product.id == INJECTION)
    compiled = stmt.compile(compile_kwargs={"literal_binds": False})
    sql_text = str(compiled)

    # The user value is never inlined into the SQL text...
    assert INJECTION not in sql_text
    assert "DROP TABLE" not in sql_text
    # ...it is represented by a named bound-parameter placeholder instead.
    assert compiled.params
    for name in compiled.params:
        assert f":{name}" in sql_text
    # ...and the value itself travels in the parameter mapping.
    assert INJECTION in compiled.params.values()


def test_search_query_binds_like_pattern_not_inline_literal():
    """The ILIKE search binds its wildcard pattern rather than inlining it."""
    pattern = f"%{INJECTION}%"
    # Mirrors the statement search_products_by_name builds internally.
    stmt = select(Product).where(Product.normalized_name.ilike(pattern))
    compiled = stmt.compile(compile_kwargs={"literal_binds": False})
    sql_text = str(compiled)

    assert INJECTION not in sql_text
    assert "DROP TABLE" not in sql_text
    assert pattern in compiled.params.values()


def test_real_repository_execution_binds_user_value_out_of_band(
    engine, db_session
):
    """Capture the SQL the driver actually runs for a real repository call.

    The executed statement must reference the value through a placeholder while
    the value itself is delivered in the separate parameter set (Req 18.2).
    """
    captured: list[tuple[str, object]] = []

    def _record(conn, cursor, statement, parameters, context, executemany):
        captured.append((statement, parameters))

    event.listen(engine, "before_cursor_execute", _record)
    try:
        result = get_product_by_id(db_session, INJECTION)
    finally:
        event.remove(engine, "before_cursor_execute", _record)

    assert result is None
    assert captured, "expected the repository to execute a SQL statement"

    statement, parameters = captured[-1]
    # The compiled SQL text carries a placeholder, never the raw value.
    assert INJECTION not in statement
    assert "DROP TABLE" not in statement
    assert "?" in statement  # pysqlite qmark placeholder
    # The user value travels out-of-band in the driver's parameter set.
    flat = (
        list(parameters)
        if isinstance(parameters, (list, tuple))
        else list(parameters.values())
    )
    assert INJECTION in flat


def test_injection_like_input_is_treated_as_inert_data(db_session):
    """An injection-style string returns no rows and leaves the table intact."""
    # Treated as a literal id / substring: matches nothing, raises nothing.
    assert get_product_by_id(db_session, INJECTION) is None
    assert list(search_products_by_name(db_session, INJECTION)) == []

    # The products table was not dropped; normal reads still succeed.
    assert get_product_by_id(db_session, "amz_amul_butter") is not None
    assert len(search_products_by_name(db_session, "a")) == 3
