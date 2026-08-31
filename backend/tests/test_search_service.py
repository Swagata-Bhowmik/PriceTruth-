"""Unit tests for the Product Search service (task 12.1, Requirement 1).

These tests seed an in-memory SQLite database with a handful of products and
verify the search service's documented outcomes plus the ``SelectedProduct``
contract:

* a matching query returns each hit's name, brand, and category (Req 1.1, 1.2);
* an empty / blank / punctuation-only query returns the prompt message
  (Req 1.4);
* a query with no matches returns the no-results message and the manual-entry
  affordance (Req 1.5);
* manual entry yields a ``SelectedProduct`` (Req 1.6);
* selecting a searched product yields a ``SelectedProduct`` (Req 1.3).

The in-memory engine uses a ``StaticPool`` so the whole test shares the single
connection that backs the ``sqlite://`` database.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Product
from app.services.search_service import (
    NO_RESULTS_MESSAGE,
    PROMPT_ENTER_QUERY_MESSAGE,
    SOURCE_MANUAL,
    SOURCE_SEARCH,
    STATUS_EMPTY_QUERY,
    STATUS_NO_RESULTS,
    STATUS_OK,
    UNKNOWN_BRAND,
    ManualEntryError,
    SelectedProduct,
    create_manual_entry,
    search_products,
    select_product,
)


@pytest.fixture()
def db_session():
    """Yield a session backed by a seeded in-memory SQLite database."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(
        bind=engine, class_=Session, expire_on_commit=False
    )
    session = testing_session()
    session.add_all(
        [
            Product(
                id="amz_nutella",
                name="Nutella Hazelnut Spread 750g",
                normalized_name="nutella hazelnut spread 750g",
                brand="Ferrero",
                category="grocery/spreads",
            ),
            Product(
                id="amz_amul_butter",
                name="Amul Butter 500g",
                normalized_name="amul butter 500g",
                brand="Amul",
                category="grocery/dairy",
            ),
            Product(
                id="fk_boat_rockerz",
                name="boAt Rockerz 450 Bluetooth Headphones",
                normalized_name="boat rockerz 450 bluetooth headphones",
                brand=None,  # exercises the non-empty-brand fallback (Property 1)
                category="electronics/headphones",
            ),
        ]
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def test_matching_query_returns_identifying_fields(db_session):
    """Req 1.1, 1.2 - a match carries a non-empty name, brand, and category."""
    result = search_products(db_session, "butter")

    assert result["status"] == STATUS_OK
    assert result["manual_entry"] is False
    assert len(result["results"]) == 1

    match = result["results"][0]
    assert match["id"] == "amz_amul_butter"
    assert match["name"] == "Amul Butter 500g"
    assert match["brand"] == "Amul"
    assert match["category"] == "grocery/dairy"
    # Every identifying field is present and non-empty (Correctness Property 1).
    assert all(match[key] for key in ("name", "brand", "category"))


def test_matching_query_is_case_and_punctuation_insensitive(db_session):
    """Req 1.1 - normalization lines the query up with normalized_name."""
    result = search_products(db_session, "  NUTELLA!!  ")

    assert result["status"] == STATUS_OK
    assert [m["id"] for m in result["results"]] == ["amz_nutella"]


def test_missing_brand_is_coerced_to_non_empty(db_session):
    """Correctness Property 1 - a null brand is shown as a non-empty fallback."""
    result = search_products(db_session, "rockerz")

    assert result["status"] == STATUS_OK
    match = result["results"][0]
    assert match["brand"] == UNKNOWN_BRAND
    assert match["brand"]  # non-empty


@pytest.mark.parametrize("query", ["", "   ", "\t\n", "!!!", "-", None])
def test_empty_or_blank_query_returns_prompt(db_session, query):
    """Req 1.4 - an empty/blank/punctuation-only query prompts for input."""
    result = search_products(db_session, query)

    assert result["status"] == STATUS_EMPTY_QUERY
    assert result["message"] == PROMPT_ENTER_QUERY_MESSAGE
    assert result["results"] == []
    assert result["manual_entry"] is False


def test_no_match_returns_no_results_and_manual_entry(db_session):
    """Req 1.5 - a query with no matches offers manual entry."""
    result = search_products(db_session, "nonexistent gadget xyz")

    assert result["status"] == STATUS_NO_RESULTS
    assert result["message"] == NO_RESULTS_MESSAGE
    assert result["results"] == []
    assert result["manual_entry"] is True


def test_manual_entry_yields_selected_product():
    """Req 1.6 - manual entry produces a SelectedProduct with the given values."""
    selected = create_manual_entry(
        name="  Local Peanut Butter 1kg ",
        displayed_price=349.0,
        reference_price=499.0,
        pack_quantity=1000.0,
        pack_unit="g",
    )

    assert isinstance(selected, SelectedProduct)
    assert selected.name == "Local Peanut Butter 1kg"  # trimmed
    assert selected.displayed_price == 349.0
    assert selected.reference_price == 499.0
    assert selected.pack_quantity == 1000.0
    assert selected.pack_unit == "g"
    assert selected.source == SOURCE_MANUAL
    assert selected.id.startswith("manual:")


def test_manual_entry_accepts_minimum_inputs():
    """Req 1.6 - only name + displayed price are strictly required."""
    selected = create_manual_entry(name="Mystery Item", displayed_price=10.0)

    assert selected.displayed_price == 10.0
    assert selected.reference_price is None
    assert selected.pack_quantity is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "   ", "displayed_price": 10.0},  # blank name
        {"name": "Valid", "displayed_price": 0},  # non-positive price
        {"name": "Valid", "displayed_price": -5.0},  # negative price
        {"name": "Valid", "displayed_price": float("nan")},  # non-finite price
        {"name": "Valid", "displayed_price": True},  # bool is not a number
        {"name": "Valid", "displayed_price": 10.0, "reference_price": -1.0},
        {"name": "Valid", "displayed_price": 10.0, "pack_quantity": 0},
    ],
)
def test_manual_entry_rejects_invalid_inputs(kwargs):
    """Req 1.6 - minimal validation rejects bad name/price/quantity."""
    with pytest.raises(ManualEntryError):
        create_manual_entry(**kwargs)


def test_select_product_returns_selected_product(db_session):
    """Req 1.3 - a searched product resolves to a SelectedProduct."""
    selected = select_product(db_session, "amz_amul_butter")

    assert isinstance(selected, SelectedProduct)
    assert selected.id == "amz_amul_butter"
    assert selected.name == "Amul Butter 500g"
    assert selected.category == "grocery/dairy"
    assert selected.source == SOURCE_SEARCH
    # A searched selection carries no prices yet.
    assert selected.displayed_price is None


def test_select_unknown_product_returns_none(db_session):
    """Req 1.3 - an unknown id resolves to None for a not-found response."""
    assert select_product(db_session, "does_not_exist") is None
