"""Property-based test for the Product Search service (Task 12.2, Requirement 1.2).

Implements one of the design's Correctness Properties for the search entry point:

* Property 1 - Search results always carry identifying fields (Req 1.2)

The property is implemented by exactly one property-based test that carries its
traceability comment and runs at least 100 generated examples.

Strategy
--------
Hypothesis generates a product corpus whose brands are sometimes missing / blank
and whose categories are sometimes blank, so the search service's non-empty
display fallbacks (``UNKNOWN_NAME`` / ``UNKNOWN_BRAND`` / ``UNKNOWN_CATEGORY``)
are exercised. Each generated product's name embeds a shared token, and its
``normalized_name`` is derived with the very function the service uses to
normalize a query, so a search for that token is guaranteed to return the whole
corpus (the "returns matches" precondition of the property). Every returned
match must then carry a non-empty name, brand, and category.

Each example builds a fresh in-memory SQLite database (``StaticPool``) inside the
test body so that no state leaks between Hypothesis examples.
"""

import string

from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Product
from app.services.search_service import (
    STATUS_OK,
    normalize_query,
    search_products,
)

# A token embedded in every generated product name and used as the query so the
# search always returns the whole corpus (exercising the "returns matches"
# branch on every example). Lower-case ASCII survives query normalization
# unchanged, so the stored normalized_name and the normalized query line up.
SHARED_TOKEN = "pricetruthtoken"

# Name fragments: one to three short lower-case words.
_words = st.lists(
    st.text(alphabet=string.ascii_lowercase, min_size=1, max_size=6),
    min_size=1,
    max_size=3,
)

# Brand is nullable in the schema and frequently absent in crowd-sourced data,
# so it is generated as None / empty / whitespace / real text to drive the
# non-empty-brand fallback (Property 1).
_brands = st.one_of(
    st.none(),
    st.just(""),
    st.just("   "),
    st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=10),
)

# Category is non-nullable, but a source row could still carry an empty or
# whitespace value, which must also be coerced to a non-empty display value.
_categories = st.one_of(
    st.just(""),
    st.just("   "),
    st.text(alphabet=string.ascii_lowercase + "/", min_size=1, max_size=12),
)

# A product corpus: each entry supplies the raw pieces; the test assigns a
# unique id per index so the primary-key constraint is always satisfied.
_product_specs = st.lists(
    st.fixed_dictionaries(
        {"words": _words, "brand": _brands, "category": _categories}
    ),
    min_size=1,
    max_size=6,
)


def _seed_session(specs):
    """Build a fresh in-memory SQLite session seeded from ``specs``.

    Every product's name embeds :data:`SHARED_TOKEN`, and its
    ``normalized_name`` is derived with :func:`normalize_query` exactly as the
    search service normalizes an incoming query, so a search for the token
    matches every seeded row. Returns ``(engine, session)``; the caller closes
    the session and disposes the engine.
    """

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)()

    products = []
    for index, spec in enumerate(specs):
        name = " ".join([*spec["words"], SHARED_TOKEN])
        products.append(
            Product(
                id=f"p{index}",
                name=name,
                normalized_name=normalize_query(name),
                brand=spec["brand"],
                category=spec["category"],
            )
        )
    session.add_all(products)
    session.commit()
    return engine, session


# Feature: price-truth-platform, Property 1: Search results always carry identifying fields
@settings(max_examples=150, deadline=None)
@given(_product_specs)
def test_search_results_always_carry_identifying_fields(specs):
    """Validates: Requirements 1.2

    For any product corpus and any query that returns matches, every returned
    match includes a non-empty product name, brand, and category. The service
    applies its UNKNOWN_* display fallbacks when a source row omits a value, so
    the guarantee holds even when the seeded brand/category is null or blank.
    """

    engine, session = _seed_session(specs)
    try:
        # A limit above the corpus size ensures every matching row is returned,
        # so the assertion below checks the shaping of every generated product.
        result = search_products(session, SHARED_TOKEN, limit=len(specs) + 5)
    finally:
        session.close()
        engine.dispose()

    # The shared token is present in every product, so the query returns matches.
    assert result["status"] == STATUS_OK
    assert result["results"], "the shared-token query should return matches"
    assert len(result["results"]) == len(specs)

    for match in result["results"]:
        assert isinstance(match["name"], str) and match["name"].strip()
        assert isinstance(match["brand"], str) and match["brand"].strip()
        assert isinstance(match["category"], str) and match["category"].strip()
