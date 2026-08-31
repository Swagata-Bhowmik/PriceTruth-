"""Unit tests for the Cross-Platform Aggregator service (Task 10.1).

These example-based tests seed an in-memory SQLite database with ``products``
and ``platform_prices`` rows and exercise
``app.services.cross_platform_service.aggregate_cross_platform`` directly,
covering the Requirement 7 branches:

* multi-platform comparison marks the true minimum price as the best deal
  (Req 7.1, 7.2, 7.3);
* a listing's genuineness score is shown when present and omitted when absent
  (Req 7.4);
* a single platform returns the price plus a no-comparison message (Req 7.5);
* a product with no platform rows returns the unavailable message (Req 7.6).

The property-based tests for Correctness Properties 18 and 19 are owned by the
separate task 10.2; these tests stay focused on concrete examples and edge
cases.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, PlatformPrice, Product
from app.services.cross_platform_service import (
    MESSAGE_NO_DATA,
    MESSAGE_SINGLE_PLATFORM,
    aggregate_cross_platform,
)


@pytest.fixture()
def db():
    """Provide an isolated in-memory SQLite session with the schema created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session: Session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _add_product(db: Session, product_id: str) -> None:
    db.add(
        Product(
            id=product_id,
            name=f"Product {product_id}",
            normalized_name=f"product {product_id}",
            brand="BrandX",
            category="electronics/headphones",
        )
    )


def _add_platform_price(
    db: Session,
    product_id: str,
    platform: str,
    price: float,
    *,
    genuineness_score: int | None = None,
) -> None:
    db.add(
        PlatformPrice(
            product_id=product_id,
            platform=platform,
            price=price,
            product_url=f"https://example.com/{platform.lower().replace(' ', '')}/{product_id}",
            genuineness_score=genuineness_score,
        )
    )


def test_multi_platform_marks_true_minimum_as_best_deal(db):
    """Req 7.1/7.2/7.3: the cheapest of several platforms is the best deal."""
    _add_product(db, "p1")
    # Insertion order deliberately does NOT put the minimum first: Flipkart at
    # 1399 is the true minimum but is added second.
    _add_platform_price(db, "p1", "Amazon", 1499.0)
    _add_platform_price(db, "p1", "Flipkart", 1399.0)
    _add_platform_price(db, "p1", "Croma", 1599.0)
    db.commit()

    result = aggregate_cross_platform(db, "p1")

    assert result["available"] is True
    assert result["comparison_available"] is True
    assert result["best_deal_platform"] == "Flipkart"

    entries = result["platforms"]
    assert {e["platform"] for e in entries} == {"Amazon", "Flipkart", "Croma"}
    # Every entry carries a non-empty product link (Req 7.3).
    assert all(e["product_url"] for e in entries)

    # Exactly one entry is the best deal and it holds the minimum price (Req 7.2).
    best_entries = [e for e in entries if e.get("best_deal")]
    assert len(best_entries) == 1
    best = best_entries[0]
    prices = [e["price"] for e in entries]
    assert best["price"] == min(prices) == 1399.0
    assert best["platform"] == "Flipkart"
    assert all(e["price"] >= best["price"] for e in entries)


def test_genuineness_score_shown_when_present_and_omitted_when_absent(db):
    """Req 7.4: score appears only for listings that actually have one."""
    _add_product(db, "p2")
    _add_platform_price(db, "p2", "Amazon", 999.0, genuineness_score=88)
    _add_platform_price(db, "p2", "Flipkart", 1049.0)  # no score
    db.commit()

    result = aggregate_cross_platform(db, "p2")
    by_platform = {e["platform"]: e for e in result["platforms"]}

    # The listing with a score exposes it with the exact stored value...
    assert by_platform["Amazon"]["genuineness_score"] == 88
    # ...and the listing without a score omits the key entirely.
    assert "genuineness_score" not in by_platform["Flipkart"]


def test_single_platform_returns_no_comparison_message(db):
    """Req 7.5: one platform -> single price, no comparison, nothing marked."""
    _add_product(db, "p3")
    _add_platform_price(db, "p3", "Amazon", 799.0, genuineness_score=91)
    db.commit()

    result = aggregate_cross_platform(db, "p3")

    assert result["available"] is True
    assert result["comparison_available"] is False
    assert result["best_deal_platform"] is None
    assert result["message"] == MESSAGE_SINGLE_PLATFORM

    assert len(result["platforms"]) == 1
    only_entry = result["platforms"][0]
    assert only_entry["platform"] == "Amazon"
    assert only_entry["price"] == 799.0
    assert only_entry["genuineness_score"] == 91
    # No best deal is marked when there is nothing to compare against.
    assert "best_deal" not in only_entry


def test_no_platform_data_returns_unavailable(db):
    """Req 7.6: a product with no platform rows is reported unavailable."""
    _add_product(db, "p4")  # product exists but has no platform_prices rows
    db.commit()

    result = aggregate_cross_platform(db, "p4")

    assert result["available"] is False
    assert result["comparison_available"] is False
    assert result["best_deal_platform"] is None
    assert result["platforms"] == []
    assert result["message"] == MESSAGE_NO_DATA


def test_unknown_product_is_also_unavailable(db):
    """A product id with no rows at all is treated as unavailable (Req 7.6)."""
    result = aggregate_cross_platform(db, "does-not-exist")

    assert result["available"] is False
    assert result["platforms"] == []
    assert result["message"] == MESSAGE_NO_DATA
