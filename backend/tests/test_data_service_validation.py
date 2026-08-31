"""Verification for OFF external-value validation and shaping (Task 7.3).

These tests exercise :func:`app.services.data_service.validate_off_product` and
the cache-first read-path wrapper :func:`get_validated_off_product` without a
live Redis and without the network:

* the Redis ``cache_get`` / ``cache_set`` used by the service are replaced with
  an in-memory dict-backed fake (via ``monkeypatch``), and
* OFF access is driven through an :class:`httpx.MockTransport`.

They cover the three required behaviours of :func:`validate_off_product`:

* (a) a complete payload  -> every field present, crowd-sourced flag set,
* (b) a payload missing ``brands`` / ``quantity`` -> those fields marked
  unavailable while the others are returned, no error raised,
* (c) an invalid ``quantity`` (``"abc"`` or ``-5``) -> the value is rejected,
  the rejection is logged, and the field is marked unavailable,

plus supporting edge cases (non-string name, category encodings, alternate-key
recovery, non-mapping payloads) and that the shaped view flows through the
cache-first accessor while ``"unavailable"`` results are propagated unchanged.
"""

import asyncio
import logging

import httpx
import pytest

from app.services import data_service
from app.services.data_service import (
    OFF_PRODUCT_FIELDS,
    SOURCE_OPEN_FOOD_FACTS,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    get_validated_off_product,
    validate_off_product,
)


# --- Test doubles -----------------------------------------------------------


class FakeCache:
    """Minimal in-memory stand-in for the Redis get/set helpers."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.store[key] = value


@pytest.fixture
def cache(monkeypatch):
    """Replace the module-level cache get/set with an in-memory fake."""
    fake = FakeCache()
    monkeypatch.setattr(data_service, "cache_get", fake.get)
    monkeypatch.setattr(data_service, "cache_set", fake.set)
    return fake


def _run(coro):
    """Run an async coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


def _ok_transport(product: dict) -> httpx.MockTransport:
    """A transport that returns a successful OFF product body."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": 1, "product": product})

    return httpx.MockTransport(handler)


def _down_transport(status: int = 503) -> httpx.MockTransport:
    """A transport that always fails with a 5xx."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="service unavailable")

    return httpx.MockTransport(handler)


# --- (a) complete payload ---------------------------------------------------


def test_complete_payload_returns_all_fields_and_flags_crowd_sourced():
    """(a) Every used field is present, valid, and the crowd-sourced flag set."""
    raw = {
        "product_name": "Nutella",
        "brands": "Ferrero",
        "quantity": 750,
        "categories": "Spreads, Hazelnut spreads",
        "some_unused_off_field": "ignored",
    }

    shaped = validate_off_product(raw)

    assert shaped["product_name"] == "Nutella"
    assert shaped["brands"] == "Ferrero"
    assert shaped["quantity"] == 750.0  # normalised to float
    assert shaped["categories"] == "Spreads, Hazelnut spreads"
    # Nothing missing/rejected.
    assert shaped["unavailable_fields"] == []
    # Crowd-sourced origin disclosed (Req 10.3).
    assert shaped["source"] == SOURCE_OPEN_FOOD_FACTS
    assert shaped["crowd_sourced"] is True
    # Only the known fields (plus flags) are exposed; raw extras are dropped.
    assert "some_unused_off_field" not in shaped


def test_string_quantity_is_accepted_and_normalised():
    """A numeric string quantity (OFF's ``product_quantity``) is accepted."""
    shaped = validate_off_product(
        {"product_name": "Milk", "brands": "Amul", "quantity": "500", "categories": "Dairy"}
    )
    assert shaped["quantity"] == 500.0
    assert "quantity" not in shaped["unavailable_fields"]


# --- (b) missing fields degrade gracefully ----------------------------------


def test_missing_brands_and_quantity_are_marked_unavailable_without_error():
    """(b) Absent fields become unavailable; present fields are returned intact."""
    raw = {"product_name": "Generic Oats", "categories": "Cereals"}

    shaped = validate_off_product(raw)  # must not raise

    # Present fields returned unchanged.
    assert shaped["product_name"] == "Generic Oats"
    assert shaped["categories"] == "Cereals"
    # Missing fields marked unavailable (value None + listed).
    assert shaped["brands"] is None
    assert shaped["quantity"] is None
    assert set(shaped["unavailable_fields"]) == {"brands", "quantity"}
    # Still flagged crowd-sourced.
    assert shaped["crowd_sourced"] is True
    assert shaped["source"] == SOURCE_OPEN_FOOD_FACTS


def test_empty_payload_marks_every_field_unavailable():
    """An empty payload yields all-unavailable fields but a stable shape."""
    shaped = validate_off_product({})

    for field in OFF_PRODUCT_FIELDS:
        assert shaped[field] is None
    assert set(shaped["unavailable_fields"]) == set(OFF_PRODUCT_FIELDS)
    assert shaped["crowd_sourced"] is True


def test_non_mapping_payload_degrades_gracefully():
    """A non-dict payload is treated as an empty product, never an error."""
    shaped = validate_off_product(None)

    assert set(shaped["unavailable_fields"]) == set(OFF_PRODUCT_FIELDS)
    assert shaped["source"] == SOURCE_OPEN_FOOD_FACTS
    assert shaped["crowd_sourced"] is True


# --- (c) invalid values are rejected and logged -----------------------------


def test_garbage_quantity_string_is_rejected_and_logged(caplog):
    """(c) A non-numeric quantity ("abc") is rejected, logged, marked unavailable."""
    raw = {"product_name": "Chips", "brands": "Lay's", "quantity": "abc", "categories": "Snacks"}

    with caplog.at_level(logging.WARNING, logger="app.services.data_service"):
        shaped = validate_off_product(raw)

    assert shaped["quantity"] is None
    assert "quantity" in shaped["unavailable_fields"]
    # Valid siblings are unaffected.
    assert shaped["product_name"] == "Chips"
    assert shaped["brands"] == "Lay's"
    # The rejection was recorded in the application log (Req 15.4).
    assert any(
        record.getMessage() == "off_field_rejected"
        and getattr(record, "field", None) == "quantity"
        for record in caplog.records
    )


def test_negative_quantity_is_rejected_and_logged(caplog):
    """(c) A non-positive quantity (-5) is rejected, logged, marked unavailable."""
    raw = {"product_name": "Rice", "brands": "India Gate", "quantity": -5, "categories": "Grains"}

    with caplog.at_level(logging.WARNING, logger="app.services.data_service"):
        shaped = validate_off_product(raw)

    assert shaped["quantity"] is None
    assert "quantity" in shaped["unavailable_fields"]
    assert any(
        record.getMessage() == "off_field_rejected"
        and getattr(record, "field", None) == "quantity"
        for record in caplog.records
    )


def test_non_string_product_name_is_rejected_and_logged(caplog):
    """A non-string product name fails type validation and is dropped."""
    raw = {"product_name": 12345, "brands": "BrandX", "quantity": 100, "categories": "Misc"}

    with caplog.at_level(logging.WARNING, logger="app.services.data_service"):
        shaped = validate_off_product(raw)

    assert shaped["product_name"] is None
    assert "product_name" in shaped["unavailable_fields"]
    assert any(
        record.getMessage() == "off_field_rejected"
        and getattr(record, "field", None) == "product_name"
        for record in caplog.records
    )


@pytest.mark.parametrize("bad_quantity", ["abc", -5, 0, float("nan"), float("inf"), True, "750 g", []])
def test_various_invalid_quantities_are_all_rejected(bad_quantity):
    """Boolean, zero, NaN/inf, unit-bearing strings and lists are all rejected."""
    shaped = validate_off_product(
        {"product_name": "X", "brands": "Y", "quantity": bad_quantity, "categories": "Z"}
    )
    assert shaped["quantity"] is None
    assert "quantity" in shaped["unavailable_fields"]


# --- Field-encoding edge cases ----------------------------------------------


def test_categories_accepts_list_encoding():
    """A list of category tags is joined into a single comma string."""
    shaped = validate_off_product(
        {"product_name": "P", "brands": "B", "quantity": 10, "categories": ["Snacks", "Chips"]}
    )
    assert shaped["categories"] == "Snacks, Chips"
    assert "categories" not in shaped["unavailable_fields"]


def test_categories_falls_back_to_singular_category_key():
    """When ``categories`` is absent, the singular ``category`` key is used."""
    shaped = validate_off_product(
        {"product_name": "P", "brands": "B", "quantity": 10, "category": "Beverages"}
    )
    assert shaped["categories"] == "Beverages"
    assert "categories" not in shaped["unavailable_fields"]


def test_quantity_recovers_from_alternate_key_when_primary_is_invalid():
    """An invalid ``quantity`` recovers from a valid ``product_quantity``."""
    shaped = validate_off_product(
        {
            "product_name": "P",
            "brands": "B",
            "quantity": "750 g",  # unit-bearing string -> invalid
            "product_quantity": 750,  # numeric fallback -> valid
            "categories": "C",
        }
    )
    assert shaped["quantity"] == 750.0
    assert "quantity" not in shaped["unavailable_fields"]


def test_blank_text_is_rejected_as_unavailable():
    """A whitespace-only name/brand is treated as unavailable, not accepted."""
    shaped = validate_off_product(
        {"product_name": "   ", "brands": "", "quantity": 5, "categories": "C"}
    )
    assert shaped["product_name"] is None
    assert shaped["brands"] is None
    assert {"product_name", "brands"}.issubset(set(shaped["unavailable_fields"]))


# --- Read-path accessor integration -----------------------------------------


def test_get_validated_off_product_returns_shaped_product(cache):
    """A successful cache-first lookup returns validated, crowd-sourced data."""
    raw = {
        "product_name": "Nutella",
        "brands": "Ferrero",
        "quantity": 750,
        "categories": "Spreads",
    }

    result = _run(
        get_validated_off_product(
            "3017620422003", transport=_ok_transport(raw), backoff_base_seconds=0.0
        )
    )

    assert result["status"] == STATUS_OK
    product = result["product"]
    assert product["product_name"] == "Nutella"
    assert product["quantity"] == 750.0
    assert product["source"] == SOURCE_OPEN_FOOD_FACTS
    assert product["crowd_sourced"] is True
    assert product["unavailable_fields"] == []


def test_get_validated_off_product_shapes_incomplete_payload(cache):
    """Missing OFF fields are marked unavailable on the accessor's happy path."""
    raw = {"product_name": "Butter"}  # brands/quantity/categories absent

    result = _run(
        get_validated_off_product("111", transport=_ok_transport(raw), backoff_base_seconds=0.0)
    )

    assert result["status"] == STATUS_OK
    product = result["product"]
    assert product["product_name"] == "Butter"
    assert set(product["unavailable_fields"]) == {"brands", "quantity", "categories"}
    assert product["crowd_sourced"] is True


def test_get_validated_off_product_propagates_unavailable(cache):
    """When OFF is unavailable with a cold cache, the result passes through."""
    result = _run(
        get_validated_off_product("222", transport=_down_transport(), backoff_base_seconds=0.0)
    )

    assert result["status"] == STATUS_UNAVAILABLE
    assert result["product"] is None
