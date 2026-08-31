"""API tests for the Unit Price Comparator endpoint (Task 6.3).

Exercises ``POST /api/v1/unit-price/compare`` end-to-end through a FastAPI
``TestClient``. The endpoint is a thin boundary over
``app.services.unit_price_service.compare_units`` (already property-tested in
``tests/test_unit_price_property.py``), so these tests focus on the two things
the endpoint itself owns:

* returning the service's comparison result as JSON (Req 5.3, 14.4), and
* boundary input validation (Req 18.1) - an unsupported unit or non-positive
  price is rejected as a structured 422 (Req 15.3), while a non-positive
  quantity is *not* a request error and is instead excluded by the service
  with a reason (Req 5.5).

``raise_server_exceptions=True`` (the TestClient default) means any unhandled
exception escaping the endpoint would fail the request, so a normal status code
also proves the endpoint returned cleanly.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

_COMPARE_URL = "/api/v1/unit-price/compare"

# The exact request body from the design's `POST /api/v1/unit-price/compare`
# example: two comparable variants plus one with a zero quantity to be excluded.
_DESIGN_EXAMPLE_BODY = {
    "variants": [
        {"label": "Small", "price": 45.0, "quantity": 100, "unit": "g"},
        {"label": "Family", "price": 199.0, "quantity": 1, "unit": "kg"},
        {"label": "Broken", "price": 60.0, "quantity": 0, "unit": "g"},
    ]
}


def test_compare_design_example_returns_expected_comparison():
    """The design example yields Family as best value and excludes Broken.

    Validates the success contract (Req 5.3, 14.4): each included variant
    reports its price and computed unit price, the lowest unit price is flagged
    best value (Req 5.2), and the zero-quantity variant is excluded with a
    machine-readable reason (Req 5.5).

    Small:  45 / 100 g            = 0.45  per g
    Family: 199 / (1 kg -> 1000 g) = 0.199 per g  -> best value
    Broken: quantity 0             -> excluded (non_positive_quantity)
    """
    resp = client.post(_COMPARE_URL, json=_DESIGN_EXAMPLE_BODY)

    assert resp.status_code == 200
    body = resp.json()

    # Mass family -> standardised to grams.
    assert body["standard_unit"] == "g"

    # Only the two valid-quantity variants appear in the comparison, in order.
    comparison = body["comparison"]
    assert [entry["label"] for entry in comparison] == ["Small", "Family"]

    by_label = {entry["label"]: entry for entry in comparison}

    small = by_label["Small"]
    assert small["unit_price"] == pytest.approx(0.45)
    # Only the best value carries the flag; other entries omit it entirely.
    assert "best_value" not in small

    family = by_label["Family"]
    assert family["unit_price"] == pytest.approx(0.199)
    assert family["best_value"] is True

    # The zero-quantity variant is excluded (not compared) with a reason.
    assert body["excluded"] == [
        {"label": "Broken", "reason": "non_positive_quantity"}
    ]


def test_invalid_unit_returns_422_structured_error():
    """An unsupported unit is rejected at the boundary as a structured 422.

    ``unit="oz"`` fails the ``^(g|kg|ml|l)$`` pattern (Req 18.1), so the request
    never reaches the service and the central handler renders the shared error
    payload (Req 15.3): ``{"error": {"code", "message", "status", "details"}}``.
    """
    body = {
        "variants": [
            {"label": "Bottle", "price": 99.0, "quantity": 500, "unit": "oz"},
        ]
    }

    resp = client.post(_COMPARE_URL, json=body)

    assert resp.status_code == 422
    payload = resp.json()

    # Structured error contract (Req 15.3).
    assert set(payload["error"]) >= {"code", "message", "status", "details"}
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["status"] == 422
    assert isinstance(payload["error"]["message"], str) and payload["error"]["message"]
    # The validation detail should point at the offending `unit` field.
    errors = payload["error"]["details"]["errors"]
    assert any("unit" in err.get("loc", []) for err in errors)


def test_non_positive_price_returns_422_structured_error():
    """A non-positive price violates the ``gt=0`` boundary constraint (Req 18.1)."""
    body = {
        "variants": [
            {"label": "Freebie", "price": 0, "quantity": 100, "unit": "g"},
        ]
    }

    resp = client.post(_COMPARE_URL, json=body)

    assert resp.status_code == 422
    payload = resp.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["status"] == 422
    errors = payload["error"]["details"]["errors"]
    assert any("price" in err.get("loc", []) for err in errors)


def test_non_positive_quantity_is_excluded_not_rejected():
    """A non-positive quantity flows through and is excluded, not rejected (Req 5.5).

    ``quantity`` carries no boundary constraint, so a zero (or negative) value
    is *not* a 422: it reaches the service and is reported in ``excluded`` with
    a reason, while a comparable variant is still compared. This is the
    graceful-exclusion counterpart to the strict ``unit``/``price`` validation.
    """
    body = {
        "variants": [
            {"label": "Known", "price": 50.0, "quantity": 250, "unit": "ml"},
            {"label": "ZeroQty", "price": 80.0, "quantity": 0, "unit": "ml"},
        ]
    }

    resp = client.post(_COMPARE_URL, json=body)

    assert resp.status_code == 200
    body_json = resp.json()

    assert body_json["standard_unit"] == "ml"
    assert [entry["label"] for entry in body_json["comparison"]] == ["Known"]
    assert body_json["excluded"] == [
        {"label": "ZeroQty", "reason": "non_positive_quantity"}
    ]


def test_null_quantity_is_rejected_at_boundary():
    """A null quantity violates the required ``float`` type -> structured 422.

    With ``quantity: float`` (required, non-nullable) a completely missing value
    is a malformed request caught by boundary validation (Req 18.1), distinct
    from a present-but-non-positive quantity, which the service excludes above.
    """
    body = {
        "variants": [
            {"label": "NoQty", "price": 80.0, "quantity": None, "unit": "ml"},
        ]
    }

    resp = client.post(_COMPARE_URL, json=body)

    assert resp.status_code == 422
    payload = resp.json()
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["status"] == 422
    errors = payload["error"]["details"]["errors"]
    assert any("quantity" in err.get("loc", []) for err in errors)
