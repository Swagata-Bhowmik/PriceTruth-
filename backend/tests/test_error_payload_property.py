"""Property-based verification of the structured error payload (Task 1.3).

Implements the design's **Property 23: "All error responses use the structured
payload"** which validates **Requirement 15.3**:

    WHEN the Price_Truth_Platform returns an error response, THE
    Price_Truth_Platform SHALL return a structured error payload containing a
    human-readable message and a status code.

The single error contract (design, Error Handling section) is::

    {"error": {"code": <str>, "message": <str>, "status": <int>, "details": <object>}}

Rather than assert this for a handful of hand-picked errors, the tests below
generate a wide variety of error scenarios with Hypothesis and assert the
contract holds for every one:

* A temporary, test-only route raises domain :class:`AppError` instances whose
  ``code`` / ``message`` / ``status`` / ``details`` are all Hypothesis-varied,
  and the response is asserted to match the schema exactly and to echo the
  status. This exercises the *real* ``AppError`` handler registered in
  ``app.main`` end-to-end through FastAPI's ``TestClient``.
* A second test-only route is fed Hypothesis-varied invalid bodies to trigger
  the ``RequestValidationError`` (422) handler, asserting the same schema.
* A complementary check drives :meth:`ErrorPayload.build` directly to confirm
  the model itself always serializes to the contract.

The temporary ``/__prop__/*`` routes are registered on the real imported app
(mirroring ``tests/test_error_handlers.py``) so the assertions run against the
actually-registered exception handlers, not a stand-in.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, Field

from app.core.errors import AppError, ErrorPayload
from app.main import app

# ---------------------------------------------------------------------------
# Temporary, test-only routes registered on the real app.
#
# These let Hypothesis drive the genuinely-registered exception handlers in
# ``app.main`` over HTTP. Unique ``/__prop__/*`` paths avoid collision with the
# ``/__test__/*`` routes defined in tests/test_error_handlers.py.
# ---------------------------------------------------------------------------


class _AppErrorSpec(BaseModel):
    """Body describing the domain error the route under test should raise."""

    code: str
    message: str
    status: int
    details: dict[str, Any] = Field(default_factory=dict)


@app.post("/__prop__/raise-app-error")
async def _raise_app_error_route(spec: _AppErrorSpec):
    """Raise an AppError built from the request body (drives the real handler)."""
    raise AppError(
        code=spec.code,
        message=spec.message,
        status=spec.status,
        details=spec.details,
    )


class _ValidatedBody(BaseModel):
    """A route body with a strictly-typed field to provoke validation errors."""

    quantity: int


@app.post("/__prop__/validate")
async def _validate_route(body: _ValidatedBody):  # pragma: no cover - never valid
    return {"ok": True}


client = TestClient(app)


# ---------------------------------------------------------------------------
# Shared contract assertion
# ---------------------------------------------------------------------------

# The set of HTTP error statuses the platform's handlers legitimately produce.
_VALID_ERROR_STATUSES = [400, 401, 403, 404, 409, 422, 500, 503]


def _assert_error_contract(body: dict) -> dict:
    """Assert ``body`` conforms to ``{"error": {code, message, status, details}}``.

    Enforces the full Requirement 15.3 contract: the payload carries a
    human-readable (non-empty string) message and an integer status code, plus
    the stable ``code`` identifier and a ``details`` object.
    """
    assert set(body.keys()) == {"error"}, body
    err = body["error"]
    assert set(err.keys()) == {"code", "message", "status", "details"}, err

    # Human-readable message: a non-empty string (Req 15.3).
    assert isinstance(err["message"], str)
    assert err["message"].strip() != ""

    # Status code: an integer in the valid HTTP error range (Req 15.3).
    assert isinstance(err["status"], int)
    assert not isinstance(err["status"], bool)
    assert 100 <= err["status"] <= 599

    # Stable machine-readable code and a details object are always present.
    assert isinstance(err["code"], str) and err["code"] != ""
    assert isinstance(err["details"], dict)
    return err


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Non-empty, whitespace-free identifier for the error ``code``.
_code_strategy = st.text(
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_",
    min_size=1,
    max_size=48,
)

# Non-empty, printable-ASCII message with at least one non-whitespace character.
_message_strategy = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=120,
).filter(lambda s: s.strip() != "")

_status_strategy = st.sampled_from(_VALID_ERROR_STATUSES)

# Arbitrary JSON-safe ``details`` object: string keys, scalar JSON values.
_json_scalar = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000_000, max_value=1_000_000_000),
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), max_size=48),
)
_details_strategy = st.dictionaries(
    keys=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=16),
    values=_json_scalar,
    max_size=6,
)


# Feature: price-truth-platform, Property 23: All error responses use the structured payload
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    code=_code_strategy,
    message=_message_strategy,
    status=_status_strategy,
    details=_details_strategy,
)
def test_app_error_responses_use_structured_payload(code, message, status, details):
    """Any raised domain AppError renders the structured payload and echoes status."""
    resp = client.post(
        "/__prop__/raise-app-error",
        json={
            "code": code,
            "message": message,
            "status": status,
            "details": details,
        },
    )

    # The response HTTP status echoes the domain error's status (Req 15.3).
    assert resp.status_code == status

    err = _assert_error_contract(resp.json())
    # The payload faithfully carries the raised error's fields.
    assert err["code"] == code
    assert err["message"] == message
    assert err["status"] == status
    assert err["details"] == details


# Values that can never satisfy a required ``int`` field, so each provokes the
# RequestValidationError (422) handler.
_invalid_quantity = st.one_of(
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12),
    st.lists(st.integers(), max_size=4),
    st.dictionaries(
        st.text(alphabet="abc", min_size=1, max_size=3), st.integers(), max_size=3
    ),
    st.none(),
)
_invalid_bodies = st.one_of(
    st.builds(lambda v: {"quantity": v}, _invalid_quantity),
    st.just({}),  # required field missing
    st.just({"unrelated_field": 1}),  # required field missing, extra field present
)


# Feature: price-truth-platform, Property 23: All error responses use the structured payload
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(body=_invalid_bodies)
def test_validation_errors_use_structured_payload(body):
    """Any invalid request body yields the same 422 structured payload."""
    resp = client.post("/__prop__/validate", json=body)

    assert resp.status_code == 422
    err = _assert_error_contract(resp.json())
    assert err["code"] == "VALIDATION_ERROR"
    assert err["status"] == 422


# Feature: price-truth-platform, Property 23: All error responses use the structured payload
@settings(max_examples=200, deadline=None)
@given(
    code=_code_strategy,
    message=_message_strategy,
    status=_status_strategy,
    details=_details_strategy,
)
def test_error_payload_model_conforms_to_contract(code, message, status, details):
    """Complementary check: ErrorPayload.build always serializes to the contract."""
    payload = ErrorPayload.build(
        code=code, message=message, status=status, details=details
    )
    dumped = payload.model_dump()

    err = _assert_error_contract(dumped)
    assert err == {
        "code": code,
        "message": message,
        "status": status,
        "details": details,
    }
