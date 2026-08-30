"""Lightweight verification for the central exception handlers (Task 1.2).

Confirms every error path renders the single structured payload defined in
``app.core.errors`` (Req 15.3) and that a database-connectivity failure maps to
a 503 with a retry message rather than an unhandled failure (Req 16.4).

The temporary ``/__test__/*`` routes are registered on the real app so the
assertions exercise the real, registered exception handlers end-to-end. The
full property-based test for the payload is a separate task (1.3).
"""

from fastapi.testclient import TestClient
from pydantic import BaseModel
from sqlalchemy.exc import OperationalError

from app.core.errors import AppError, ErrorPayload
from app.main import app


class _Body(BaseModel):
    quantity: int


@app.post("/__test__/validate")
async def _validate_route(body: _Body):  # pragma: no cover - body is never valid here
    return {"ok": True}


@app.get("/__test__/app-error")
async def _app_error_route():
    raise AppError(
        code="DISCOUNT_NOT_EVALUABLE",
        message="A discount cannot be evaluated.",
        status=422,
        details={"reason": "reference_missing"},
    )


@app.get("/__test__/db-down")
async def _db_down_route():
    raise OperationalError("SELECT 1", {}, Exception("connection refused"))


@app.get("/__test__/boom")
async def _boom_route():
    raise RuntimeError("unexpected failure")


client = TestClient(app)


def _assert_error_shape(body: dict) -> dict:
    """Assert body matches {"error": {code, message, status, details}} exactly."""
    assert set(body.keys()) == {"error"}
    err = body["error"]
    assert set(err.keys()) == {"code", "message", "status", "details"}
    assert isinstance(err["code"], str) and err["code"]
    assert isinstance(err["message"], str) and err["message"]
    assert isinstance(err["status"], int)
    assert isinstance(err["details"], dict)
    return err


def test_error_payload_serializes_to_contract():
    payload = ErrorPayload.build(code="X", message="m", status=418)
    assert payload.model_dump() == {
        "error": {"code": "X", "message": "m", "status": 418, "details": {}}
    }


def test_validation_error_returns_422_structured_payload():
    resp = client.post("/__test__/validate", json={"quantity": "not-an-int"})
    assert resp.status_code == 422
    err = _assert_error_shape(resp.json())
    assert err["code"] == "VALIDATION_ERROR"
    assert err["status"] == 422


def test_app_error_maps_to_its_status():
    resp = client.get("/__test__/app-error")
    assert resp.status_code == 422
    err = _assert_error_shape(resp.json())
    assert err["code"] == "DISCOUNT_NOT_EVALUABLE"
    assert err["status"] == 422
    assert err["details"] == {"reason": "reference_missing"}


def test_db_operational_error_returns_503_with_retry_message():
    resp = client.get("/__test__/db-down")
    assert resp.status_code == 503
    err = _assert_error_shape(resp.json())
    assert err["code"] == "DATABASE_UNAVAILABLE"
    assert err["status"] == 503
    assert "try again" in err["message"].lower()


def test_unhandled_exception_returns_500_structured_payload():
    # ServerErrorMiddleware re-raises after handling the 500, so disable
    # re-raising to observe the response the client would actually receive.
    safe_client = TestClient(app, raise_server_exceptions=False)
    resp = safe_client.get("/__test__/boom")
    assert resp.status_code == 500
    err = _assert_error_shape(resp.json())
    assert err["code"] == "INTERNAL_SERVER_ERROR"
    assert err["status"] == 500
