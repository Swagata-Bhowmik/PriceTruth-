"""Unit tests for the ``GET /health`` endpoint (Task 1.6).

Focus is the health endpoint in ``app.api.v1.meta`` and the availability
contract it implements:

* **Req 16.1** - the endpoint returns a success status when the backend and its
  database (and Redis) connections are operational.
* **Req 16.4** - while a dependency is unreachable, the endpoint returns a
  service-unavailable (503) status rather than surfacing an unhandled failure.

The structured error-payload contract (Req 15.3) and the central exception
handlers are already exercised end-to-end in ``tests/test_error_handlers.py``
and ``tests/test_error_payload_property.py``, so those are intentionally not
duplicated here.

The endpoint probes the two backing services through the module-level helpers
``_check_database`` and ``_check_redis``. ``health_check`` calls those helpers
by their bare global names, so monkeypatching them on ``app.api.v1.meta``
substitutes the probe at call time - letting every scenario below run without a
live PostgreSQL or Redis instance.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.v1 import meta
from app.main import app

# ``raise_server_exceptions=True`` (the default) means that if the endpoint ever
# let an exception escape, ``client.get`` would re-raise it and fail the test.
# Every assertion that a request returns a normal 200/503 response therefore
# also proves the endpoint did not surface an unhandled failure (Req 16.4).
client = TestClient(app)

_EXPECTED_SERVICE = "price-truth-api"


def _set_checks(monkeypatch, *, database_up: bool, redis_up: bool) -> None:
    """Substitute both dependency probes with fixed boolean results.

    Patches the helpers on ``app.api.v1.meta`` so ``health_check`` resolves the
    stubs through its module globals, exercising the real handler logic without
    touching a live database or Redis.
    """

    monkeypatch.setattr(meta, "_check_database", lambda: database_up)
    monkeypatch.setattr(meta, "_check_redis", lambda: redis_up)


def test_health_returns_200_healthy_when_both_dependencies_up(monkeypatch):
    """Both dependencies operational -> 200 with the healthy contract (Req 16.1)."""
    _set_checks(monkeypatch, database_up=True, redis_up=True)

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "status": "healthy",
        "service": _EXPECTED_SERVICE,
        "checks": {"database": "up", "redis": "up"},
    }


def test_health_returns_503_unhealthy_when_database_down(monkeypatch):
    """Database down (Redis up) -> 503 with a per-dependency breakdown (Req 16.4).

    The request completing with a structured 503 body - rather than raising -
    is the "rather than an unhandled failure" guarantee from Req 16.4.
    """
    _set_checks(monkeypatch, database_up=False, redis_up=True)

    resp = client.get("/health")

    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "unhealthy"
    assert body["checks"] == {"database": "down", "redis": "up"}
    # Full contract shape is identical to the healthy path so clients parse one body.
    assert body == {
        "status": "unhealthy",
        "service": _EXPECTED_SERVICE,
        "checks": {"database": "down", "redis": "up"},
    }


def test_health_returns_503_unhealthy_when_redis_down(monkeypatch):
    """Redis down (database up) -> 503 with database up / redis down (Req 16.4)."""
    _set_checks(monkeypatch, database_up=True, redis_up=False)

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json() == {
        "status": "unhealthy",
        "service": _EXPECTED_SERVICE,
        "checks": {"database": "up", "redis": "down"},
    }


def test_health_returns_503_unhealthy_when_both_dependencies_down(monkeypatch):
    """Both dependencies down -> 503 with both marked down (Req 16.4)."""
    _set_checks(monkeypatch, database_up=False, redis_up=False)

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json() == {
        "status": "unhealthy",
        "service": _EXPECTED_SERVICE,
        "checks": {"database": "down", "redis": "down"},
    }


def test_check_database_swallows_errors_and_returns_false(monkeypatch):
    """A failing DB probe is caught and reported as down, never raised (Req 16.4).

    This is the source of the endpoint's never-raise guarantee: the helper
    catches any connectivity error and returns ``False`` instead of propagating.
    """

    def _boom_session():
        raise RuntimeError("database connection refused")

    monkeypatch.setattr(meta, "SessionLocal", _boom_session)

    assert meta._check_database() is False


def test_check_redis_swallows_errors_and_returns_false(monkeypatch):
    """A failing Redis probe is caught and reported as down, never raised (Req 16.4)."""

    def _boom_client():
        raise RuntimeError("redis unreachable")

    monkeypatch.setattr(meta, "get_redis_client", _boom_client)

    assert meta._check_redis() is False
