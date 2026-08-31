"""Security-control API tests (Task 14.3).

Covers the cross-cutting security hardening added in tasks 14.1 and 14.2 and
their governing acceptance criteria:

* **Rate limiting (Req 18.4).** A dedicated low-limit app instance proves that
  once a single client exceeds its allowance the next request is rejected with
  HTTP 429 carrying the structured ``RATE_LIMIT_EXCEEDED`` payload. A dedicated
  app is used (rather than the global app, whose limiter is intentionally
  disabled for the test suite) so the 429 behaviour is exercised deterministically
  while reusing the *production* exception handler
  (:func:`app.main.rate_limit_exceeded_handler`).
* **CORS (Req 18.3).** Against the real application, a request whose ``Origin``
  matches the single configured frontend origin gets that origin echoed in
  ``access-control-allow-origin``; a request from any other origin does not.
* **Injection treated as data (Req 18.2).** A SQL-injection-like search query is
  passed through the ``GET /api/v1/search`` endpoint backed by a seeded
  in-memory SQLite database. Because every query flows through parameter-bound
  ORM queries, the string is matched as *data* (it matches nothing) and the
  products table is left intact - nothing is dropped and no rows leak.
* **No secrets in responses (Req 18.6).** The root, health, and a sample error
  response are checked to ensure no credential material (the ``DATABASE_URL`` /
  ``REDIS_URL`` or the embedded password) appears in any body.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1 import meta
from app.core.config import get_settings
from app.db.models import Base, Product
from app.db.session import get_db
from app.main import app, rate_limit_exceeded_handler


# ===========================================================================
# Rate limiting (Req 18.4)
# ===========================================================================


def _build_rate_limited_app(limit: str = "2/minute") -> FastAPI:
    """Return a minimal app whose single route is capped at ``limit`` per client.

    Mirrors the production wiring in ``app.main`` - a remote-address-keyed
    ``Limiter`` with a default limit, ``SlowAPIMiddleware``, and the *same*
    structured ``RateLimitExceeded`` handler - but with the limiter explicitly
    enabled and a deliberately tiny limit so the breach can be triggered in a
    couple of requests. Its own limiter uses fresh in-memory storage, so it is
    isolated from the (disabled) global limiter and from other tests.
    """

    rate_limited_app = FastAPI()
    test_limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[limit],
        enabled=True,
    )
    rate_limited_app.state.limiter = test_limiter
    rate_limited_app.add_middleware(SlowAPIMiddleware)
    rate_limited_app.add_exception_handler(
        RateLimitExceeded, rate_limit_exceeded_handler
    )

    @rate_limited_app.get("/ping")
    def ping() -> dict[str, bool]:
        return {"ok": True}

    return rate_limited_app


def test_requests_within_limit_succeed():
    """Requests up to the limit return 200 (the limiter allows the allowance)."""
    client = TestClient(_build_rate_limited_app("2/minute"))

    first = client.get("/ping")
    second = client.get("/ping")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == {"ok": True}


def test_exceeding_limit_returns_structured_429():
    """The request past the limit is rejected with the structured 429 payload.

    With a ``2/minute`` limit the third request from the same client exceeds the
    allowance and must return HTTP 429 whose body is the single structured error
    contract (Req 15.3) with a stable ``RATE_LIMIT_EXCEEDED`` code (Req 18.4).
    """
    client = TestClient(_build_rate_limited_app("2/minute"))

    client.get("/ping")
    client.get("/ping")
    throttled = client.get("/ping")

    assert throttled.status_code == 429
    body = throttled.json()
    assert set(body["error"]) >= {"code", "message", "status", "details"}
    assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
    assert body["error"]["status"] == 429
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def test_global_limiter_disabled_for_test_suite():
    """The application-wide limiter is disabled so the suite is not throttled.

    ``conftest.py`` sets ``RATE_LIMIT_ENABLED=false`` before settings load, so
    the real app's limiter must report as disabled - this is what lets the rest
    of the 200+ test suite fire many requests from one client without tripping
    the 60/minute limit (Req 18.4 is still enforced in production, where the
    flag defaults to true).
    """
    assert app.state.limiter.enabled is False
    assert get_settings().RATE_LIMIT_ENABLED is False


# ===========================================================================
# CORS (Req 18.3)
# ===========================================================================


def test_cors_echoes_configured_origin():
    """A request from the configured frontend origin gets that origin echoed."""
    client = TestClient(app)
    allowed_origin = get_settings().CORS_ALLOWED_ORIGIN

    resp = client.get("/", headers={"Origin": allowed_origin})

    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == allowed_origin


def test_cors_preflight_allows_configured_origin():
    """An OPTIONS preflight from the configured origin is accepted and echoed."""
    client = TestClient(app)
    allowed_origin = get_settings().CORS_ALLOWED_ORIGIN

    resp = client.options(
        "/",
        headers={
            "Origin": allowed_origin,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert resp.headers.get("access-control-allow-origin") == allowed_origin


def test_cors_does_not_echo_disallowed_origin():
    """A request from a non-configured origin is not granted that origin.

    Starlette's CORS middleware simply omits the ``access-control-allow-origin``
    header for a disallowed origin, so the disallowed origin is never echoed
    back - cross-origin access is restricted to the configured frontend origin
    (Req 18.3).
    """
    client = TestClient(app)
    disallowed_origin = "https://malicious.example.com"

    resp = client.get("/", headers={"Origin": disallowed_origin})

    assert resp.headers.get("access-control-allow-origin") != disallowed_origin


# ===========================================================================
# Injection treated as data (Req 18.2)
# ===========================================================================


def _seed_products(session) -> None:
    """Seed three products so a successful search and a table-intact check work."""
    session.add_all(
        [
            Product(
                id="amul-butter-500",
                name="Amul Butter 500 g",
                normalized_name="amul butter 500 g",
                brand="Amul",
                category="fmcg/dairy",
            ),
            Product(
                id="amul-cheese-200",
                name="Amul Cheese Slices 200 g",
                normalized_name="amul cheese slices 200 g",
                brand="Amul",
                category="fmcg/dairy",
            ),
            Product(
                id="tata-salt-1kg",
                name="Tata Salt 1 kg",
                normalized_name="tata salt 1 kg",
                brand="Tata",
                category="fmcg/staples",
            ),
        ]
    )
    session.commit()


@pytest.fixture()
def seeded_client():
    """Yield ``(client, session_factory)`` over a seeded in-memory SQLite DB.

    A single ``StaticPool`` engine backs both the seed session and every
    request-scoped session so all requests observe the same ``:memory:``
    database; the ``get_db`` dependency is overridden accordingly and cleaned up
    on teardown.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    testing_session_local = sessionmaker(bind=engine, expire_on_commit=False)

    seed_session = testing_session_local()
    try:
        _seed_products(seed_session)
    finally:
        seed_session.close()

    def _override_get_db():
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(app), testing_session_local
    finally:
        app.dependency_overrides.clear()
        engine.dispose()


@pytest.mark.parametrize(
    "injection",
    [
        "'; DROP TABLE products; --",
        "' OR '1'='1",
        "amul'); DELETE FROM products; --",
    ],
)
def test_injection_string_is_treated_as_data(seeded_client, injection):
    """A SQL-injection-like query returns cleanly and never mutates the table.

    The endpoint responds 200 (no unhandled error), the injection matches no
    products (no rows leak), and the products table is left fully intact - proof
    that user input is bound as a parameter rather than concatenated into SQL
    (Req 18.2).
    """
    client, session_factory = seeded_client

    resp = client.get("/api/v1/search", params={"q": injection})

    # Returns cleanly - the string was handled as data, not executed.
    assert resp.status_code == 200
    body = resp.json()
    # No rows leaked: an injection payload matches no real product name.
    assert body["results"] == []

    # The table is intact: all three seeded rows still exist (nothing dropped
    # or deleted by the payload).
    check_session = session_factory()
    try:
        remaining = check_session.execute(select(Product)).scalars().all()
    finally:
        check_session.close()
    assert len(remaining) == 3


def test_legitimate_search_still_works_after_injection_attempts(seeded_client):
    """A normal query returns the seeded matches, confirming the table survived."""
    client, _ = seeded_client

    # Fire an injection attempt first, then a legitimate query.
    client.get("/api/v1/search", params={"q": "'; DROP TABLE products; --"})
    resp = client.get("/api/v1/search", params={"q": "amul"})

    assert resp.status_code == 200
    body = resp.json()
    assert {r["id"] for r in body["results"]} == {
        "amul-butter-500",
        "amul-cheese-200",
    }


# ===========================================================================
# No secrets in responses (Req 18.6)
# ===========================================================================


def _secret_markers() -> list[str]:
    """Return credential fragments that must never appear in any response body."""
    settings = get_settings()
    return [
        settings.DATABASE_URL,  # full connection URL (contains credentials)
        settings.REDIS_URL,
        "postgres:postgres",  # the embedded user:password fragment
    ]


def _assert_no_secrets(text: str) -> None:
    """Assert that no known credential fragment is present in ``text``."""
    for marker in _secret_markers():
        assert marker not in text


def test_root_response_contains_no_secrets():
    """``GET /`` exposes API status but no credential material (Req 18.6)."""
    client = TestClient(app)

    resp = client.get("/")

    assert resp.status_code == 200
    _assert_no_secrets(resp.text)


def test_health_response_contains_no_secrets(monkeypatch):
    """``GET /health`` reports connectivity without leaking credentials (Req 18.6).

    The dependency probes are stubbed so the endpoint returns deterministically
    (no live DB/Redis needed); the assertion is that the body carries none of
    the credential fragments regardless of the reported status.
    """
    monkeypatch.setattr(meta, "_check_database", lambda: True)
    monkeypatch.setattr(meta, "_check_redis", lambda: True)
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    _assert_no_secrets(resp.text)


def test_error_response_contains_no_secrets():
    """A structured error response carries no credential material (Req 18.6).

    A non-positive ``displayed_price`` is rejected at the Pydantic boundary as a
    structured 422; the payload must describe the validation failure without
    ever echoing connection strings or passwords.
    """
    client = TestClient(app)

    resp = client.post(
        "/api/v1/manual-entry",
        json={"name": "Valid Name", "displayed_price": 0},
    )

    assert resp.status_code == 422
    assert resp.json()["error"]["status"] == 422
    _assert_no_secrets(resp.text)
