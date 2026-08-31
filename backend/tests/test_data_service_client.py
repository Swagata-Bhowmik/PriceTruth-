"""Focused verification for the OFF client transport behaviour (Task 7.1).

These tests exercise ``app.services.data_service.fetch_off_product`` without
touching the network by injecting an :class:`httpx.MockTransport`. They cover
exactly the Task 7.1 contract:

* a successful product response -> ``status == "ok"`` with the product payload,
  the correct URL, and the custom ``User-Agent`` (Req 9.2 client shape);
* repeated timeouts / 5xx -> at most 2 retries (<= 3 attempts total) then
  ``status == "unavailable"`` (Req 15.2);
* a 404 "product not found" body -> handled gracefully, no retries.

Each test drives the coroutine with ``asyncio.run`` so it runs regardless of
the project's pytest-asyncio configuration.
"""

import asyncio

import httpx

from app.services.data_service import (
    MAX_RETRIES,
    REASON_INVALID_BARCODE,
    REASON_NOT_FOUND,
    REASON_SERVER_ERROR,
    REASON_TIMEOUT,
    USER_AGENT,
    fetch_off_product,
)


def _run(coro):
    """Run an async coroutine to completion in a fresh event loop."""
    return asyncio.run(coro)


def test_successful_response_returns_ok_with_product_and_headers():
    """(a) A 200 product response -> status "ok" with the product payload."""
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["user_agent"] = request.headers.get("user-agent", "")
        return httpx.Response(
            200,
            json={
                "code": "3017620422003",
                "status": 1,
                "product": {"product_name": "Nutella", "brands": "Ferrero"},
            },
        )

    result = _run(
        fetch_off_product(
            "3017620422003",
            transport=httpx.MockTransport(handler),
            backoff_base_seconds=0.0,
        )
    )

    assert result["status"] == "ok"
    assert result["product"] == {"product_name": "Nutella", "brands": "Ferrero"}
    assert result["reason"] is None
    # URL is built from settings (base + version) and ends with the product path.
    assert captured["url"].endswith("/api/v2/product/3017620422003.json")
    # OFF requires a custom User-Agent on every request.
    assert captured["user_agent"] == USER_AGENT


def test_repeated_timeouts_exhaust_retries_then_unavailable():
    """(b) Persistent timeouts -> exactly 3 attempts (1 + 2 retries), unavailable."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ReadTimeout("simulated timeout", request=request)

    result = _run(
        fetch_off_product(
            "999",
            transport=httpx.MockTransport(handler),
            backoff_base_seconds=0.0,
        )
    )

    assert calls["count"] == MAX_RETRIES + 1 == 3
    assert result["status"] == "unavailable"
    assert result["product"] is None
    assert result["reason"] == REASON_TIMEOUT


def test_repeated_server_errors_exhaust_retries_then_unavailable():
    """(b) Persistent 5xx -> exactly 3 attempts then unavailable."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503, text="service unavailable")

    result = _run(
        fetch_off_product(
            "500",
            transport=httpx.MockTransport(handler),
            backoff_base_seconds=0.0,
        )
    )

    assert calls["count"] == 3
    assert result["status"] == "unavailable"
    assert result["reason"] == REASON_SERVER_ERROR


def test_transient_error_then_success_stops_retrying_early():
    """"At most" 2 retries: a 500 then a 200 succeeds on the second attempt."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            return httpx.Response(500)
        return httpx.Response(
            200, json={"status": 1, "product": {"product_name": "Recovered"}}
        )

    result = _run(
        fetch_off_product(
            "42",
            transport=httpx.MockTransport(handler),
            backoff_base_seconds=0.0,
        )
    )

    assert calls["count"] == 2  # recovered on first retry; no third attempt
    assert result["status"] == "ok"
    assert result["product"] == {"product_name": "Recovered"}


def test_404_not_found_body_handled_gracefully_without_retry():
    """(c) A 404 "product not found" body -> unavailable/not_found, no retries."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(
            404, json={"status": 0, "status_verbose": "product not found"}
        )

    result = _run(
        fetch_off_product(
            "000",
            transport=httpx.MockTransport(handler),
            backoff_base_seconds=0.0,
        )
    )

    assert calls["count"] == 1  # 404 is definitive - not retried
    assert result["status"] == "unavailable"
    assert result["product"] is None
    assert result["reason"] == REASON_NOT_FOUND


def test_200_status_zero_body_treated_as_not_found():
    """OFF signals not-found via status==0 even on HTTP 200 -> not_found."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"status": 0, "status_verbose": "product not found"}
        )

    result = _run(
        fetch_off_product(
            "111",
            transport=httpx.MockTransport(handler),
            backoff_base_seconds=0.0,
        )
    )

    assert result["status"] == "unavailable"
    assert result["reason"] == REASON_NOT_FOUND


def test_blank_barcode_short_circuits_without_network_call():
    """A blank barcode returns unavailable without invoking the transport."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"status": 1, "product": {}})

    result = _run(
        fetch_off_product(
            "   ",
            transport=httpx.MockTransport(handler),
            backoff_base_seconds=0.0,
        )
    )

    assert calls["count"] == 0
    assert result["status"] == "unavailable"
    assert result["reason"] == REASON_INVALID_BARCODE
