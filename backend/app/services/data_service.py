"""Data Service - Open Food Facts (OFF) HTTP client foundation.

This module owns the low-level access to the Open Food Facts public product
API. Task 7.1 implements only the *transport* concerns:

* build the product-read URL from configuration
  (``GET {OFF_BASE_URL}/api/{OFF_VERSION}/product/{barcode}.json``),
* send a custom ``User-Agent`` as OFF requires,
* apply a 5-second timeout (Req 9.2), and
* retry at most twice (up to three attempts total) on timeout / transport /
  5xx failures before returning a *data-unavailable* result instead of raising
  (Req 15.2).

The public entry point :func:`fetch_off_product` always returns a small,
structured result::

    {"status": "ok" | "unavailable", "product": <raw json | None>, "reason": <str | None>}

It never raises for the failure modes above, so a caller (and the later
feature modules) can branch on ``status`` deterministically.

Extension points (later tasks)
------------------------------
This module is deliberately split so tasks 7.2 and 7.3 extend it without
touching the transport logic:

* **Task 7.2 (Redis caching)** wraps :func:`fetch_off_product` with a
  cache-first lookup and, on ``"unavailable"``, a cached-value fallback. It can
  reuse :func:`ok_result` / :func:`unavailable_result` to keep the result shape
  identical.
* **Task 7.3 (validation + missing-field handling)** consumes the ``product``
  payload of an ``"ok"`` result, validates each value against its expected
  type/range, marks missing fields unavailable, and flags the crowd-sourced
  origin. It does not need to change how bytes are fetched.

Requirements: 9.2 (5-second timeout), 15.2 (<=2 retries, then data-unavailable).
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, TypedDict
from urllib.parse import quote

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

__all__ = [
    "fetch_off_product",
    "ok_result",
    "unavailable_result",
    "OffResult",
    "USER_AGENT",
    "OFF_TIMEOUT_SECONDS",
    "MAX_RETRIES",
    "STATUS_OK",
    "STATUS_UNAVAILABLE",
    "REASON_NOT_FOUND",
    "REASON_TIMEOUT",
    "REASON_TRANSPORT_ERROR",
    "REASON_SERVER_ERROR",
    "REASON_UNEXPECTED_STATUS",
    "REASON_INVALID_RESPONSE",
    "REASON_INVALID_BARCODE",
]

logger = get_logger(__name__)

# --- Transport constants ---------------------------------------------------

#: Custom identification header OFF requires on every request. The contact
#: placeholder is intentionally non-sensitive and can be replaced via a real
#: contact address without any behavioural change.
USER_AGENT = "PriceTruth/1.0 (contact-email)"

#: Hard request timeout in seconds, applied to connect/read/write/pool
#: (Req 9.2). A call that exceeds this is treated as a retryable failure.
OFF_TIMEOUT_SECONDS = 5.0

#: Maximum number of *retries* after the first attempt (Req 15.2). Total
#: attempts therefore never exceed ``MAX_RETRIES + 1`` (= 3).
MAX_RETRIES = 2

#: Base delay for exponential backoff between attempts. Kept small; tests pass
#: ``0.0`` to run without waiting.
_DEFAULT_BACKOFF_BASE_SECONDS = 0.5

# --- Result vocabulary -----------------------------------------------------

STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"

# Machine-readable reasons attached to an "unavailable" result.
REASON_NOT_FOUND = "not_found"
REASON_TIMEOUT = "timeout"
REASON_TRANSPORT_ERROR = "transport_error"
REASON_SERVER_ERROR = "server_error"
REASON_UNEXPECTED_STATUS = "unexpected_status"
REASON_INVALID_RESPONSE = "invalid_response"
REASON_INVALID_BARCODE = "invalid_barcode"


class OffResult(TypedDict):
    """Structured outcome of an OFF product fetch.

    Attributes:
        status: ``"ok"`` when a product payload was retrieved, otherwise
            ``"unavailable"``.
        product: The raw OFF product JSON on success, else ``None``. Later
            tasks validate and shape this payload.
        reason: A machine-readable cause when ``status`` is ``"unavailable"``,
            else ``None``.
    """

    status: str
    product: Optional[Any]
    reason: Optional[str]


def ok_result(product: Any) -> OffResult:
    """Build a successful result carrying the raw OFF product payload."""

    return {"status": STATUS_OK, "product": product, "reason": None}


def unavailable_result(reason: str) -> OffResult:
    """Build a data-unavailable result carrying a machine-readable reason."""

    return {"status": STATUS_UNAVAILABLE, "product": None, "reason": reason}


def _build_product_url(barcode: str) -> str:
    """Compose the OFF product-read URL from configuration.

    Base URL and API version come from :func:`app.core.config.get_settings`, so
    the endpoint (and version, e.g. ``v2`` -> ``v3``) is changeable purely via
    environment variables. The barcode is percent-encoded (``safe=""``) so it
    cannot break out of the URL path.
    """

    settings = get_settings()
    base = settings.OFF_BASE_URL.rstrip("/")
    version = settings.OFF_VERSION.strip("/")
    return f"{base}/api/{version}/product/{quote(barcode, safe='')}.json"


def _build_client(
    *, timeout: float, transport: Optional[httpx.AsyncBaseTransport]
) -> httpx.AsyncClient:
    """Create an ``AsyncClient`` with the OFF timeout and identification header.

    An optional ``transport`` lets tests inject an :class:`httpx.MockTransport`
    so retry/timeout behaviour is verified without touching the network.
    """

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    timeout_config = httpx.Timeout(timeout)
    if transport is not None:
        return httpx.AsyncClient(
            timeout=timeout_config, headers=headers, transport=transport
        )
    return httpx.AsyncClient(timeout=timeout_config, headers=headers)


def _interpret_success(barcode: str, response: httpx.Response) -> OffResult:
    """Turn a 2xx OFF response into a structured result.

    OFF signals "product not found" with ``status == 0`` even on HTTP 200, so
    that case is mapped to an unavailable/``not_found`` result. A malformed
    (non-JSON) body yields ``invalid_response`` rather than an exception.
    """

    try:
        data = response.json()
    except ValueError:
        logger.warning(
            "off_invalid_json_response",
            extra={"barcode": barcode, "status_code": response.status_code},
        )
        return unavailable_result(REASON_INVALID_RESPONSE)

    if isinstance(data, dict) and data.get("status") == 0:
        logger.info("off_product_not_found", extra={"barcode": barcode})
        return unavailable_result(REASON_NOT_FOUND)

    # OFF wraps the product under a "product" key; return that payload when
    # present, otherwise fall back to the whole body so nothing is lost for the
    # validation task (7.3) that consumes this.
    product = data.get("product") if isinstance(data, dict) else None
    if product is None:
        product = data
    return ok_result(product)


def _log_attempt_failure(
    *,
    barcode: str,
    attempt: int,
    max_attempts: int,
    reason: str,
    status_code: Optional[int] = None,
    error: Optional[BaseException] = None,
) -> None:
    """Record a single failed attempt with structured context (Req 15.2)."""

    extra: dict[str, Any] = {
        "barcode": barcode,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "reason": reason,
        "will_retry": attempt < max_attempts,
    }
    if status_code is not None:
        extra["status_code"] = status_code
    if error is not None:
        extra["error"] = type(error).__name__
    logger.warning("off_request_attempt_failed", extra=extra)


async def _sleep_backoff(base_seconds: float, attempt: int) -> None:
    """Wait between attempts using exponential backoff (skipped when base<=0)."""

    if base_seconds <= 0:
        return
    await asyncio.sleep(base_seconds * (2 ** (attempt - 1)))


async def fetch_off_product(
    barcode: str,
    *,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    timeout: float = OFF_TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
    backoff_base_seconds: float = _DEFAULT_BACKOFF_BASE_SECONDS,
) -> OffResult:
    """Fetch a product from Open Food Facts with a timeout and bounded retries.

    Calls ``GET {OFF_BASE_URL}/api/{OFF_VERSION}/product/{barcode}.json`` with a
    custom ``User-Agent`` and a ``timeout``-second cap (Req 9.2). Timeout,
    transport (network), and 5xx responses are retried up to ``max_retries``
    times (Req 15.2); once retries are exhausted the call returns an
    ``"unavailable"`` result instead of raising. A 404 (or an OFF
    ``status == 0`` body) is a definitive "not found" and is *not* retried.

    Args:
        barcode: The product barcode to look up. Blank input short-circuits to
            an ``invalid_barcode`` result without any network call.
        transport: Optional transport override for testing (e.g.
            :class:`httpx.MockTransport`).
        timeout: Per-request timeout in seconds (defaults to 5, Req 9.2).
        max_retries: Maximum retries after the first attempt (defaults to 2,
            Req 15.2).
        backoff_base_seconds: Base delay for exponential backoff between
            attempts; ``0`` disables waiting.

    Returns:
        An :class:`OffResult`. ``status`` is ``"ok"`` with the raw product
        payload, or ``"unavailable"`` with a machine-readable ``reason``.
    """

    barcode_str = (barcode or "").strip()
    if not barcode_str:
        logger.warning("off_invalid_barcode", extra={"barcode": barcode})
        return unavailable_result(REASON_INVALID_BARCODE)

    url = _build_product_url(barcode_str)
    max_attempts = max_retries + 1
    last_reason = REASON_TRANSPORT_ERROR

    async with _build_client(timeout=timeout, transport=transport) as client:
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.get(url)
            except httpx.TimeoutException as exc:
                # A read/connect timeout (Req 9.2) - retryable.
                last_reason = REASON_TIMEOUT
                _log_attempt_failure(
                    barcode=barcode_str,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    reason=last_reason,
                    error=exc,
                )
            except httpx.TransportError as exc:
                # Network/connection failure - retryable.
                last_reason = REASON_TRANSPORT_ERROR
                _log_attempt_failure(
                    barcode=barcode_str,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    reason=last_reason,
                    error=exc,
                )
            else:
                code = response.status_code
                if 200 <= code < 300:
                    return _interpret_success(barcode_str, response)
                if code == 404:
                    logger.info(
                        "off_product_not_found",
                        extra={"barcode": barcode_str, "status_code": code},
                    )
                    return unavailable_result(REASON_NOT_FOUND)
                if code >= 500:
                    # Server-side error - retryable (Req 15.2).
                    last_reason = REASON_SERVER_ERROR
                    _log_attempt_failure(
                        barcode=barcode_str,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        reason=last_reason,
                        status_code=code,
                    )
                else:
                    # 3xx or non-404 4xx: definitive, not retried.
                    logger.warning(
                        "off_unexpected_status",
                        extra={"barcode": barcode_str, "status_code": code},
                    )
                    return unavailable_result(REASON_UNEXPECTED_STATUS)

            # Reached only after a retryable failure; back off before retrying.
            if attempt < max_attempts:
                await _sleep_backoff(backoff_base_seconds, attempt)

        # All attempts exhausted without a usable response (Req 15.2).
        logger.error(
            "off_request_exhausted",
            extra={
                "barcode": barcode_str,
                "attempts": max_attempts,
                "reason": last_reason,
            },
        )
        return unavailable_result(last_reason)
