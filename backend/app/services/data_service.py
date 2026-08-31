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
import hashlib
import json
from typing import Any, Callable, Optional, TypedDict
from urllib.parse import quote

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.redis_client import cache_get, cache_set

__all__ = [
    "fetch_off_product",
    "get_off_product_cached",
    "cached_or_compute",
    "cache_get_json",
    "cache_set_json",
    "off_product_cache_key",
    "search_cache_key",
    "discount_cache_key",
    "category_stats_cache_key",
    "cross_platform_cache_key",
    "ok_result",
    "unavailable_result",
    "OffResult",
    "USER_AGENT",
    "OFF_TIMEOUT_SECONDS",
    "OFF_CACHE_TTL_SECONDS",
    "SEARCH_CACHE_TTL_SECONDS",
    "DISCOUNT_CACHE_TTL_SECONDS",
    "CROSS_PLATFORM_CACHE_TTL_SECONDS",
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


# ===========================================================================
# Task 7.2 - Redis caching layer for OFF lookups and computed results
# ===========================================================================
#
# This section adds a cache-first wrapper around :func:`fetch_off_product`
# plus small, reusable JSON cache helpers the other feature services
# (discount / search / cross-platform / category stats) can share. It builds
# on ``app.db.redis_client`` (:func:`cache_get` / :func:`cache_set`) and never
# touches the transport logic above, so Task 7.1's behaviour is preserved.
#
# Design references: the "Redis Cache Model" table and the OFF client section
# of design.md. A cache value is a deterministic JSON function of its inputs,
# so a cached response equals a freshly computed one within the validity
# period (Req 9.4, 12.3); the OFF wrapper additionally falls back to a cached
# product when a live OFF call is unavailable (Req 9.2).

# --- Cache TTLs (seconds), mirroring the design's Redis Cache Model ---------

#: ``off:product:{barcode}`` -> validated OFF product JSON, 24h (Req 9.4, 12.3).
OFF_CACHE_TTL_SECONDS = 24 * 60 * 60
#: ``search:{sha1(query)}`` -> search results list, 1h (Req 12.3).
SEARCH_CACHE_TTL_SECONDS = 60 * 60
#: ``discount:{category}:{displayed}:{reference}`` -> score+band+SHAP, 1h
#: (Req 11.1, 12.3).
DISCOUNT_CACHE_TTL_SECONDS = 60 * 60
#: ``crossplatform:{product_id}`` -> platform price comparison, 6h (Req 12.3).
CROSS_PLATFORM_CACHE_TTL_SECONDS = 6 * 60 * 60


# --- Cache key builders (one per documented key pattern) --------------------


def off_product_cache_key(barcode: str) -> str:
    """Return the Redis key for a cached OFF product: ``off:product:{barcode}``."""

    return f"off:product:{barcode}"


def search_cache_key(query: str) -> str:
    """Return the Redis key for a cached search result: ``search:{sha1(query)}``.

    The raw query is hashed so arbitrary user text becomes a fixed-length,
    injection-safe key component.
    """

    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()
    return f"search:{digest}"


def discount_cache_key(
    category: str, displayed_price: float, reference_price: float
) -> str:
    """Return the Redis key for a cached discount check.

    Shape: ``discount:{category}:{displayed_price}:{reference_price}`` - keyed
    purely by the inputs that determine the result (Req 12.3).
    """

    return f"discount:{category}:{displayed_price}:{reference_price}"


def category_stats_cache_key(category: str) -> str:
    """Return the Redis key for cached category statistics: ``category_stats:{category}``."""

    return f"category_stats:{category}"


def cross_platform_cache_key(product_id: str) -> str:
    """Return the Redis key for a cached cross-platform comparison.

    Shape: ``crossplatform:{product_id}``.
    """

    return f"crossplatform:{product_id}"


# --- Reusable JSON cache helpers -------------------------------------------
#
# These wrap the raw string get/set from ``redis_client`` with JSON
# (de)serialisation and best-effort error handling: a cache backend outage or
# a corrupt value must never break a request, so read failures degrade to a
# miss and write failures are skipped (Req 15.1 spirit). Callers depend only
# on the plain get/set names ``cache_get`` / ``cache_set`` at module scope,
# which keeps them trivially replaceable in tests.


def cache_get_json(key: str) -> Optional[Any]:
    """Return the JSON-decoded value cached under ``key``, or ``None``.

    ``None`` means "treat as a cache miss": the key is absent, the backend is
    unreachable, or the stored value is not valid JSON (which is logged).
    """

    try:
        raw = cache_get(key)
    except Exception:  # noqa: BLE001 - cache is best-effort; degrade to a miss
        logger.warning("cache_get_failed", extra={"key": key})
        return None

    if raw is None:
        return None

    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("cache_corrupt_value", extra={"key": key})
        return None


def cache_set_json(key: str, value: Any, ttl_seconds: int) -> Optional[Any]:
    """Serialise ``value`` to JSON and store it under ``key`` with a TTL.

    Returns the JSON-normalised value that was stored (i.e.
    ``json.loads(json.dumps(value))``) so callers can hand back a value that is
    identical to what a later cache hit will return - this is what makes a
    freshly computed response equal to its cached form (Req 9.4, 12.3). Returns
    ``None`` when the value cannot be serialised or the write fails, in which
    case nothing is cached.
    """

    try:
        serialized = json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        logger.warning("cache_unserializable_value", extra={"key": key})
        return None

    try:
        cache_set(key, serialized, ttl_seconds)
    except Exception:  # noqa: BLE001 - cache write is best-effort
        logger.warning("cache_set_failed", extra={"key": key})
        return None

    return json.loads(serialized)


def cached_or_compute(key: str, ttl_seconds: int, compute_fn: Callable[[], Any]) -> Any:
    """Return a cached JSON result for ``key`` or compute, cache, and return it.

    This is the generic result-cache the discount / search / cross-platform /
    category-stats services reuse so repeated requests for the same inputs are
    served from Redis rather than recomputed (Req 12.3). ``compute_fn`` is only
    invoked on a cache miss and must return a JSON-serialisable value.

    Determinism: on a miss the value is normalised through JSON before being
    returned, so the value returned on the computing call is identical to the
    value a subsequent cache hit returns (Req 9.4, 12.3). If the computed value
    cannot be serialised it is returned as-is and simply not cached.

    Args:
        key: Fully-qualified Redis key (use the key builders above).
        ttl_seconds: Cache validity period for this entry.
        compute_fn: Zero-argument callable producing the fresh result.

    Returns:
        The cached value on a hit, otherwise the freshly computed value.
    """

    cached = cache_get_json(key)
    if cached is not None:
        logger.debug("result_cache_hit", extra={"key": key})
        return cached

    fresh = compute_fn()
    normalized = cache_set_json(key, fresh, ttl_seconds)
    # ``normalized`` is None only when the value was not cacheable; fall back to
    # the raw computed value in that case.
    return normalized if normalized is not None else fresh


async def get_off_product_cached(
    barcode: str,
    *,
    force_refresh: bool = False,
    transport: Optional[httpx.AsyncBaseTransport] = None,
    ttl_seconds: int = OFF_CACHE_TTL_SECONDS,
    **fetch_kwargs: Any,
) -> OffResult:
    """Cache-first OFF product lookup with a cached-value failure fallback.

    Behaviour (Req 9.2, 9.4, 12.3):

    1. Unless ``force_refresh`` is set, look up ``off:product:{barcode}`` in
       Redis first; on a hit return the product **without** calling OFF.
    2. On a miss (or forced refresh) call :func:`fetch_off_product`. On an
       ``"ok"`` result, cache the product JSON under ``off:product:{barcode}``
       with a 24h TTL and return it.
    3. On an ``"unavailable"`` result, fall back to a cached product if one
       exists (Req 9.2 "return a cached result where a cache hit exists"),
       otherwise return the unavailable result.

    ``force_refresh`` supports revalidation while still degrading to the last
    good cached product if OFF is momentarily unavailable. A blank barcode is
    delegated straight to :func:`fetch_off_product` (which reports
    ``invalid_barcode`` without any network or cache access).

    Args:
        barcode: The product barcode to look up.
        force_refresh: When ``True``, skip the initial cache read and query OFF,
            still falling back to the cached product if OFF is unavailable.
        transport: Optional transport override for testing (e.g.
            :class:`httpx.MockTransport`).
        ttl_seconds: TTL for the cached product (defaults to 24h).
        **fetch_kwargs: Forwarded to :func:`fetch_off_product` (e.g. ``timeout``,
            ``max_retries``, ``backoff_base_seconds``).

    Returns:
        An :class:`OffResult`, identical in shape to :func:`fetch_off_product`.
    """

    barcode_str = (barcode or "").strip()
    if not barcode_str:
        # No valid key to cache under: let the transport layer report the
        # invalid barcode (no network call, no cache access).
        return await fetch_off_product(barcode, transport=transport, **fetch_kwargs)

    key = off_product_cache_key(barcode_str)

    # 1. Cache-first read (skipped on a forced refresh).
    if not force_refresh:
        cached = cache_get_json(key)
        if cached is not None:
            logger.debug("off_cache_hit", extra={"barcode": barcode_str})
            return ok_result(cached)

    # 2. Cache miss or forced refresh: consult OFF via the Task 7.1 client.
    result = await fetch_off_product(barcode_str, transport=transport, **fetch_kwargs)

    if result["status"] == STATUS_OK:
        # Store the product payload for 24h. Task 7.3 will validate/shape the
        # payload before this cache write; the transport result is used as-is
        # for now.
        cache_set_json(key, result["product"], ttl_seconds)
        return result

    # 3. OFF unavailable: serve the last cached product if we have one.
    cached = cache_get_json(key)
    if cached is not None:
        logger.info(
            "off_cache_fallback_on_unavailable",
            extra={"barcode": barcode_str, "reason": result["reason"]},
        )
        return ok_result(cached)

    return result
