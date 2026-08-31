"""Verification for the OFF/result caching layer (Task 7.2).

These tests exercise ``app.services.data_service`` without a live Redis and
without the network:

* the Redis ``cache_get`` / ``cache_set`` used by the service are replaced with
  an in-memory dict-backed fake (via ``monkeypatch``), and
* OFF access is driven through an :class:`httpx.MockTransport`.

They demonstrate the four required behaviours of :func:`get_off_product_cached`:

* (a) cache miss -> OFF fetched -> value cached,
* (b) second call is a cache hit served WITHOUT an OFF call,
* (c) OFF unavailable but a cached value exists -> the cached value is returned,
* (d) OFF unavailable and no cache -> a data-unavailable result,

plus the determinism of the generic :func:`cached_or_compute` helper: repeated
requests are served from cache and a cached response equals a freshly computed
one (Req 9.4, 12.3).
"""

import asyncio
import json

import httpx
import pytest

from app.services import data_service
from app.services.data_service import (
    REASON_SERVER_ERROR,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    cached_or_compute,
    get_off_product_cached,
    off_product_cache_key,
)


class FakeCache:
    """A minimal in-memory stand-in for the Redis get/set helpers.

    Stores serialised strings exactly like ``redis_client.cache_set`` would and
    counts calls so tests can assert cache vs OFF usage. The TTL is accepted and
    ignored - expiry is irrelevant within a single test.
    """

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.get_calls = 0
        self.set_calls = 0

    def get(self, key: str):
        self.get_calls += 1
        return self.store.get(key)

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        self.set_calls += 1
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


def _ok_transport(counter: dict, product: dict) -> httpx.MockTransport:
    """A transport that returns a successful OFF product body and counts calls."""

    def handler(request: httpx.Request) -> httpx.Response:
        counter["count"] += 1
        return httpx.Response(200, json={"status": 1, "product": product})

    return httpx.MockTransport(handler)


def _down_transport(counter: dict, status: int = 503) -> httpx.MockTransport:
    """A transport that always fails with a 5xx and counts calls."""

    def handler(request: httpx.Request) -> httpx.Response:
        counter["count"] += 1
        return httpx.Response(status, text="service unavailable")

    return httpx.MockTransport(handler)


def test_a_cache_miss_fetches_from_off_and_stores(cache):
    """(a) A cold cache triggers one OFF call and stores the product for 24h."""
    off = {"count": 0}
    barcode = "3017620422003"
    product = {"product_name": "Nutella", "brands": "Ferrero"}

    result = _run(
        get_off_product_cached(
            barcode,
            transport=_ok_transport(off, product),
            backoff_base_seconds=0.0,
        )
    )

    assert result["status"] == STATUS_OK
    assert result["product"] == product
    assert off["count"] == 1  # OFF was consulted exactly once
    # The product was cached under the documented key as JSON.
    key = off_product_cache_key(barcode)
    assert key in cache.store
    assert json.loads(cache.store[key]) == product
    assert cache.set_calls == 1


def test_b_second_call_is_cache_hit_without_off(cache):
    """(b) A repeat lookup is served from cache and never calls OFF again."""
    off = {"count": 0}
    barcode = "111"
    product = {"product_name": "Amul Butter", "brands": "Amul"}
    transport = _ok_transport(off, product)

    first = _run(
        get_off_product_cached(barcode, transport=transport, backoff_base_seconds=0.0)
    )
    second = _run(
        get_off_product_cached(barcode, transport=transport, backoff_base_seconds=0.0)
    )

    assert first["status"] == STATUS_OK
    assert second["status"] == STATUS_OK
    assert second["product"] == product
    # Only the first (miss) call reached OFF; the second was a pure cache hit.
    assert off["count"] == 1


def test_c_off_unavailable_falls_back_to_cached(cache):
    """(c) When OFF is unavailable and a cached product exists, return it."""
    barcode = "222"
    product = {"product_name": "Tata Salt", "brands": "Tata"}
    key = off_product_cache_key(barcode)
    # Seed the cache as if a previous successful lookup had stored the product.
    cache.store[key] = json.dumps(product, sort_keys=True)

    off = {"count": 0}
    # force_refresh skips the initial cache read so OFF is consulted despite the
    # cached value; OFF is down, so the wrapper must fall back to the cache.
    result = _run(
        get_off_product_cached(
            barcode,
            force_refresh=True,
            transport=_down_transport(off),
            backoff_base_seconds=0.0,
        )
    )

    assert off["count"] >= 1  # OFF was attempted (and failed)
    assert result["status"] == STATUS_OK  # fell back to the cached product
    assert result["product"] == product


def test_d_off_unavailable_and_no_cache_returns_unavailable(cache):
    """(d) OFF unavailable with a cold cache yields a data-unavailable result."""
    off = {"count": 0}

    result = _run(
        get_off_product_cached(
            "333",
            transport=_down_transport(off),
            backoff_base_seconds=0.0,
        )
    )

    assert result["status"] == STATUS_UNAVAILABLE
    assert result["product"] is None
    assert result["reason"] == REASON_SERVER_ERROR
    # Retries were bounded (1 + 2) by the underlying Task 7.1 client.
    assert off["count"] == 3


def test_cached_or_compute_serves_repeats_from_cache_and_is_deterministic(cache):
    """cached_or_compute computes once, then serves an identical cached value."""
    calls = {"count": 0}

    def compute():
        calls["count"] += 1
        return {"score": 42, "band": "likely_inflated", "items": [1, 2, 3]}

    key = "discount:electronics:1499.0:4999.0"

    first = cached_or_compute(key, 3600, compute)
    second = cached_or_compute(key, 3600, compute)

    assert calls["count"] == 1  # computed once; repeat served from cache (Req 12.3)
    assert first == second  # cached response equals fresh computation (Req 9.4, 12.3)
    assert second == {"score": 42, "band": "likely_inflated", "items": [1, 2, 3]}
