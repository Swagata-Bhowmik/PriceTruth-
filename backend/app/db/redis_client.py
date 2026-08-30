"""Redis client and small cache helpers.

Redis is the platform's cache layer (OFF product lookups, computed discount
results, category statistics, and so on). This module exposes a lazily
constructed client plus two thin helpers, :func:`cache_get` and
:func:`cache_set`, that the service layer uses without needing to know how
the connection is managed (Req 17.5).

The connection URL is read only from :func:`app.core.config.get_settings`
(Req 13.1). The client is created on first use rather than at import time,
and ``redis-py`` itself connects lazily on the first command, so importing
this module never requires a live Redis server.
"""

from __future__ import annotations

from typing import Optional

import redis

from app.core.config import get_settings

# Process-wide client, created on first use. ``redis.Redis`` maintains an
# internal connection pool, so a single instance is safe to reuse across
# requests and threads.
_client: Optional["redis.Redis"] = None


def get_redis_client() -> "redis.Redis":
    """Return the process-wide Redis client, creating it on first use.

    ``from_url`` does not open a socket; the connection is established lazily
    when the first command runs. ``decode_responses=True`` makes reads return
    ``str`` values rather than ``bytes`` so callers work with plain strings.
    """

    global _client
    if _client is None:
        settings = get_settings()
        _client = redis.Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
        )
    return _client


def cache_get(key: str) -> Optional[str]:
    """Return the cached value for ``key``, or ``None`` if it is absent.

    Structured values are stored as serialised strings (for example JSON) by
    the caller; this helper returns whatever string was stored.
    """

    return get_redis_client().get(key)


def cache_set(key: str, value: str, ttl_seconds: int) -> None:
    """Store ``value`` under ``key`` with a time-to-live.

    Uses Redis ``SETEX`` so the entry expires automatically after
    ``ttl_seconds`` seconds, which is how the per-key cache validity periods
    from the design (OFF products, discount results, etc.) are enforced.
    """

    get_redis_client().setex(name=key, time=ttl_seconds, value=value)
