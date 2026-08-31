"""Meta endpoints: liveness plus dependency health checks.

Exposes the root-level ``GET /health`` endpoint. Beyond simple liveness, it
probes the two backing services the platform depends on - PostgreSQL and Redis
- and reports a success status only when both are operational (Req 16.1).

The endpoint is designed to never raise: a failed dependency is caught and
reported as ``"down"`` with an overall service-unavailable (503) status and a
retry-friendly body, rather than surfacing an unhandled error (Req 16.4). This
also means it deliberately does not rely on the central ``OperationalError``
handler in ``app.main`` - it owns its own failure reporting so the response
always carries the ``{"status", "checks"}`` contract below.

The ``/api/v1/data-sources`` disclosure endpoint (Req 10.2-10.4) is added to
this same module by a later task and is intentionally absent for now.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.db.redis_client import get_redis_client
from app.db.session import SessionLocal

logger = logging.getLogger(__name__)

# No prefix: the router is mounted at the application root so the path resolves
# at ``/health`` rather than under ``/api/v1`` (see the design endpoint table).
router = APIRouter(tags=["Meta"])


def _check_database() -> bool:
    """Return ``True`` when a trivial query succeeds against the database.

    Runs ``SELECT 1`` on a short-lived session. Any failure - the database being
    unreachable, a driver or connection error, etc. - is caught and reported as
    a failed check so the health endpoint never raises (Req 16.4).
    """

    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            return True
        finally:
            db.close()
    except Exception:  # noqa: BLE001 - a liveness probe must never raise
        logger.warning(
            "Health check: database connectivity probe failed", exc_info=True
        )
        return False


def _check_redis() -> bool:
    """Return ``True`` when the Redis server answers a ``PING``.

    Any failure - server unreachable, connection error, etc. - is caught and
    reported as a failed check so the health endpoint never raises (Req 16.4).
    """

    try:
        return bool(get_redis_client().ping())
    except Exception:  # noqa: BLE001 - a liveness probe must never raise
        logger.warning(
            "Health check: redis connectivity probe failed", exc_info=True
        )
        return False


@router.get("/health")
def health_check() -> JSONResponse:
    """Report liveness plus DB and Redis connectivity (Req 16.1, 16.4).

    Returns HTTP 200 with ``status: "healthy"`` when both dependencies are
    operational, and HTTP 503 with ``status: "unhealthy"`` when either is down.
    The body shape is identical in both cases so a client parses one contract::

        {"status": "healthy"|"unhealthy",
         "service": "price-truth-api",
         "checks": {"database": "up"|"down", "redis": "up"|"down"}}

    Defined as a synchronous handler so Starlette runs the blocking DB/Redis
    probes in a worker thread instead of on the event loop.
    """

    database_up = _check_database()
    redis_up = _check_redis()
    healthy = database_up and redis_up

    body = {
        "status": "healthy" if healthy else "unhealthy",
        "service": "price-truth-api",
        "checks": {
            "database": "up" if database_up else "down",
            "redis": "up" if redis_up else "down",
        },
    }
    return JSONResponse(status_code=200 if healthy else 503, content=body)
