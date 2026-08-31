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

The ``/api/v1/data-sources`` disclosure endpoint (Req 10.2-10.4) also lives in
this module. Because the router is mounted at the application root (no prefix),
that route declares its full ``/api/v1/data-sources`` path itself rather than
relying on a router prefix.
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


# ---------------------------------------------------------------------------
# Data-sources disclosure endpoint (Req 10.2, 10.3, 10.4)
#
# The platform's integrity claim rests on being honest about where its data
# comes from and what it cannot do. This endpoint is the machine-readable
# backing for the frontend's "Data Sources" panel: an accessible description of
# each data source and its known limitations (Req 10.2), including the
# crowd-sourced Open Food Facts notice (Req 10.3), the category-level /
# snapshot-data buy-timing disclosure (Req 10.1), the transparent
# weak-supervision labelling caveat, and the explicit statement that live
# scraping of Amazon/Flipkart is not a core data source (Req 10.4).
#
# The content is static disclosure text - it touches neither the database nor
# Redis - so the handler is a pure function of module constants. The individual
# disclosure statements are module constants so tests can assert them by
# identity rather than by brittle substring matching.
# ---------------------------------------------------------------------------

# Req 10.1: buy-timing predictions are category-level and snapshot-derived.
DISCLOSURE_BUY_TIMING_CATEGORY_LEVEL = (
    "Buy-timing recommendations are made at the category level - for a product "
    "category as a whole, not for an individual product on a single future "
    "date - and are derived from point-in-time snapshot data rather than a "
    "continuous per-product price history."
)

# Req 10.3: Open Food Facts data is crowd-sourced and may be incomplete.
DISCLOSURE_OFF_CROWD_SOURCED = (
    "Product attributes retrieved from Open Food Facts are crowd-sourced and "
    "may be incomplete, inconsistent, or missing for some products."
)

# The discount labels are a transparent heuristic, not ground-truth labels.
DISCLOSURE_WEAK_SUPERVISION_LABELS = (
    "Discount-genuineness labels are produced by a transparent weak-supervision "
    "heuristic derived from category price distributions, not from ground-truth "
    "'fake discount' labels, so classifications are indicative rather than "
    "definitive."
)

# Req 10.4: live scraping of Amazon/Flipkart is not a core data source.
DISCLOSURE_NO_LIVE_SCRAPING = (
    "Live scraping of Amazon and Flipkart is not used as a core data source; "
    "platform prices come from ingested public datasets and the Open Food Facts "
    "API."
)

#: The data sources the platform draws on, each with its known limitations
#: (Req 10.2). Kept as a module constant so the disclosure is defined once.
DATA_SOURCES: list[dict[str, object]] = [
    {
        "name": "Amazon Sales Dataset (Kaggle)",
        "type": "dataset",
        "origin": "kaggle",
        "access": "bulk_csv",
        "description": (
            "Publicly available Amazon India product listings ingested once "
            "from Kaggle, providing displayed price, reference price, discount, "
            "rating, rating count, category, and pack quantity."
        ),
        "limitations": [
            "Point-in-time snapshot data, not a continuous per-product price "
            "history.",
        ],
    },
    {
        "name": "Flipkart Products Dataset (Kaggle)",
        "type": "dataset",
        "origin": "kaggle",
        "access": "bulk_csv",
        "description": (
            "Publicly available Flipkart product listings ingested once from "
            "Kaggle, providing price, category, and related product attributes."
        ),
        "limitations": [
            "Point-in-time snapshot data, not a continuous per-product price "
            "history.",
        ],
    },
    {
        "name": "Open Food Facts",
        "type": "api",
        "origin": "open_food_facts",
        "access": "public_api",
        "crowd_sourced": True,
        "description": (
            "A free, public, crowd-sourced product database queried live (and "
            "cached) for product name, brand, quantity, and category."
        ),
        "limitations": [
            DISCLOSURE_OFF_CROWD_SOURCED,
        ],
    },
]


@router.get("/api/v1/data-sources")
def data_sources() -> dict[str, object]:
    """Describe the platform's data sources and known limitations (Req 10.2-10.4).

    Returns a static disclosure document backing the frontend's Data Sources
    panel. The body lists each data source with its known limitations (Req 10.2)
    and a ``disclosures`` block carrying the four honest-limitations statements:
    the crowd-sourced Open Food Facts notice (Req 10.3), the category-level /
    snapshot-data buy-timing disclosure (Req 10.1), the transparent
    weak-supervision labelling caveat, and the explicit no-live-scraping
    statement (Req 10.4). Shape::

        {
            "data_sources": [{"name", "type", "origin", "description",
                              "limitations", ...}, ...],
            "disclosures": {
                "buy_timing_category_level": str,   # Req 10.1
                "open_food_facts_crowd_sourced": str,  # Req 10.3
                "discount_labels_weak_supervision": str,
                "no_live_scraping": str,            # Req 10.4
            },
            "limitations": [str, ...],  # flat list mirroring the disclosures
        }

    The path is declared in full because the meta router is mounted at the
    application root without a prefix, so the resolved path is
    ``/api/v1/data-sources`` (Req 14.4).
    """

    disclosures = {
        "buy_timing_category_level": DISCLOSURE_BUY_TIMING_CATEGORY_LEVEL,
        "open_food_facts_crowd_sourced": DISCLOSURE_OFF_CROWD_SOURCED,
        "discount_labels_weak_supervision": DISCLOSURE_WEAK_SUPERVISION_LABELS,
        "no_live_scraping": DISCLOSURE_NO_LIVE_SCRAPING,
    }
    return {
        "data_sources": DATA_SOURCES,
        "disclosures": disclosures,
        # A flat list so a consumer can render every limitation without walking
        # the structured blocks above.
        "limitations": list(disclosures.values()),
    }
