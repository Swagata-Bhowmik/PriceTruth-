"""
Price Truth - FastAPI Main Application Entry Point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from sqlalchemy.exc import OperationalError

from app.api.v1 import (
    buy_timing,
    cross_platform,
    dashboard,
    discount,
    meta,
    search,
    shrinkflation,
    unit_price,
)
from app.core.config import get_settings
from app.core.errors import AppError, ErrorPayload
from app.core.logging import get_logger
from app.ml.discount_model import get_model
from app.ml.explainer import get_explainer


# ---------------------------------------------------------------------------
# Startup / shutdown lifespan
#
# The discount model and its SHAP explainer are loaded exactly once per process
# and stashed on ``app.state`` so request handlers reuse a single instance
# instead of re-reading the pickle or rebuilding the explainer per call
# (Req 11.2, 11.3, 12.4). ``get_model`` / ``get_explainer`` are themselves
# memoized, so ``app.state`` simply references those cached singletons.
#
# A missing or unloadable model MUST NOT crash startup: both loaders already log
# a warning and return ``None``, and the block below defends further so the app
# still boots and serves every other feature (Req 15.1). Scoring/explanation
# just report as unavailable to their consumers.
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model + SHAP explainer once at startup into ``app.state``."""
    logger = get_logger(__name__)

    model = None
    explainer = None
    try:
        model = get_model()
    except Exception:  # noqa: BLE001 - loader already guards; stay boot-safe
        logger.warning(
            "Unexpected error loading the discount model at startup; "
            "genuineness scoring is disabled.",
            exc_info=True,
        )
        model = None

    if model is None:
        logger.warning(
            "Discount model unavailable at startup; scoring and SHAP "
            "explanations are disabled."
        )
    else:
        try:
            explainer = get_explainer()
        except Exception:  # noqa: BLE001 - builder already guards; stay boot-safe
            logger.warning(
                "Unexpected error building the SHAP explainer at startup; "
                "explanations are disabled.",
                exc_info=True,
            )
            explainer = None
        if explainer is None:
            logger.warning(
                "SHAP explainer unavailable at startup; explanations are disabled."
            )

    app.state.discount_model = model
    app.state.discount_explainer = explainer

    yield

    # No teardown needed: the cached singletons live for the process lifetime.


app = FastAPI(
    title="Price Truth API",
    description="ML-powered e-commerce discount verification and shrinkflation tracking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Security middleware: request rate limiting (Req 18.4) and CORS (Req 18.3)
#
# Rate limiting (Task 14.1, Req 18.4)
# -----------------------------------
# A ``slowapi`` limiter keyed by the client's remote address enforces a default
# per-client limit (``RATE_LIMIT``, default 60/minute). ``SlowAPIMiddleware``
# applies that default to every route; a client that exceeds it gets the
# structured 429 payload rendered by ``rate_limit_exceeded_handler`` below,
# which reuses ``ErrorPayload`` so a throttled request shares the single error
# contract every other failure uses (Req 15.3). The limiter is stashed on
# ``app.state`` because the middleware reads it from there per request.
#
# The limiter's ``enabled`` flag is wired to ``RATE_LIMIT_ENABLED`` so it can be
# switched off from the environment. When disabled the middleware is fully
# transparent (it just forwards the request), which is how the general test
# suite - a single client issuing far more than 60 requests/minute - runs
# unaffected.
#
# CORS (Task 14.2, Req 18.3)
# --------------------------
# Cross-origin requests are restricted to the single configured frontend origin
# (``CORS_ALLOWED_ORIGIN``) instead of the previous permissive ``"*"``. CORS is
# added last so it is the outermost middleware and therefore annotates every
# response - including a 429 from the rate limiter - with the appropriate
# cross-origin headers.
#
# Input validation (Req 18.1) needs no wiring here: every endpoint already
# validates its inputs at the Pydantic boundary (path/query/body models), so
# malformed input is rejected as a structured 422 before any handler runs.
# ---------------------------------------------------------------------------

_settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_settings.RATE_LIMIT],
    enabled=_settings.RATE_LIMIT_ENABLED,
)
# The SlowAPI middleware resolves the limiter from ``app.state`` on each request.
app.state.limiter = limiter


def rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Render a rate-limit breach as the structured 429 payload (Req 18.4, 15.3).

    Reuses :class:`~app.core.errors.ErrorPayload` so a throttled request returns
    the same ``{"error": {code, message, status, details}}`` shape as every
    other error, with a stable ``RATE_LIMIT_EXCEEDED`` code and HTTP 429.

    Defined as a synchronous function on purpose: ``SlowAPIMiddleware`` invokes
    the registered handler directly on its synchronous check path and falls back
    to slowapi's default handler for a coroutine, so a sync handler is required
    for this structured payload to actually be returned.
    """

    payload = ErrorPayload.build(
        code="RATE_LIMIT_EXCEEDED",
        message=(
            "Rate limit exceeded. Please slow down and try again in a little "
            "while."
        ),
        status=429,
        details={"limit": str(exc.limit.limit)},
    )
    return JSONResponse(status_code=payload.error.status, content=payload.model_dump())


# Added first -> innermost: the limit is checked just before the route runs.
app.add_middleware(SlowAPIMiddleware)
# Translate a limiter breach into the structured payload (Req 15.3, 18.4).
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# CORS: restrict cross-origin requests to the single configured frontend origin
# (Req 18.3). Added last -> outermost, so it annotates every response.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[_settings.CORS_ALLOWED_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Central exception handlers
#
# Every error the API returns is serialized into the single structured payload
# defined in app.core.errors, so all failures share one contract (Req 15.3).
# A dedicated handler maps a database-connectivity failure to a 503 with a
# retry message instead of surfacing an unhandled error (Req 16.4).
# ---------------------------------------------------------------------------


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Translate a raised domain error into the structured payload at its status."""
    payload = exc.to_payload()
    return JSONResponse(status_code=exc.status, content=payload.model_dump())


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return request-validation failures as a 422 structured payload (Req 15.3)."""
    payload = ErrorPayload.build(
        code="VALIDATION_ERROR",
        message="One or more request parameters failed validation.",
        status=422,
        details={"errors": jsonable_encoder(exc.errors())},
    )
    return JSONResponse(status_code=payload.error.status, content=payload.model_dump())


@app.exception_handler(OperationalError)
async def handle_db_operational_error(
    request: Request, exc: OperationalError
) -> JSONResponse:
    """Map a database-connectivity failure to a 503 with a retry message (Req 16.4)."""
    payload = ErrorPayload.build(
        code="DATABASE_UNAVAILABLE",
        message=(
            "The service is temporarily unable to reach its database. "
            "Please try again in a few moments."
        ),
        status=503,
    )
    return JSONResponse(status_code=payload.error.status, content=payload.model_dump())


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: convert any otherwise-unhandled error into a 500 payload (Req 15.3)."""
    payload = ErrorPayload.build(
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected internal error occurred.",
        status=500,
    )
    return JSONResponse(status_code=payload.error.status, content=payload.model_dump())


@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "message": "Price Truth API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }


# Root-level meta router. Registered without a prefix so GET /health resolves
# at the application root (liveness + DB/Redis connectivity check), not under
# /api/v1 (Req 16.1, 16.4). This real check replaces the former static
# placeholder that always reported healthy.
app.include_router(meta.router)

# Unit Price Comparator: POST /api/v1/unit-price/compare (Req 5.3, 14.4, 18.1).
app.include_router(unit_price.router, prefix="/api/v1", tags=["Unit Price"])

# Shrinkflation Timeline: GET /api/v1/shrinkflation/{product_id} (Req 4.1, 4.4, 14.4).
app.include_router(shrinkflation.router, prefix="/api/v1", tags=["Shrinkflation"])

# Cross-Platform Aggregator: GET /api/v1/cross-platform/{product_id} (Req 7.1, 7.3, 14.4).
app.include_router(cross_platform.router, prefix="/api/v1", tags=["Cross-Platform"])

# Buy Timing Signal: GET /api/v1/buy-timing/{category} (Req 6.1, 6.4, 14.4).
app.include_router(buy_timing.router, prefix="/api/v1", tags=["Buy Timing"])

# Product Search: GET /api/v1/search + POST /api/v1/manual-entry (Req 1.1, 1.5, 1.6, 14.4).
app.include_router(search.router, prefix="/api/v1", tags=["Search"])

# True Discount Checker: POST /api/v1/discount-check - genuineness score,
# classification, and the SHAP breakdown (Req 2.4, 3.1, 3.2, 3.3, 3.5, 11.3, 18.1).
app.include_router(discount.router, prefix="/api/v1", tags=["Discount Checker"])

# Composite Dashboard: GET /api/v1/dashboard/{product_id} - composes all five
# feature modules for a product with per-module containment (Req 8.1, 8.5, 15.1, 14.4).
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])


# Import routers (will be added in later phases)
# from app.api.v1 import discount_checker, shrinkflation, price_comparison, buy_timing
# app.include_router(discount_checker.router, prefix="/api/v1", tags=["Discount Checker"])
# app.include_router(shrinkflation.router, prefix="/api/v1", tags=["Shrinkflation"])
# app.include_router(price_comparison.router, prefix="/api/v1", tags=["Price Comparison"])
# app.include_router(buy_timing.router, prefix="/api/v1", tags=["Buy Timing"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
