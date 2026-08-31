"""Composite Dashboard API endpoint with per-module containment (Task 13.1).

Exposes ``GET /api/v1/dashboard/{product_id}`` - the single cohesive view that
presents all five feature modules for one selected product in one response
(Req 8.1). Unlike the per-feature endpoints (each of which owns one module),
this endpoint *composes* every feature service for a resolved product and
returns their results side by side so the frontend can render the whole
dashboard from a single call (Req 14.4).

The defining behaviour is **containment** (Req 15.1, 8.5): each feature module
is computed inside its own ``try``/``except`` so a failure - or an empty /
not-evaluable result - in one module can never break the others. A module's
slot is either its normal result ``dict`` or a uniform unavailable state
``{"available": False, "message": ...}``; a contained exception is logged
through :mod:`app.core.logging` and never propagates out of the endpoint. The
product itself is resolved first: if it does not exist the whole request is a
:class:`~app.core.errors.AppError` (``PRODUCT_NOT_FOUND``, HTTP 404), because
there is no dashboard to build without a product.

Division of responsibility
---------------------------
All feature logic lives in the individual services (each with its own unit and
property tests); this module only resolves the product, derives each module's
inputs, calls the service, and contains failures. The one piece of shaping it
borrows is the discount endpoint's rule for attaching the SHAP breakdown to a
*scored* discount result and dropping the raw engineered ``features`` (Req 3.1),
so the discount slot here matches ``POST /api/v1/discount-check``.

Per-module composition
-----------------------
* ``product`` - the resolved product's ``id``, ``name``, ``brand``, ``category``.
* ``discount`` - the price pair is taken from the product's most relevant
  ``price_snapshots`` row and scored by
  :func:`app.services.discount_service.check_discount`; a scored result gets the
  SHAP ``explanation`` attached. A missing snapshot, or the not-evaluable
  pre-condition (:class:`AppError` ``DISCOUNT_NOT_EVALUABLE``), degrades to an
  unavailable slot instead of a 500.
* ``shrinkflation`` -
  :func:`app.services.shrinkflation_service.get_shrinkflation_timeline`.
* ``unit_price`` - best-effort: when the product has two or more distinct pack
  variants in ``pack_size_history`` they are compared by
  :func:`app.services.unit_price_service.compare_units`; otherwise the slot is an
  unavailable state (a comparison needs multiple variants).
* ``buy_timing`` -
  :func:`app.services.buy_timing_service.recommend_buy_timing` for the product's
  category.
* ``cross_platform`` -
  :func:`app.services.cross_platform_service.aggregate_cross_platform`.

The feature services are imported as *module-level names* (rather than accessed
through their packages) so a test can monkeypatch this module's reference to one
service to prove containment, and each module is invoked through a small
callable so that patched name is resolved at call time.

The router is mounted under the ``/api/v1`` prefix by ``app.main``; it carries
the ``/dashboard`` segment itself, so the resolved path is
``/api/v1/dashboard/{product_id}``. ``{product_id}`` uses the default string
converter because product identifiers (e.g. ``amz_0001``) contain no slashes.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.models import PriceSnapshot, Product
from app.db.repositories import get_product_by_id, list_pack_size_history
from app.db.session import get_db
from app.ml.explainer import explain
from app.services.buy_timing_service import recommend_buy_timing
from app.services.cross_platform_service import aggregate_cross_platform
from app.services.discount_service import check_discount
from app.services.shrinkflation_service import get_shrinkflation_timeline
from app.services.unit_price_service import compare_units

# The router owns the ``/dashboard`` segment only; ``app.main`` includes it with
# ``prefix="/api/v1"`` so the resolved path is ``/api/v1/dashboard/{product_id}``.
router = APIRouter(prefix="/dashboard")

#: Stable machine-readable code for an unknown product (Req 15.3).
PRODUCT_NOT_FOUND_CODE = "PRODUCT_NOT_FOUND"

# Per-module unavailable messages used when a module is contained or has no data
# (Req 8.5). Kept as constants so tests and the frontend can rely on the copy.
_MSG_MODULE_FAILED = (
    "This module is temporarily unavailable; the rest of the dashboard is "
    "unaffected."
)
_MSG_NO_SNAPSHOT = (
    "No price snapshot is available to evaluate a discount for this product."
)
_MSG_UNIT_PRICE_NEEDS_VARIANTS = (
    "Unit-price comparison needs multiple pack variants."
)


def _unavailable(message: str) -> dict[str, Any]:
    """Build the uniform unavailable slot shape used across modules (Req 8.5)."""

    return {"available": False, "message": message}


def _contain(
    module_name: str,
    compute: Callable[[], dict[str, Any]],
    logger: Any,
    *,
    unavailable_message: str = _MSG_MODULE_FAILED,
) -> dict[str, Any]:
    """Run one module's ``compute`` under containment (Req 15.1, 8.5).

    Returns ``compute()`` unchanged on success. If the module raises an
    :class:`~app.core.errors.AppError` (an expected domain condition such as the
    discount not-evaluable pre-condition) the slot degrades to an unavailable
    state carrying that error's message. Any other exception is contained,
    logged with a stack trace, and turned into a generic unavailable slot so a
    single broken module never breaks the rest of the dashboard.
    """

    try:
        return compute()
    except AppError as exc:
        # An expected domain degradation (e.g. DISCOUNT_NOT_EVALUABLE): surface
        # the service's own message, and record the contained code. Note the
        # ``dashboard_module`` key: ``module`` is a reserved LogRecord attribute
        # and passing it through ``extra`` raises inside logging.
        logger.info(
            "Dashboard module '%s' is unavailable: %s",
            module_name,
            exc.code,
            extra={"dashboard_module": module_name, "code": exc.code},
        )
        return _unavailable(exc.message)
    except Exception:  # noqa: BLE001 - containment is the whole point (Req 15.1)
        logger.warning(
            "Dashboard module '%s' raised an error; contained so the remaining "
            "modules still return.",
            module_name,
            exc_info=True,
            extra={"dashboard_module": module_name},
        )
        return _unavailable(unavailable_message)


def _most_relevant_snapshot(db: Session, product_id: str) -> Optional[PriceSnapshot]:
    """Return the price snapshot best suited to a discount check, or ``None``.

    The public data is snapshot-level, so a product may carry several rows. The
    "most relevant" row for evaluating a displayed discount is one that actually
    has a *valid* reference (a reference price strictly above the displayed
    price); among equally-valid rows the most recently captured wins. When no
    row has a usable reference the most recent row is still returned so the
    discount service can report the not-evaluable pre-condition (Req 2.5), which
    the dashboard then contains into an unavailable slot.

    ``product_id`` is bound as a query parameter (Req 18.2).
    """

    rows = (
        db.execute(select(PriceSnapshot).where(PriceSnapshot.product_id == product_id))
        .scalars()
        .all()
    )
    if not rows:
        return None

    def _rank(row: PriceSnapshot) -> tuple[int, date]:
        has_valid_reference = (
            row.reference_price is not None
            and row.reference_price > row.displayed_price
        )
        # ``captured_at`` is optional; treat a missing date as the earliest so a
        # dated row is preferred over an undated one.
        captured_at = row.captured_at or date.min
        return (1 if has_valid_reference else 0, captured_at)

    return max(rows, key=_rank)


def _attach_explanation(result: dict[str, Any]) -> dict[str, Any]:
    """Attach the SHAP breakdown to a scored discount result (mirrors the endpoint).

    A scored result (non-``None`` ``genuineness_score`` carrying the engineered
    ``features``) has the SHAP explanation attached under ``explanation`` and the
    internal ``features`` key dropped, so the discount slot matches the shape of
    ``POST /api/v1/discount-check`` (Req 3.1). A limited-verification /
    scoring-unavailable result (``genuineness_score`` is ``None``) has nothing to
    explain and is returned unchanged.
    """

    features = result.get("features")
    if result.get("genuineness_score") is None or not isinstance(features, dict):
        return result

    explanation = explain(features)
    cleaned = {key: value for key, value in result.items() if key != "features"}
    cleaned["explanation"] = explanation
    return cleaned


def _pack_variants(db: Session, product_id: str) -> list[dict[str, Any]]:
    """Build the distinct pack variants for a unit-price comparison (Req 5).

    Reads the product's ``pack_size_history`` and de-duplicates it on the
    ``(pack_quantity, pack_unit, selling_price)`` triple so repeated
    observations of the same variant collapse to one comparison entry. Each
    variant is shaped for :func:`app.services.unit_price_service.compare_units`
    (``label``/``price``/``quantity``/``unit``); the label embeds the price so it
    stays unique across otherwise-identical pack sizes.
    """

    variants: list[dict[str, Any]] = []
    seen: set[tuple[float, str, float]] = set()
    for row in list_pack_size_history(db, product_id):
        key = (row.pack_quantity, row.pack_unit, row.selling_price)
        if key in seen:
            continue
        seen.add(key)
        variants.append(
            {
                "label": f"{row.pack_quantity:g}{row.pack_unit} @ {row.selling_price:g}",
                "price": row.selling_price,
                "quantity": row.pack_quantity,
                "unit": row.pack_unit,
            }
        )
    return variants


@router.get("/{product_id}")
def read_dashboard(
    product_id: str, db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Compose every feature module for a product into one dashboard (Req 8.1).

    Resolves the product, then composes the five feature modules, each contained
    independently so one failure or empty result never breaks the others
    (Req 15.1, 8.5). The assembled response always carries the same six slots and
    is serialised to JSON by FastAPI (Req 14.4)::

        {
            "product_id": str,
            "product":        {"id", "name", "brand", "category"},
            "discount":       {...scored + "explanation"...} | {"available": False, "message"},
            "shrinkflation":  {...timeline...},
            "unit_price":     {...comparison...} | {"available": False, "message"},
            "buy_timing":     {...recommendation + disclosure...},
            "cross_platform": {...platforms...},
        }

    Args:
        product_id: The selected product (path parameter; no slashes).
        db: Request-scoped SQLAlchemy session from the ``get_db`` dependency,
            closed automatically once the response is produced.

    Returns:
        The composed dashboard ``dict`` (Req 14.4).

    Raises:
        AppError: with code ``PRODUCT_NOT_FOUND`` (HTTP 404) when no product has
            the given id. Rendered by the central handler in ``app.main``.
    """

    logger = get_logger(__name__)

    product = get_product_by_id(db, product_id)
    if product is None:
        raise AppError(
            code=PRODUCT_NOT_FOUND_CODE,
            message=f"No product was found with id '{product_id}'.",
            status=404,
            details={"product_id": product_id},
        )

    # The resolved product is the one fact the dashboard cannot degrade on, so
    # it is built directly rather than contained.
    product_slot = {
        "id": product.id,
        "name": product.name,
        "brand": product.brand,
        "category": product.category,
    }

    def _compute_discount() -> dict[str, Any]:
        snapshot = _most_relevant_snapshot(db, product.id)
        if snapshot is None:
            # No snapshot -> no price pair to evaluate; degrade like a
            # not-evaluable discount rather than raising (Req 8.5).
            return _unavailable(_MSG_NO_SNAPSHOT)
        # check_discount raises AppError(DISCOUNT_NOT_EVALUABLE) for a snapshot
        # whose reference is missing / not above the displayed price; _contain
        # turns that into an unavailable slot with the service's message.
        result = check_discount(
            db,
            product.category,
            snapshot.displayed_price,
            snapshot.reference_price,
        )
        return _attach_explanation(result)

    def _compute_unit_price() -> dict[str, Any]:
        variants = _pack_variants(db, product.id)
        if len(variants) < 2:
            return _unavailable(_MSG_UNIT_PRICE_NEEDS_VARIANTS)
        return compare_units(variants)

    return {
        "product_id": product_id,
        "product": product_slot,
        "discount": _contain("discount", _compute_discount, logger),
        "shrinkflation": _contain(
            "shrinkflation",
            lambda: get_shrinkflation_timeline(db, product.id),
            logger,
        ),
        "unit_price": _contain("unit_price", _compute_unit_price, logger),
        "buy_timing": _contain(
            "buy_timing",
            lambda: recommend_buy_timing(db, product.category),
            logger,
        ),
        "cross_platform": _contain(
            "cross_platform",
            lambda: aggregate_cross_platform(db, product.id),
            logger,
        ),
    }
