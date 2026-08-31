"""Product Search service - business logic for Requirement 1.

The search entry point is how a shopper gets a product into the five feature
modules. This module owns that logic and nothing else:

* :func:`search_products` matches the ``products`` table by normalized name and
  shapes each hit into a display record carrying the product's name, brand, and
  category (Req 1.1, 1.2). An empty / whitespace / punctuation-only query is
  answered with a prompt to enter a product name (Req 1.4); a query that finds
  nothing is answered with a no-results message plus a manual-entry affordance
  flag (Req 1.5).
* :class:`SelectedProduct` is the single representation of a chosen product that
  every feature module (Discount Checker, Shrinkflation Timeline, Unit Price
  Comparator, Buy Timing Analyzer, Cross-Platform Aggregator) accepts as input
  (Req 1.3). :func:`select_product` resolves a searched product id into one.
* :func:`create_manual_entry` builds the *same* :class:`SelectedProduct` from a
  manually entered name, displayed price, reference price, and pack quantity
  (Req 1.6), routed through the same minimal validation so a manual product is
  indistinguishable to downstream modules from a searched one.

Design boundaries
------------------
The logic here is deliberately pure and framework-free so it is unit- and
property-testable in isolation and reusable behind the FastAPI endpoints added
in task 12.3. Request/response *schemas* (Pydantic) and result caching belong
to that endpoint task, not here. Database access goes exclusively through
:mod:`app.db.repositories` helpers, which use parameter-bound queries
(Req 18.2); this module never builds SQL.

Query normalization
--------------------
``products.normalized_name`` is stored lower-cased with punctuation stripped,
so the query is normalized the same way before it is handed to the repository's
``ILIKE`` match. A query that contains no searchable characters after
normalization (all whitespace or all punctuation) is treated as *empty* and
answered with the prompt message (Req 1.4) rather than being sent to the
database as an empty wildcard that would match every row.
"""

from __future__ import annotations

import math
import numbers
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.db import repositories
from app.db.models import Product

__all__ = [
    "SelectedProduct",
    "ManualEntryError",
    "search_products",
    "select_product",
    "create_manual_entry",
    "shape_search_result",
    "normalize_query",
    "STATUS_OK",
    "STATUS_EMPTY_QUERY",
    "STATUS_NO_RESULTS",
    "SOURCE_SEARCH",
    "SOURCE_MANUAL",
    "PROMPT_ENTER_QUERY_MESSAGE",
    "NO_RESULTS_MESSAGE",
    "UNKNOWN_NAME",
    "UNKNOWN_BRAND",
    "UNKNOWN_CATEGORY",
]

# --- Result vocabulary -----------------------------------------------------

#: The query matched one or more products.
STATUS_OK = "ok"
#: The query had no searchable content (Req 1.4).
STATUS_EMPTY_QUERY = "empty_query"
#: The query was valid but matched nothing (Req 1.5).
STATUS_NO_RESULTS = "no_results"

#: Origin markers on a :class:`SelectedProduct`.
SOURCE_SEARCH = "search"
SOURCE_MANUAL = "manual"

#: User-facing prompt shown for an empty query (Req 1.4).
PROMPT_ENTER_QUERY_MESSAGE = "Enter a product name to search."
#: User-facing message shown when nothing matched, alongside the manual-entry
#: affordance (Req 1.5).
NO_RESULTS_MESSAGE = (
    "No matching products found. You can enter the product's price and pack "
    "details manually."
)

# Display fallbacks so every search result carries a non-empty name, brand, and
# category (Req 1.2 / Correctness Property 1) even when the source row omits a
# value (``brand`` is nullable, and a data row could carry an empty string).
UNKNOWN_NAME = "Unknown product"
UNKNOWN_BRAND = "Unknown"
UNKNOWN_CATEGORY = "Uncategorized"

# --- Query normalization ---------------------------------------------------

# Anything that is not a word character or whitespace is treated as punctuation
# for search purposes and replaced by a space, so "amul-butter" and
# "amul butter" both reduce to the same searchable term.
_PUNCTUATION_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_query(query: Optional[str]) -> str:
    """Return the lower-cased, punctuation-stripped form of ``query``.

    Mirrors how ``products.normalized_name`` is stored so the repository's
    ``ILIKE`` substring match lines up. Returns an empty string when the query
    is ``None``, blank, or has no searchable characters after normalization.
    """

    if not query or not isinstance(query, str):
        return ""
    lowered = query.strip().lower()
    without_punctuation = _PUNCTUATION_RE.sub(" ", lowered)
    return _WHITESPACE_RE.sub(" ", without_punctuation).strip()


# --- Selected product ------------------------------------------------------


@dataclass(frozen=True)
class SelectedProduct:
    """A chosen product handed to every feature module (Req 1.3).

    Produced either by selecting a searched product (:func:`select_product`) or
    by manual entry (:func:`create_manual_entry`). The optional price and pack
    fields are populated for manual entries (Req 1.6) and left ``None`` for a
    searched selection, where the shopper supplies prices later at the point of
    a discount check. The dataclass is frozen so a selected product cannot be
    mutated as it flows between feature modules.
    """

    id: str
    name: str
    category: Optional[str] = None
    brand: Optional[str] = None
    source: str = SOURCE_SEARCH
    displayed_price: Optional[float] = None
    reference_price: Optional[float] = None
    pack_quantity: Optional[float] = None
    pack_unit: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view for JSON serialization at the endpoint."""

        return asdict(self)


class ManualEntryError(ValueError):
    """Raised when manually entered product details fail minimal validation.

    Carries the offending ``field`` so the endpoint layer (task 12.3) can map
    it into the structured error payload without re-parsing the message.
    """

    def __init__(self, message: str, *, field: str) -> None:
        super().__init__(message)
        self.field = field


# --- Search ----------------------------------------------------------------


def _coerce_non_empty(value: Any, fallback: str) -> str:
    """Return ``value`` as a trimmed non-empty string, or ``fallback``."""

    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def shape_search_result(product: Product) -> dict[str, Any]:
    """Shape a product row into a search-result record (Req 1.2).

    Guarantees a non-empty name, brand, and category (Correctness Property 1)
    by applying a display fallback when the source row omits a value. ``id`` is
    included so the caller can select the product afterwards (Req 1.3).
    """

    return {
        "id": product.id,
        "name": _coerce_non_empty(product.name, UNKNOWN_NAME),
        "brand": _coerce_non_empty(product.brand, UNKNOWN_BRAND),
        "category": _coerce_non_empty(product.category, UNKNOWN_CATEGORY),
    }


def search_products(db: Session, query: str, *, limit: int = 20) -> dict[str, Any]:
    """Search products by name and shape the outcome for the caller (Req 1).

    Returns a structured dict whose ``status`` is one of :data:`STATUS_OK`,
    :data:`STATUS_EMPTY_QUERY`, or :data:`STATUS_NO_RESULTS`:

    * empty / blank / punctuation-only query -> prompt message, no manual-entry
      affordance (Req 1.4);
    * a query that matches nothing -> no-results message and
      ``manual_entry=True`` (Req 1.5);
    * otherwise -> a ``results`` list where each entry carries the product's
      id plus its name, brand, and category (Req 1.1, 1.2).

    The shape is deliberately uniform (the same keys always present) so the
    endpoint layer can serialize it directly.
    """

    normalized = normalize_query(query)
    if not normalized:
        return {
            "status": STATUS_EMPTY_QUERY,
            "query": "",
            "results": [],
            "message": PROMPT_ENTER_QUERY_MESSAGE,
            "manual_entry": False,
        }

    products = repositories.search_products_by_name(db, normalized, limit=limit)
    if not products:
        return {
            "status": STATUS_NO_RESULTS,
            "query": normalized,
            "results": [],
            "message": NO_RESULTS_MESSAGE,
            "manual_entry": True,
        }

    return {
        "status": STATUS_OK,
        "query": normalized,
        "results": [shape_search_result(product) for product in products],
        "message": None,
        "manual_entry": False,
    }


def select_product(db: Session, product_id: str) -> Optional[SelectedProduct]:
    """Resolve a chosen product id into a :class:`SelectedProduct` (Req 1.3).

    Returns ``None`` when no product has that id, letting the endpoint layer
    surface a not-found response. Uses the parameter-bound repository helper
    ``get_product_by_id`` (Req 18.2); the returned object is the same type a
    manual entry yields, so every feature module consumes it identically.
    """

    if not product_id or not isinstance(product_id, str):
        return None
    product = repositories.get_product_by_id(db, product_id)
    if product is None:
        return None
    return SelectedProduct(
        id=product.id,
        name=_coerce_non_empty(product.name, UNKNOWN_NAME),
        category=product.category,
        brand=product.brand,
        source=SOURCE_SEARCH,
    )


# --- Manual entry ----------------------------------------------------------


def _validate_optional_price(value: Any, field: str) -> Optional[float]:
    """Validate an optional strictly-positive, finite number (or ``None``).

    Returns ``None`` when ``value`` is ``None`` (the field was not supplied).
    ``bool`` is rejected explicitly because it is a subclass of ``int`` and is
    never a meaningful price or quantity. Raises :class:`ManualEntryError` on
    any other invalid value.
    """

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ManualEntryError(f"{field} must be a number.", field=field)
    number = float(value)
    if not math.isfinite(number):
        raise ManualEntryError(f"{field} must be a finite number.", field=field)
    if number <= 0:
        raise ManualEntryError(f"{field} must be greater than zero.", field=field)
    return number


def _validate_required_price(value: Any, field: str) -> float:
    """Validate a required, strictly-positive, finite number."""

    number = _validate_optional_price(value, field)
    if number is None:
        raise ManualEntryError(f"{field} is required.", field=field)
    return number


def create_manual_entry(
    *,
    name: str,
    displayed_price: Any,
    reference_price: Any = None,
    pack_quantity: Any = None,
    pack_unit: Optional[str] = None,
    category: Optional[str] = None,
    brand: Optional[str] = None,
) -> SelectedProduct:
    """Build a :class:`SelectedProduct` from manual-entry inputs (Req 1.6).

    Accepts the product name, displayed price, reference price, and pack
    quantity a shopper types when no searched product matched (Req 1.5, 1.6).
    Validation is intentionally minimal and pure: the name must be non-empty,
    the displayed price is required, and every supplied price/quantity must be a
    positive, finite number. The reference-vs-displayed relationship is *not*
    enforced here - that is the discount checker's Req 2.5 pre-condition,
    checked downstream. ``category``, ``brand``, and ``pack_unit`` are optional
    context (not mandated by Req 1.6) and are normalized to ``None`` when blank.

    The result is the same :class:`SelectedProduct` a searched selection yields
    (with ``source == "manual"``), so every feature module consumes it
    identically.
    """

    if not isinstance(name, str) or not name.strip():
        raise ManualEntryError("A product name is required.", field="name")

    clean_name = name.strip()
    validated_displayed = _validate_required_price(displayed_price, "displayed_price")
    validated_reference = _validate_optional_price(reference_price, "reference_price")
    validated_quantity = _validate_optional_price(pack_quantity, "pack_quantity")

    def _clean_optional_text(value: Optional[str]) -> Optional[str]:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    return SelectedProduct(
        id=f"manual:{uuid.uuid4().hex}",
        name=clean_name,
        category=_clean_optional_text(category),
        brand=_clean_optional_text(brand),
        source=SOURCE_MANUAL,
        displayed_price=validated_displayed,
        reference_price=validated_reference,
        pack_quantity=validated_quantity,
        pack_unit=_clean_optional_text(pack_unit),
    )
