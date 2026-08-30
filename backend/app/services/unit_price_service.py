"""Unit Price Comparator service - pure business logic for Requirement 5.

This module computes and compares the *unit price* (price per standard unit of
quantity) across two or more product variants. It is intentionally a **pure
function** with no I/O and no framework dependencies so that:

* the property-based tests (task 6.2, Correctness Properties 12-15) can call it
  directly with generated inputs, and
* the FastAPI endpoint (task 6.3) can wrap it behind request/response schemas.

Standardisation (Requirement 5.4)
---------------------------------
Every pack quantity is converted to a common standard unit before the unit
price is computed:

* mass   -> grams (g):        ``kg`` is multiplied by 1000
* volume -> millilitres (ml): ``l``  is multiplied by 1000

The standard unit for a comparison is therefore ``"g"`` for mass and ``"ml"``
for volume, derived from the measure family of the variants.

Unit price (Requirement 5.1) is ``price / quantity_in_standard_unit`` and the
variant with the lowest unit price is flagged as the best value
(Requirement 5.2). Each included variant reports its price, standardised
quantity, and unit price in a single comparison structure (Requirement 5.3).

Invalid variants (Requirement 5.5)
----------------------------------
A variant whose pack quantity is missing, non-numeric, non-finite, or
non-positive cannot yield a meaningful unit price. It is removed from the
comparison and reported in the ``excluded`` list with a machine-readable
reason. Excluded variants never appear in ``comparison``.

Preconditions
-------------
Each variant is expected to provide a positive, finite ``price``; this is
enforced upstream at the API boundary (``VariantIn``, Req 18.1). Quantity is
validated here because Requirement 5.5 mandates graceful exclusion of
missing/non-positive quantities.

Mixed measure families
----------------------
The normal contract assumes every variant belongs to the *same* measure family
(all mass or all volume), and the top-level ``standard_unit`` describes that
family. If the input mixes families, the function still does not raise: each
variant is standardised within its own family (mass -> g, volume -> ml), every
valid variant is kept in the comparison exactly once (so Property 15 holds),
and ``standard_unit`` reports the predominant family. Comparing a per-gram
price against a per-millilitre price is not physically meaningful, so callers
are expected to pass a single family; the mixed case is a graceful fallback,
not a supported comparison.
"""

from __future__ import annotations

import math
import numbers
from collections import Counter
from typing import Any, Iterable, Mapping

__all__ = [
    "compare_units",
    "REASON_MISSING_QUANTITY",
    "REASON_NON_POSITIVE_QUANTITY",
    "REASON_INVALID_QUANTITY",
    "REASON_UNSUPPORTED_UNIT",
]

# Conversion factor from each supported unit to its family's standard base unit.
# Mass standardises to grams; volume standardises to millilitres. Integer
# factors are used so an integer input quantity keeps an integer standardised
# quantity (e.g. 1 kg -> 1000 g), matching the design's API example.
_CONVERSION_FACTOR: dict[str, int] = {"g": 1, "kg": 1000, "ml": 1, "l": 1000}

# Measure family for each supported unit.
_UNIT_FAMILY: dict[str, str] = {"g": "mass", "kg": "mass", "ml": "volume", "l": "volume"}

# Standard base unit reported for each measure family.
_FAMILY_STANDARD_UNIT: dict[str, str] = {"mass": "g", "volume": "ml"}

# Fallback when no variant carries a recognisable unit (keeps the response
# shape's ``standard_unit`` a valid string rather than ``None``).
_DEFAULT_STANDARD_UNIT = "g"

# Machine-readable exclusion reasons (Requirement 5.5).
REASON_MISSING_QUANTITY = "missing_quantity"
REASON_NON_POSITIVE_QUANTITY = "non_positive_quantity"
REASON_INVALID_QUANTITY = "invalid_quantity"
REASON_UNSUPPORTED_UNIT = "unsupported_unit"


def _normalize_unit(unit: Any) -> str | None:
    """Return the canonical lower-case unit, or ``None`` if unsupported."""
    if not isinstance(unit, str):
        return None
    normalized = unit.strip().lower()
    return normalized if normalized in _CONVERSION_FACTOR else None


def _quantity_reason(quantity: Any) -> str | None:
    """Return an exclusion reason for an invalid quantity, or ``None`` if valid.

    A valid pack quantity must be present, a real (non-boolean) number, finite,
    and strictly positive.
    """
    if quantity is None:
        return REASON_MISSING_QUANTITY
    # ``bool`` is a subclass of ``int``; a boolean is not a real quantity.
    if isinstance(quantity, bool) or not isinstance(quantity, numbers.Real):
        return REASON_INVALID_QUANTITY
    if not math.isfinite(float(quantity)):
        return REASON_INVALID_QUANTITY
    if quantity <= 0:
        return REASON_NON_POSITIVE_QUANTITY
    return None


def _resolve_standard_unit(families: list[str]) -> str:
    """Pick the standard unit from the predominant measure family.

    Ties are broken by first appearance so the result is deterministic.
    """
    if not families:
        return _DEFAULT_STANDARD_UNIT
    counts = Counter(families)
    top_count = max(counts.values())
    for family in families:  # first-seen order among the tied modes
        if counts[family] == top_count:
            return _FAMILY_STANDARD_UNIT[family]
    return _DEFAULT_STANDARD_UNIT


def compare_units(variants: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare the per-standard-unit price of product variants.

    Args:
        variants: An iterable of mappings, each with ``label`` (str),
            ``price`` (positive float), ``quantity`` (float), and ``unit`` in
            ``{"g", "kg", "ml", "l"}``.

    Returns:
        A dict of the shape::

            {
                "standard_unit": "g" | "ml",
                "comparison": [
                    {"label", "price", "quantity_std", "unit_price"[, "best_value"]},
                    ...
                ],
                "excluded": [{"label", "reason"}, ...],
            }

        ``comparison`` preserves input order and contains every variant with a
        valid quantity exactly once; the single lowest-unit-price entry carries
        ``"best_value": True``. ``excluded`` preserves input order and lists
        every variant that could not be compared, each with a machine-readable
        ``reason``.
    """
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    families: list[str] = []

    for index, variant in enumerate(variants):
        label = variant.get("label", f"variant_{index}")
        price = variant.get("price")
        raw_quantity = variant.get("quantity")
        unit = _normalize_unit(variant.get("unit"))

        # The measure family is defined by the unit alone, so record it even for
        # variants that are later excluded on quantity - it still informs which
        # standard unit best describes the overall comparison.
        if unit is not None:
            families.append(_UNIT_FAMILY[unit])
        else:
            excluded.append({"label": label, "reason": REASON_UNSUPPORTED_UNIT})
            continue

        reason = _quantity_reason(raw_quantity)
        if reason is not None:
            excluded.append({"label": label, "reason": reason})
            continue

        # Standardise within the variant's own family (Req 5.4) and compute the
        # unit price (Req 5.1). No rounding: the stored value is exactly
        # price / quantity_std so the identity and best-value properties hold.
        quantity_std = raw_quantity * _CONVERSION_FACTOR[unit]
        unit_price = price / quantity_std
        included.append(
            {
                "label": label,
                "price": price,
                "quantity_std": quantity_std,
                "unit_price": unit_price,
            }
        )

    standard_unit = _resolve_standard_unit(families)

    # Mark exactly one best value: the first variant achieving the minimum unit
    # price (Req 5.2). ``min`` returns the first minimal element on ties.
    if included:
        best = min(included, key=lambda item: item["unit_price"])
        best["best_value"] = True

    return {
        "standard_unit": standard_unit,
        "comparison": included,
        "excluded": excluded,
    }
