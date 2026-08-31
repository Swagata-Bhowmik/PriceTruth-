"""Unit tests for the offline ingestion cleaning + statistics helpers (Task 3.6).

Covers the pure, side-effect-free helpers in ``data/scripts/ingest.py`` that the
ingestion pipeline (tasks 3.2 / 3.3) relies on to turn messy Kaggle cells into
the cleaned rows and per-category distribution statistics the discount model
consumes:

* :func:`ingest.to_number` - currency/percent/thousands-separator stripping and
  coercion to ``float`` (Req 9.5).
* :func:`ingest.clean_discount_pct` - clamping a discount into ``[0, 100]``
  (Req 9.5).
* :func:`ingest.normalize_category` - mapping the Amazon pipe hierarchy and the
  Flipkart bracketed ``category_tree`` onto a canonical slug (Req 9.5).
* :func:`ingest.normalize_name` - lower-casing + whitespace collapse (Req 9.5).
* :func:`ingest._compute_category_stats` - reducing in-memory snapshots into the
  ``mean/median/std/p25/p75`` price distribution, discount / rating norms, and
  ``sample_size`` per category, including the ``_finite`` guard that turns the
  undefined single-row sample std (NaN) into ``0.0`` (Req 2.3).

``ingest.py`` lives at ``<repo_root>/data/scripts`` rather than under
``backend/app``, so that directory is placed on ``sys.path`` before importing it
(``ingest`` itself puts ``backend/`` on ``sys.path`` for ``app.db.models``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ``ingest.py`` is not an installed package - it is the standalone offline script
# at ``<repo_root>/data/scripts``. Compute that directory relative to this test
# file (tests -> backend -> repo_root) and expose it for ``import ingest``.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "data" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import ingest  # noqa: E402  (import after sys.path wiring)
from ingest import (  # noqa: E402
    _compute_category_stats,
    clean_discount_pct,
    normalize_category,
    normalize_name,
    to_number,
)

# ``ingest`` re-exports the ORM models it imports from ``app.db.models`` (and, as
# a side effect of importing it, has already put ``backend/`` on ``sys.path``).
Product = ingest.Product
PriceSnapshot = ingest.PriceSnapshot


# ---------------------------------------------------------------------------
# to_number: currency / percent / thousands-separator stripping (Req 9.5)
# ---------------------------------------------------------------------------
class TestToNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("\u20b91,299", 1299.0),   # rupee sign + thousands separator
            ("\u20b91,299.50", 1299.5),  # decimal preserved
            ("Rs. 499", 499.0),        # currency prefix
            ("75%", 75.0),             # trailing percent
            ("32,210", 32210.0),       # bare thousands separator
            ("1299", 1299.0),          # plain integer string
            (1299, 1299.0),            # numeric passthrough (int)
            (1299.5, 1299.5),          # numeric passthrough (float)
        ],
    )
    def test_parses_decorated_numbers(self, raw, expected):
        assert to_number(raw) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "raw",
        [
            "",        # blank
            "   ",     # whitespace only
            "N/A",     # missing token
            "nan",     # missing token
            "abc",     # no digits
            "--",      # sign only, no digits
            None,      # explicit None
        ],
    )
    def test_returns_none_for_blank_or_garbage(self, raw):
        assert to_number(raw) is None


# ---------------------------------------------------------------------------
# clean_discount_pct: clamp into [0, 100] (Req 9.5)
# ---------------------------------------------------------------------------
class TestCleanDiscountPct:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (150, 100.0),      # above range -> clamp high
            (150.0, 100.0),
            ("150%", 100.0),   # percent stripped, then clamped
            (-5, 0.0),         # below range -> clamp low
            ("-5", 0.0),
            (50, 50.0),        # in range -> unchanged
            ("50%", 50.0),
            (0, 0.0),          # boundary
            (100, 100.0),      # boundary
        ],
    )
    def test_clamps_to_closed_range(self, raw, expected):
        assert clean_discount_pct(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "n/a"])
    def test_missing_stays_none(self, raw):
        assert clean_discount_pct(raw) is None


# ---------------------------------------------------------------------------
# normalize_category: Amazon pipe hierarchy + Flipkart bracketed tree (Req 9.5)
# ---------------------------------------------------------------------------
class TestNormalizeCategory:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # Amazon pipe hierarchy -> leaf keyword match.
            ("Electronics|Audio|Headphones", "electronics/headphones"),
            ("Electronics|Television|Smart TV", "electronics/tv"),
            ("Grocery|Snacks|Biscuits", "grocery/biscuits"),
            # Flipkart bracketed JSON-ish tree -> same leaf resolution.
            ('["Electronics >> Audio >> Headphones"]', "electronics/headphones"),
            ('["Grocery >> Cooking Essentials >> Edible Oil"]', "grocery/edible-oil"),
        ],
    )
    def test_maps_both_hierarchies_to_canonical_slug(self, raw, expected):
        assert normalize_category(raw) == expected

    @pytest.mark.parametrize("raw", ["Fashion|Men|Shirts", None])
    def test_unmatched_or_missing_returns_none(self, raw):
        assert normalize_category(raw) is None


# ---------------------------------------------------------------------------
# normalize_name: lower-case + whitespace collapse (Req 9.5)
# ---------------------------------------------------------------------------
class TestNormalizeName:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Sony   WH-1000XM4  Headphones ", "sony wh-1000xm4 headphones"),
            ("AMUL\tButter\n500g", "amul butter 500g"),
            ("Already normal", "already normal"),
        ],
    )
    def test_lowercases_and_collapses_whitespace(self, raw, expected):
        assert normalize_name(raw) == expected


# ---------------------------------------------------------------------------
# _compute_category_stats: per-category distribution reduction (Req 2.3)
# ---------------------------------------------------------------------------
def _snapshot(product_id, displayed, discount=None, rating=None, rating_count=None):
    """Build a transient PriceSnapshot carrying only the fields the reducer reads."""
    return PriceSnapshot(
        product_id=product_id,
        platform="Amazon",
        reference_price=None,
        displayed_price=displayed,
        discount_pct=discount,
        rating=rating,
        rating_count=rating_count,
    )


class TestComputeCategoryStats:
    def test_multi_row_category_matches_hand_computed_statistics(self):
        """A three-row category yields the hand-computed mean/median/std/p25/p75.

        Displayed prices ``[100, 200, 300]`` -> mean 200, median 200,
        sample std (ddof=1) 100, p25 150, p75 250 (pandas' default linear
        interpolation). Discounts ``[10, 20, 30]`` -> mean 20, std 10; ratings
        ``[4, 5, 3]`` -> mean 4; rating counts ``[100, 200, 300]`` -> mean 200.
        """
        products = {"hp": Product(id="hp", name="Headphone", category="electronics/headphones")}
        snapshots = [
            _snapshot("hp", 100.0, discount=10.0, rating=4.0, rating_count=100),
            _snapshot("hp", 200.0, discount=20.0, rating=5.0, rating_count=200),
            _snapshot("hp", 300.0, discount=30.0, rating=3.0, rating_count=300),
        ]

        stats = {row.category: row for row in _compute_category_stats(products, snapshots)}
        assert set(stats) == {"electronics/headphones"}

        hp = stats["electronics/headphones"]
        assert hp.sample_size == 3
        assert hp.mean_price == pytest.approx(200.0)
        assert hp.median_price == pytest.approx(200.0)
        assert hp.std_price == pytest.approx(100.0)
        assert hp.p25_price == pytest.approx(150.0)
        assert hp.p75_price == pytest.approx(250.0)
        assert hp.mean_discount_pct == pytest.approx(20.0)
        assert hp.std_discount_pct == pytest.approx(10.0)
        assert hp.mean_rating == pytest.approx(4.0)
        assert hp.mean_rating_count == pytest.approx(200.0)

    def test_single_row_category_yields_zero_std_not_nan(self):
        """A one-row category has an undefined sample std; ``_finite`` -> 0.0.

        Also confirms empty discount/rating columns (all missing) collapse to
        ``0.0`` rather than NaN, since these are non-nullable float columns.
        """
        products = {"bis": Product(id="bis", name="Biscuit", category="grocery/biscuits")}
        snapshots = [_snapshot("bis", 50.0)]  # no discount / rating / rating_count

        (row,) = _compute_category_stats(products, snapshots)

        assert row.category == "grocery/biscuits"
        assert row.sample_size == 1
        assert row.mean_price == pytest.approx(50.0)
        assert row.median_price == pytest.approx(50.0)
        # The single-row sample std is NaN; the _finite guard normalises it.
        assert row.std_price == 0.0
        # Empty (all-missing) columns also normalise to a finite 0.0.
        assert row.std_discount_pct == 0.0
        assert row.mean_discount_pct == 0.0
        assert row.mean_rating == 0.0
        assert row.mean_rating_count == 0.0

    def test_groups_by_category_and_skips_snapshots_without_a_category(self):
        """Snapshots are grouped by their product's slug; uncategorised are dropped."""
        products = {
            "hp": Product(id="hp", name="Headphone", category="electronics/headphones"),
            "bis": Product(id="bis", name="Biscuit", category="grocery/biscuits"),
            "unk": Product(id="unk", name="Mystery", category=None),
        }
        snapshots = [
            _snapshot("hp", 100.0),
            _snapshot("hp", 300.0),
            _snapshot("bis", 50.0),
            _snapshot("unk", 999.0),   # no category -> excluded
        ]

        stats = {row.category: row for row in _compute_category_stats(products, snapshots)}

        assert set(stats) == {"electronics/headphones", "grocery/biscuits"}
        assert stats["electronics/headphones"].sample_size == 2
        assert stats["electronics/headphones"].mean_price == pytest.approx(200.0)
        assert stats["grocery/biscuits"].sample_size == 1

    def test_empty_input_returns_no_rows(self):
        assert _compute_category_stats({}, []) == []
