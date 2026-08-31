"""Unit tests for the Buy Timing Signal (Task 11.1).

These example-based tests cover the seasonality module
(:mod:`app.ml.seasonality`) and the buy-timing service
(:mod:`app.services.buy_timing_service`). They seed an in-memory SQLite
database with a category's ``category_seasonality`` rows (built through the
pure profile builder so both modules are exercised together) and assert the
Requirement 6 branches:

* a mid-year evaluation returns ``wait`` pointing at the October Big Billion
  Days window, with the category-level/snapshot disclosure (Req 6.1, 6.2, 6.4,
  6.5, 10.1);
* evaluating in the best window returns ``buy_now`` (Req 6.1);
* a category with no rows returns the unavailable message (Req 6.6);
* the Indian sale calendar contains the four named events (Req 6.5).

The property-based tests for Correctness Properties 16 and 17 are owned by the
separate task 11.2; these tests stay focused on concrete examples and edges.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, CategorySeasonality
from app.ml.seasonality import (
    INDIAN_SALE_CALENDAR,
    NAMED_SALE_EVENTS,
    build_category_profile,
    find_best_window,
    sale_event_for_month,
)
from app.services.buy_timing_service import (
    DISCLOSURE,
    MESSAGE_UNAVAILABLE,
    recommend_buy_timing,
)

TV_CATEGORY = "electronics/tv"


@pytest.fixture()
def db():
    """Provide an isolated in-memory SQLite session with the schema created."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session: Session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_profile(db: Session, category: str, monthly_signal=None) -> None:
    """Persist a built category profile into ``category_seasonality``."""
    for point in build_category_profile(monthly_signal):
        db.add(CategorySeasonality(category=category, **point))
    db.commit()


# ---------------------------------------------------------------------------
# Seasonality module: sale calendar + profile builder
# ---------------------------------------------------------------------------


def test_sale_calendar_contains_the_four_named_events():
    """Req 6.5: the calendar names Big Billion Days, Diwali, Republic Day, Prime Day."""
    assert NAMED_SALE_EVENTS == {
        "Big Billion Days",
        "Diwali",
        "Republic Day Sale",
        "Prime Day",
    }
    # Anchored to the expected months.
    assert INDIAN_SALE_CALENDAR[1] == "Republic Day Sale"
    assert INDIAN_SALE_CALENDAR[7] == "Prime Day"
    assert INDIAN_SALE_CALENDAR[10] == "Big Billion Days"
    assert INDIAN_SALE_CALENDAR[11] == "Diwali"
    assert sale_event_for_month(10) == "Big Billion Days"
    assert sale_event_for_month(3) is None


def test_build_profile_defaults_to_calendar_prior_with_october_best_window():
    """No temporal signal -> calendar prior; Big Billion Days is the best window."""
    profile = build_category_profile()

    assert len(profile) == 12
    assert [p["month"] for p in profile] == list(range(1, 13))

    october = next(p for p in profile if p["month"] == 10)
    assert october["is_best_window"] is True
    assert october["sale_event"] == "Big Billion Days"
    # Exactly one deepest window under the default prior.
    assert sum(1 for p in profile if p["is_best_window"]) == 1

    best = find_best_window(profile)
    assert best["month"] == 10
    assert all(
        p["relative_price_index"] >= best["relative_price_index"] for p in profile
    )


def test_build_profile_uses_monthly_signal_when_available():
    """A usable monthly signal overrides the prior for the best window."""
    # July is by far the cheapest month in the observed signal.
    signal = {m: 1000.0 for m in range(1, 13)}
    signal[7] = 500.0
    profile = build_category_profile(signal)

    best = find_best_window(profile)
    assert best["month"] == 7
    july = next(p for p in profile if p["month"] == 7)
    assert july["is_best_window"] is True
    assert july["relative_price_index"] < 1.0


# ---------------------------------------------------------------------------
# Buy-timing service
# ---------------------------------------------------------------------------


def test_midyear_returns_wait_with_october_window_and_disclosure(db):
    """Req 6.1/6.2/6.4/6.5/10.1: mid-year -> wait for the October sale window."""
    _seed_profile(db, TV_CATEGORY)

    result = recommend_buy_timing(db, TV_CATEGORY, current_month=6)

    assert result["available"] is True
    assert result["recommendation"] == "wait"
    assert result["level"] == "category"
    assert result["current_month"] == 6

    # Req 6.2 + 6.5: the deepest-discount window is October / Big Billion Days.
    window = result["best_window"]
    assert window["month"] == 10
    assert window["month_name"] == "October"
    assert window["sale_event"] == "Big Billion Days"
    assert window["expected_reduction_pct"] > 0

    # Req 6.4 / 10.1: category-level + snapshot-data disclosure always present.
    assert result["disclosure"] == DISCLOSURE
    assert "category-level" in result["disclosure"]
    assert "snapshot" in result["disclosure"]

    # Req 6.5: the message references the sale calendar event on the window.
    assert "October" in result["message"]
    assert "Big Billion Days" in result["message"]


def test_current_month_is_best_window_returns_buy_now(db):
    """Req 6.1: evaluating in the best window recommends buying now."""
    _seed_profile(db, TV_CATEGORY)

    result = recommend_buy_timing(db, TV_CATEGORY, current_month=10)

    assert result["available"] is True
    assert result["recommendation"] == "buy_now"
    assert result["best_window"]["month"] == 10
    # Disclosure is attached to buy_now results too (Req 6.4 / 10.1).
    assert result["disclosure"] == DISCLOSURE


def test_best_window_already_passed_returns_buy_now(db):
    """Req 6.1: once the year's deepest window is behind, recommend buying now."""
    _seed_profile(db, TV_CATEGORY)

    # December: the October best window has passed and nothing cheaper is ahead.
    result = recommend_buy_timing(db, TV_CATEGORY, current_month=12)

    assert result["recommendation"] == "buy_now"
    assert result["best_window"]["month"] == 10


def test_unknown_category_returns_unavailable(db):
    """Req 6.6: a category with no seasonal rows is reported unavailable."""
    result = recommend_buy_timing(db, "toys/board-games", current_month=6)

    assert result["available"] is False
    assert result["recommendation"] is None
    assert result["best_window"] is None
    assert result["message"] == MESSAGE_UNAVAILABLE
    # The category-level/snapshot disclosure still holds even when unavailable.
    assert result["disclosure"] == DISCLOSURE


def test_current_month_defaults_to_today(db):
    """Omitting current_month evaluates against today's month without error."""
    _seed_profile(db, TV_CATEGORY)

    result = recommend_buy_timing(db, TV_CATEGORY)

    assert result["available"] is True
    assert 1 <= result["current_month"] <= 12
    assert result["recommendation"] in {"buy_now", "wait"}


def test_invalid_current_month_raises(db):
    """A month outside 1-12 is rejected."""
    _seed_profile(db, TV_CATEGORY)

    with pytest.raises(ValueError):
        recommend_buy_timing(db, TV_CATEGORY, current_month=13)
