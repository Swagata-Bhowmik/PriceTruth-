"""Verification for the True Discount Checker service (Task 8.1, Req 2.1-2.6, 15.1).

Example-based unit tests that seed an in-memory SQLite database from
``app.db.models.Base.metadata`` and drive
``app.services.discount_service.check_discount`` through the real repository
helper. They confirm:

* the not-evaluable pre-condition raises ``DISCOUNT_NOT_EVALUABLE`` (422) when
  the reference price is missing or is not strictly greater than the displayed
  price, and never returns a score (Req 2.5);
* a category with no statistics returns a ``verification_limited`` result with
  price context and no score (Req 2.6);
* a missing trained model degrades to a ``scoring_unavailable`` result rather
  than raising (Req 15.1);
* the score-to-band mapping is correct at the 59/60/89/90 boundaries, driven by
  monkeypatched controlled probabilities (Req 2.1, 2.2);
* the effective discount percentage equals ``(reference - displayed) / reference
  * 100`` and the result echoes both prices (Req 2.4);
* the normal path against the *real* trained model returns an integer score in
  [0, 100] with a valid band and the engineered features (Req 2.1, 2.3).

The numbered property-based tests (Correctness Properties 2-5) are a separate
task (8.2); these are concrete examples that anchor the behaviour.
"""

import math

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.errors import AppError
from app.db.models import Base, CategoryPriceStats
from app.ml import discount_model
from app.ml.discount_model import FEATURE_NAMES
from app.services import discount_service
from app.services.discount_service import (
    CLASSIFICATION_GENUINE,
    CLASSIFICATION_LIKELY_INFLATED,
    CLASSIFICATION_MODERATE,
    CLASSIFICATION_SCORING_UNAVAILABLE,
    CLASSIFICATION_VERIFICATION_LIMITED,
    NOT_EVALUABLE_CODE,
    check_discount,
)

_CATEGORY = "electronics/headphones"


def _make_session() -> tuple[Session, "object"]:
    """Create a fresh in-memory SQLite session with all tables provisioned.

    ``StaticPool`` keeps a single underlying connection so the ``:memory:``
    database survives across queries within one test.
    """

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return factory(), engine


@pytest.fixture()
def db_session():
    session, engine = _make_session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _seed_stats(session: Session, category: str = _CATEGORY) -> None:
    """Seed one plausible category price-distribution row for ``category``."""

    session.add(
        CategoryPriceStats(
            category=category,
            mean_price=2500.0,
            median_price=2200.0,
            std_price=800.0,
            p25_price=1800.0,
            p75_price=3000.0,
            mean_discount_pct=45.0,
            std_discount_pct=15.0,
            mean_rating=4.0,
            mean_rating_count=1200.0,
            sample_size=500,
        )
    )
    session.commit()


# ---------------------------------------------------------------------------
# Req 2.5 - not evaluable without a valid reference price
# ---------------------------------------------------------------------------
def test_missing_reference_raises_not_evaluable(db_session):
    """Req 2.5: a missing reference price cannot be evaluated; no score returned."""
    _seed_stats(db_session)

    with pytest.raises(AppError) as exc_info:
        check_discount(db_session, _CATEGORY, displayed_price=1499.0, reference_price=None)

    error = exc_info.value
    assert error.code == NOT_EVALUABLE_CODE
    assert error.status == 422
    assert error.details.get("reason")  # a stated reason is present


@pytest.mark.parametrize("reference_price", [1499.0, 999.0, 0.0])
def test_reference_not_greater_than_displayed_raises(db_session, reference_price):
    """Req 2.5: reference <= displayed cannot be evaluated (equal and below)."""
    _seed_stats(db_session)

    with pytest.raises(AppError) as exc_info:
        check_discount(
            db_session,
            _CATEGORY,
            displayed_price=1499.0,
            reference_price=reference_price,
        )

    error = exc_info.value
    assert error.code == NOT_EVALUABLE_CODE
    assert error.status == 422
    assert error.details.get("reason")


def test_not_evaluable_takes_precedence_over_missing_stats(db_session):
    """Req 2.5 is checked before data access: no stats row, bad reference -> raise."""
    # No stats seeded for this category; the pre-condition must still fire first.
    with pytest.raises(AppError) as exc_info:
        check_discount(db_session, _CATEGORY, displayed_price=100.0, reference_price=None)

    assert exc_info.value.code == NOT_EVALUABLE_CODE


# ---------------------------------------------------------------------------
# Req 2.6 - limited verification when category stats are unavailable
# ---------------------------------------------------------------------------
def test_missing_stats_returns_verification_limited(db_session):
    """Req 2.6: no category stats -> limited verification, price context, no score."""
    # Intentionally do not seed any CategoryPriceStats row.
    result = check_discount(
        db_session,
        "unknown/category",
        displayed_price=1499.0,
        reference_price=4999.0,
    )

    assert result["classification"] == CLASSIFICATION_VERIFICATION_LIMITED
    assert result["genuineness_score"] is None
    assert result["message"]
    assert result["displayed_price"] == 1499.0
    assert result["reference_price"] == 4999.0
    assert result["price_context"] == {
        "displayed_price": 1499.0,
        "reference_price": 4999.0,
    }
    # Effective discount is still reported (the discount is evaluable, Req 2.4).
    assert math.isclose(
        result["effective_discount_pct"],
        (4999.0 - 1499.0) / 4999.0 * 100.0,
        rel_tol=1e-9,
    )


# ---------------------------------------------------------------------------
# Req 15.1 - resilience: degrade rather than crash when no model is loaded
# ---------------------------------------------------------------------------
def test_missing_model_returns_scoring_unavailable(db_session, monkeypatch):
    """Req 15.1: with no trained model, return scoring_unavailable, not a 500."""
    _seed_stats(db_session)
    monkeypatch.setattr(discount_model, "get_model", lambda: None)

    result = check_discount(
        db_session,
        _CATEGORY,
        displayed_price=1499.0,
        reference_price=4999.0,
    )

    assert result["classification"] == CLASSIFICATION_SCORING_UNAVAILABLE
    assert result["genuineness_score"] is None
    assert result["message"]
    assert result["price_context"]["displayed_price"] == 1499.0


# ---------------------------------------------------------------------------
# Req 2.1 / 2.2 - score range and band boundaries
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("probability", "expected_score", "expected_band"),
    [
        (0.00, 0, CLASSIFICATION_LIKELY_INFLATED),
        (0.59, 59, CLASSIFICATION_LIKELY_INFLATED),  # just below moderate
        (0.60, 60, CLASSIFICATION_MODERATE),  # lower moderate boundary
        (0.89, 89, CLASSIFICATION_MODERATE),  # just below genuine
        (0.90, 90, CLASSIFICATION_GENUINE),  # lower genuine boundary
        (1.00, 100, CLASSIFICATION_GENUINE),
    ],
)
def test_band_boundaries(
    db_session, monkeypatch, probability, expected_score, expected_band
):
    """Req 2.1/2.2: round(p*100) is the score and maps to the correct band."""
    _seed_stats(db_session)

    def fake_predict(features, model):
        return probability

    monkeypatch.setattr(discount_model, "predict_genuineness", fake_predict)

    # A non-None sentinel model bypasses get_model(); the fake ignores it.
    result = check_discount(
        db_session,
        _CATEGORY,
        displayed_price=1000.0,
        reference_price=5000.0,
        model=object(),
    )

    assert result["genuineness_score"] == expected_score
    assert isinstance(result["genuineness_score"], int)
    assert result["classification"] == expected_band


# ---------------------------------------------------------------------------
# Req 2.4 - effective discount identity and price echo
# ---------------------------------------------------------------------------
def test_effective_discount_identity_and_price_echo(db_session, monkeypatch):
    """Req 2.4: effective discount == (ref - disp)/ref*100 and prices are echoed."""
    _seed_stats(db_session)
    monkeypatch.setattr(discount_model, "predict_genuineness", lambda f, m: 0.5)

    displayed, reference = 1499.0, 4999.0
    result = check_discount(
        db_session,
        _CATEGORY,
        displayed_price=displayed,
        reference_price=reference,
        model=object(),
    )

    assert result["displayed_price"] == displayed
    assert result["reference_price"] == reference
    assert math.isclose(
        result["effective_discount_pct"],
        (reference - displayed) / reference * 100.0,
        rel_tol=1e-12,
    )


# ---------------------------------------------------------------------------
# Req 2.1 / 2.3 - normal path against the real trained model
# ---------------------------------------------------------------------------
def test_normal_scored_result_uses_real_model(db_session):
    """Req 2.1/2.3: real model -> integer score in [0,100], valid band, features."""
    _seed_stats(db_session)

    model = discount_model.get_model()
    if model is None:  # pragma: no cover - model artifact is expected in this repo
        pytest.skip("Trained discount model artifact is unavailable.")

    result = check_discount(
        db_session,
        _CATEGORY,
        displayed_price=1499.0,
        reference_price=4999.0,
    )

    score = result["genuineness_score"]
    assert isinstance(score, int)
    assert 0 <= score <= 100  # Req 2.1
    assert result["classification"] in {
        CLASSIFICATION_GENUINE,
        CLASSIFICATION_MODERATE,
        CLASSIFICATION_LIKELY_INFLATED,
    }
    # The engineered features are returned so the endpoint can reuse them for
    # the SHAP breakdown without re-engineering (Req 2.3).
    assert set(result["features"].keys()) == set(FEATURE_NAMES)
    assert math.isclose(
        result["effective_discount_pct"],
        (4999.0 - 1499.0) / 4999.0 * 100.0,
        rel_tol=1e-9,
    )


if __name__ == "__main__":  # pragma: no cover - manual "report outputs" run
    import json

    session, engine = _make_session()
    try:
        limited = check_discount(
            session, "unknown/category", displayed_price=1499.0, reference_price=4999.0
        )
        _seed_stats(session)
        try:
            check_discount(session, _CATEGORY, displayed_price=1499.0, reference_price=None)
            not_evaluable = "no error raised"
        except AppError as exc:
            not_evaluable = exc.to_payload().model_dump()
        model = discount_model.get_model()
        scored = (
            check_discount(session, _CATEGORY, displayed_price=1499.0, reference_price=4999.0)
            if model is not None
            else "model unavailable"
        )
    finally:
        session.close()
        engine.dispose()

    print("--- verification_limited (no stats) ---")
    print(json.dumps(limited, indent=2, default=str))
    print("--- not_evaluable (missing reference) ---")
    print(json.dumps(not_evaluable, indent=2, default=str))
    print("--- scored (real model) ---")
    print(json.dumps(scored, indent=2, default=str))
