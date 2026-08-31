"""Property-based tests for the True Discount Checker service (Task 8.2).

These tests exercise ``app.services.discount_service.check_discount`` with
Hypothesis, implementing four of the design's Correctness Properties:

* Property 2 - Genuineness score is always within range                 (Req 2.1)
* Property 3 - Discount band is a correct total function of the score    (Req 2.2)
* Property 4 - Effective discount percentage identity                    (Req 2.4)
* Property 5 - A discount cannot be evaluated without a valid reference  (Req 2.5)

Each property is implemented by exactly one property-based test carrying its
traceability comment and running at least 100 generated examples.

The category price statistics the service reads are seeded once into a
module-level in-memory SQLite database built from ``app.db.models.Base.metadata``
(``StaticPool`` so the ``:memory:`` connection - and the seeded row - survives
across every session and example). Seeding once at import time, rather than
through a function-scoped pytest fixture, keeps the generated-input loop off the
database-setup path and avoids Hypothesis's function-scoped-fixture health
check; each example just opens a lightweight read-only session on the shared
engine.

For Properties 2 and 3 the model probability is controlled by patching
``app.ml.discount_model.predict_genuineness`` (the discount service calls it as
``discount_model.predict_genuineness``, so the patch must live on that module)
and passing ``model=object()`` so the real ``get_model()`` loader is never
needed. Property 4 holds on the scored path regardless of the score, and
Property 5 asserts the not-evaluable pre-condition fires before any scoring is
attempted.
"""

import math
from unittest.mock import patch

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.errors import AppError
from app.db.models import Base, CategoryPriceStats
from app.ml import discount_model
from app.services.discount_service import (
    CLASSIFICATION_GENUINE,
    CLASSIFICATION_LIKELY_INFLATED,
    CLASSIFICATION_MODERATE,
    GENUINE_THRESHOLD,
    MODERATE_THRESHOLD,
    NOT_EVALUABLE_CODE,
    SCORE_MAX,
    SCORE_MIN,
    check_discount,
)

# The single category every example scores against; a constant label keeps the
# seeded stats row and the service call in agreement (the properties do not
# depend on the label varying).
_CATEGORY = "electronics/headphones"


def _build_seeded_engine():
    """Build one in-memory SQLite engine seeded with a category stats row.

    ``StaticPool`` keeps a single underlying connection so the ``:memory:``
    database (and the seeded ``CategoryPriceStats`` row) survives across the many
    sessions the examples open. Called once at import time so the
    generated-input loop never rebuilds or reseeds the database.
    """

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as seed_session:
        seed_session.add(
            CategoryPriceStats(
                category=_CATEGORY,
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
        seed_session.commit()
    return engine


_ENGINE = _build_seeded_engine()
_SessionFactory = sessionmaker(bind=_ENGINE, expire_on_commit=False)


# Positive, finite prices. Kept strictly positive so a reference above the
# displayed price is a well-defined, non-degenerate discount.
_positive_price = st.floats(
    min_value=1e-2, max_value=1e6, allow_nan=False, allow_infinity=False
)

# A model probability p(genuine) in the closed unit interval - the value the
# patched inference returns for Properties 2 and 3.
_probability = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)


def _patched_probability(probability):
    """Patch the service's inference to return a fixed ``probability``.

    The discount service calls ``discount_model.predict_genuineness(features,
    model)``, so the patch is applied to that attribute on the ``discount_model``
    module. The replacement ignores its arguments and returns the controlled
    probability, letting a test drive the resulting score deterministically.
    """

    return patch.object(
        discount_model,
        "predict_genuineness",
        lambda features, model: probability,
    )


# Feature: price-truth-platform, Property 2: Genuineness score is always within range
@settings(max_examples=200, deadline=None)
@given(displayed=_positive_price, delta=_positive_price, probability=_probability)
def test_genuineness_score_is_within_range(displayed, delta, probability):
    """Validates: Requirements 2.1

    For any evaluable discount whose category statistics exist and with a model
    available, the returned genuineness score is an integer in the closed
    interval [0, 100]. The model probability is controlled via a patched
    ``predict_genuineness`` returning a generated p in [0, 1]; the service maps
    it to ``round(p * 100)`` clamped to [0, 100].
    """
    reference = displayed + delta  # strictly greater than displayed -> evaluable

    session = _SessionFactory()
    try:
        with _patched_probability(probability):
            result = check_discount(
                session,
                _CATEGORY,
                displayed_price=displayed,
                reference_price=reference,
                model=object(),  # non-None sentinel bypasses get_model()
            )
    finally:
        session.close()

    score = result["genuineness_score"]
    assert isinstance(score, int)
    assert SCORE_MIN <= score <= SCORE_MAX
    # The score is exactly the probability mapped onto the integer scale.
    assert score == max(SCORE_MIN, min(SCORE_MAX, round(probability * 100)))


# Feature: price-truth-platform, Property 3: Discount band is a correct total function of the score
@settings(max_examples=200, deadline=None)
@given(score=st.integers(min_value=SCORE_MIN, max_value=SCORE_MAX))
@example(score=59)  # just below the moderate band
@example(score=60)  # lower moderate boundary
@example(score=89)  # just below the genuine band
@example(score=90)  # lower genuine boundary
def test_discount_band_is_total_function_of_score(score):
    """Validates: Requirements 2.2

    For any score in [0, 100] (driven by a patched ``predict_genuineness``
    returning ``score / 100``), the classification is exactly one band:
    ``score >= 90`` -> genuine, ``60 <= score < 90`` -> moderate,
    ``score < 60`` -> likely_inflated. The boundaries 59, 60, 89, and 90 are
    pinned as explicit examples.
    """
    probability = score / 100.0

    session = _SessionFactory()
    try:
        with _patched_probability(probability):
            result = check_discount(
                session,
                _CATEGORY,
                displayed_price=1000.0,
                reference_price=5000.0,
                model=object(),
            )
    finally:
        session.close()

    returned_score = result["genuineness_score"]
    assert returned_score == score  # round(score / 100 * 100) == score

    if score >= GENUINE_THRESHOLD:
        expected_band = CLASSIFICATION_GENUINE
    elif score >= MODERATE_THRESHOLD:
        expected_band = CLASSIFICATION_MODERATE
    else:
        expected_band = CLASSIFICATION_LIKELY_INFLATED

    assert result["classification"] == expected_band


# Feature: price-truth-platform, Property 4: Effective discount percentage identity
@settings(max_examples=200, deadline=None)
@given(displayed=_positive_price, delta=_positive_price)
def test_effective_discount_percentage_identity(displayed, delta):
    """Validates: Requirements 2.4

    For any displayed and reference price with reference > displayed (both
    positive), the reported effective discount percentage equals
    ``(reference - displayed) / reference * 100`` within a small numeric
    tolerance, and the result echoes both the displayed and reference prices.
    """
    reference = displayed + delta  # strictly greater than displayed -> evaluable

    session = _SessionFactory()
    try:
        # A constant probability keeps the scored path deterministic; the
        # effective-discount identity does not depend on the score.
        with _patched_probability(0.5):
            result = check_discount(
                session,
                _CATEGORY,
                displayed_price=displayed,
                reference_price=reference,
                model=object(),
            )
    finally:
        session.close()

    assert result["displayed_price"] == displayed
    assert result["reference_price"] == reference
    assert math.isclose(
        result["effective_discount_pct"],
        (reference - displayed) / reference * 100.0,
        rel_tol=1e-9,
        abs_tol=0.0,
    )


def _fail_if_scored(features, model):
    """Sentinel inference that fails if scoring is ever attempted.

    Installed for Property 5 so that reaching the scoring path (instead of
    short-circuiting on the not-evaluable pre-condition) raises rather than
    silently producing a genuineness score.
    """

    raise AssertionError(
        "predict_genuineness must not be called for a non-evaluable discount"
    )


# Feature: price-truth-platform, Property 5: A discount cannot be evaluated without a valid reference
@settings(max_examples=200, deadline=None)
@given(data=st.data())
def test_missing_or_invalid_reference_is_not_evaluable(data):
    """Validates: Requirements 2.5

    For any input where the reference price is missing (``None``) or is not
    strictly greater than the displayed price, ``check_discount`` raises
    ``AppError`` with code ``DISCOUNT_NOT_EVALUABLE`` and a stated reason, and
    never returns a genuineness score (the sentinel inference guarantees scoring
    is never attempted).
    """
    displayed = data.draw(_positive_price)
    # Reference is either missing or <= displayed (equal, below, zero, or
    # negative) - every value that makes the discount non-evaluable.
    reference = data.draw(
        st.one_of(
            st.none(),
            st.floats(
                min_value=-1e6,
                max_value=displayed,
                allow_nan=False,
                allow_infinity=False,
            ),
        )
    )

    session = _SessionFactory()
    try:
        with patch.object(discount_model, "predict_genuineness", _fail_if_scored):
            with pytest.raises(AppError) as exc_info:
                check_discount(
                    session,
                    _CATEGORY,
                    displayed_price=displayed,
                    reference_price=reference,
                    model=object(),
                )
    finally:
        session.close()

    error = exc_info.value
    assert error.code == NOT_EVALUABLE_CODE
    assert error.status == 422
    assert error.details.get("reason")  # a stated reason is present
