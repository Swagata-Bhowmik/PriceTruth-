"""Verification for the model-serving layer (Tasks 4.4 + 4.5).

These tests cover the load-once inference module and the SHAP explainer as one
coherent unit, because both hang off the *single* model instance loaded per
process:

* **Load-once (Req 12.4, 11.2).** :func:`app.ml.discount_model.get_model` loads
  the serialized XGBoost classifier at most once and returns the identical
  object on every later call, so inference and explanation never re-read the
  pickle.
* **Inference (Req 2.3, 2.1).** :func:`app.ml.discount_model.predict_genuineness`
  returns ``p(genuine)`` in [0, 1], and ranks a genuine-looking listing above an
  inflated-looking one.
* **Explainability (Req 3.1-3.5, Properties 6/7/8).**
  :func:`app.ml.explainer.explain` returns exactly one plain-language
  contribution per model feature, with a direction that matches each
  contribution's sign, and a breakdown that reconciles to the model's margin
  output (``base_value + sum(contributions) == margin`` within tolerance).
* **Startup wiring (Req 12.4, 15.1).** ``app.main`` loads the model and explainer
  once into ``app.state`` at startup, and a *missing* model must not crash
  startup - the app still boots and serves.

The genuine/inflated feature fixtures reuse the same category distribution as
``tests/test_discount_features.py`` (means/percentiles in price units, discount
stats as percentages) so the engineered features are realistic.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.ml import discount_model
from app.ml.discount_model import (
    FEATURE_NAMES,
    engineer_features,
    features_to_vector,
    get_model,
    predict_genuineness,
)
from app.ml.explainer import (
    RECONCILIATION_TOLERANCE,
    TOWARD_GENUINE,
    TOWARD_INFLATED,
    explain,
    get_explainer,
)
from app.ml.feature_labels import FEATURE_LABELS, label_for, unlabeled_features

# A representative category distribution (see test_discount_features.py). Price
# stats are in price units; discount stats are percentages in [0, 100].
CATEGORY_STATS = {
    "mean_price": 1000.0,
    "median_price": 1000.0,
    "std_price": 300.0,
    "p25_price": 800.0,
    "p75_price": 1200.0,
    "mean_discount_pct": 15.0,
    "std_discount_pct": 10.0,
    "mean_rating": 4.0,
    "mean_rating_count": 500.0,
}

# A genuine-looking listing: the selling price sits well below the category norm
# with a reference inside the normal band (a real markdown).
GENUINE_FEATURES = engineer_features(
    displayed_price=500.0,
    reference_price=1100.0,
    category_stats=CATEGORY_STATS,
    rating=4.4,
    rating_count=900,
)
# An inflated-looking listing: the selling price sits at the norm while the
# reference/"original" is a high outlier (a manufactured discount).
INFLATED_FEATURES = engineer_features(
    displayed_price=1000.0,
    reference_price=5000.0,
    category_stats=CATEGORY_STATS,
    rating=4.0,
    rating_count=500,
)

_FEATURE_LABEL_SET = {FEATURE_LABELS[name] for name in FEATURE_NAMES}


def _independent_margin(features) -> float:
    """Compute the model's raw margin output independently of the SHAP decomposition.

    Used to make the reconciliation checks a genuine test (Property 8) rather
    than a tautology: the margin here comes straight from the model, not from
    ``base_value + sum(contributions)``.
    """

    model = get_model()
    frame = pd.DataFrame([features_to_vector(features)], columns=list(FEATURE_NAMES))
    return float(model.predict(frame, output_margin=True)[0])


# ---------------------------------------------------------------------------
# Load-once inference (Task 4.4, Req 12.4 / 11.2)
# ---------------------------------------------------------------------------
def test_get_model_returns_a_usable_classifier():
    """The trained model loads and exposes the inference surface we rely on."""
    model = get_model()

    assert model is not None
    assert hasattr(model, "predict_proba")
    # classes_ is how predict_genuineness locates the genuine column.
    assert hasattr(model, "classes_")


def test_get_model_loads_once_and_returns_same_instance():
    """(a) The model is loaded once per process and reused (Req 12.4)."""
    first = get_model()
    second = get_model()

    assert first is second


# ---------------------------------------------------------------------------
# predict_genuineness (Task 4.4, Req 2.3 / 2.1)
# ---------------------------------------------------------------------------
def test_predict_genuineness_returns_probability_in_unit_interval():
    """(b) predict_genuineness returns a float in the closed interval [0, 1]."""
    p_genuine = predict_genuineness(GENUINE_FEATURES)

    assert isinstance(p_genuine, float)
    assert 0.0 <= p_genuine <= 1.0


def test_predict_genuineness_ranks_genuine_above_inflated():
    """(b) A genuine-looking listing scores a higher p(genuine) than an inflated one."""
    assert predict_genuineness(GENUINE_FEATURES) > predict_genuineness(INFLATED_FEATURES)


def test_predict_genuineness_accepts_an_explicit_model_instance():
    """Passing the loaded model explicitly matches the default (cached) path."""
    model = get_model()

    assert predict_genuineness(GENUINE_FEATURES, model=model) == pytest.approx(
        predict_genuineness(GENUINE_FEATURES)
    )


def test_predict_genuineness_raises_when_no_model_is_available(monkeypatch):
    """With no model loadable, scoring raises rather than fabricating a score."""
    monkeypatch.setattr(discount_model, "get_model", lambda: None)

    with pytest.raises(RuntimeError):
        predict_genuineness(GENUINE_FEATURES, model=None)


# ---------------------------------------------------------------------------
# Plain-language feature labels (Task 4.5, Req 3.5)
# ---------------------------------------------------------------------------
def test_label_map_covers_every_model_feature():
    """Every raw feature has a plain-language label (map stays in lock-step)."""
    assert unlabeled_features() == ()
    for name in FEATURE_NAMES:
        assert name in FEATURE_LABELS


def test_label_for_returns_plain_language_never_the_raw_identifier():
    """label_for maps to prose and never returns a raw feature identifier (Req 3.5)."""
    for name in FEATURE_NAMES:
        label = label_for(name)
        assert label == FEATURE_LABELS[name]
        assert label != name
        assert label not in set(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# SHAP explainer (Task 4.5, Req 3.1 / 3.2 / 3.3 / 3.4 / 3.5, Properties 6/7/8)
# ---------------------------------------------------------------------------
def test_explainer_is_built_once_and_reused():
    """The SHAP explainer is constructed once per process and reused (Req 3.4/12.4)."""
    assert get_explainer() is not None
    assert get_explainer() is get_explainer()


@pytest.mark.parametrize("features", [GENUINE_FEATURES, INFLATED_FEATURES])
def test_explain_returns_one_plain_language_contribution_per_feature(features):
    """(c) Exactly one contribution per feature, each plainly labelled (Req 3.1/3.5, Property 6)."""
    result = explain(features)
    contributions = result["contributions"]

    # Req 3.1 / Property 6: exactly one entry per model feature.
    assert len(contributions) == len(FEATURE_NAMES)

    labels = [c["feature"] for c in contributions]
    # Req 3.5: labels are plain-language, never raw identifiers, one per feature.
    assert all(label not in set(FEATURE_NAMES) for label in labels)
    assert set(labels) == _FEATURE_LABEL_SET


@pytest.mark.parametrize("features", [GENUINE_FEATURES, INFLATED_FEATURES])
def test_explain_direction_matches_contribution_sign(features):
    """(c) Direction is toward_genuine for positive impact, toward_inflated for negative (Property 7)."""
    for contribution in explain(features)["contributions"]:
        impact = contribution["impact"]
        direction = contribution["direction"]

        if impact > 0:
            assert direction == TOWARD_GENUINE
        elif impact < 0:
            assert direction == TOWARD_INFLATED

        # Direction is a total function of the sign.
        assert direction in (TOWARD_GENUINE, TOWARD_INFLATED)
        assert (direction == TOWARD_GENUINE) == (impact >= 0.0)


@pytest.mark.parametrize(
    "name, features", [("genuine", GENUINE_FEATURES), ("inflated", INFLATED_FEATURES)]
)
def test_explain_reconciles_to_model_margin(name, features, capsys):
    """(c) base_value + sum(contributions) == model margin output (Req 3.3, Property 8)."""
    result = explain(features)
    contribution_sum = sum(c["impact"] for c in result["contributions"])
    total = result["base_value"] + contribution_sum
    margin = _independent_margin(features)

    # The breakdown reconciles to the model's margin output within tolerance...
    assert math.isclose(total, margin, abs_tol=RECONCILIATION_TOLERANCE)
    # ...and the presented final_score IS that margin (so it reconciles, Req 3.3).
    assert math.isclose(result["final_score"], margin, abs_tol=RECONCILIATION_TOLERANCE)

    # Surface the reconciliation numbers for the task report (visible even at -q).
    with capsys.disabled():
        print(
            f"\n  [{name}] reconciliation: base_value={result['base_value']:.6f} "
            f"sum(contributions)={contribution_sum:.6f} "
            f"base+sum={total:.6f} model_margin={margin:.6f} "
            f"final_score={result['final_score']:.6f} "
            f"|diff|={abs(total - margin):.2e} (tol={RECONCILIATION_TOLERANCE:g})"
        )


@pytest.mark.parametrize("features", [GENUINE_FEATURES, INFLATED_FEATURES])
def test_explanation_margin_is_consistent_with_predicted_probability(features):
    """Explanation and score come from the same model: sigmoid(margin) == p(genuine) (Req 3.4)."""
    final_margin = explain(features)["final_score"]
    p_from_margin = 1.0 / (1.0 + math.exp(-final_margin))

    assert math.isclose(p_from_margin, predict_genuineness(features), abs_tol=1e-4)


# ---------------------------------------------------------------------------
# Startup wiring (Task 4.4, Req 12.4 / 15.1)
# ---------------------------------------------------------------------------
def test_startup_loads_model_and_explainer_into_app_state():
    """Startup stashes the single model + explainer on app.state (Req 12.4)."""
    # Entering the TestClient context manager runs the lifespan startup.
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert app.state.discount_model is get_model()
        assert app.state.discount_explainer is get_explainer()


def test_startup_survives_a_missing_model_without_crashing(monkeypatch):
    """A missing model must NOT crash startup; the app still boots and serves (Req 15.1)."""
    # Simulate a deployment with no trained model artefact.
    monkeypatch.setattr(main_module, "get_model", lambda: None)

    with TestClient(app) as client:
        # Startup completed (context entered) and the app still serves requests.
        assert client.get("/").status_code == 200
        assert app.state.discount_model is None
        assert app.state.discount_explainer is None
