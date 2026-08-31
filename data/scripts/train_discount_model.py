"""Train the True Discount Checker XGBoost model and persist it (task 4.3, Req 2.3).

This offline script turns the ingested snapshot corpus into the serialized
binary classifier the inference module (task 4.4) and the SHAP explainer
(task 4.5) load at startup. It is the one place the model is *trained*; the
feature contract, the feature transform, and the weak-supervision labeling rule
all live in :mod:`app.ml.discount_model` and are merely *applied* here, so the
training features are, by construction, identical to what request-time inference
computes for a single listing.

End-to-end it:

1. **Populates a throwaway training database** from the synthetic fixtures by
   calling :func:`ingest.ingest` against a local SQLite file under
   ``data/models/`` (``_train.db``). Importing ``ingest`` is what puts
   ``backend/`` on ``sys.path`` (a side effect of that module), which is how the
   ``app.*`` packages become importable here - the same pattern the other
   ``data/scripts`` use. The temp database is created fresh and removed at the
   end, so the script is safe to re-run.
2. **Reads the training rows** back out: ``price_snapshots`` joined to
   ``products`` (for the category) into a DataFrame of
   ``category, displayed_price, reference_price, rating, rating_count``, plus the
   per-category ``category_price_stats`` reduced into a ``stats_by_category``
   mapping.
3. **Engineers features and labels** through
   :func:`app.ml.discount_model.engineer_features_frame` (columns are exactly
   :data:`~app.ml.discount_model.FEATURE_NAMES`, in the frozen order) and
   :func:`app.ml.discount_model.label_frame` (the disclosed weak-supervision
   rule).
4. **Trains an XGBoost binary classifier** on a stratified 80/20 split with a
   fixed seed and prints accuracy / precision / recall / F1 on the held-out test
   split.
5. **Serializes the fitted model** with joblib to
   :data:`app.ml.discount_model.MODEL_PATH` (``data/models/discount_model.pkl``),
   the exact path task 4.4 loads and calls ``predict_proba`` on.

**Expected metrics.** The labels come from the *same* feature-derived
weak-supervision rule the model trains on, so the model is essentially learning
that transparent rule. **Near-perfect metrics on the synthetic fixtures are
therefore expected and fine** - they show the model has faithfully absorbed the
disclosed heuristic, not that it has discovered ground truth (the public data
carries no ground-truth "fake discount" label; see Req 10.1).

Usage::

    backend\\venv\\Scripts\\python.exe data\\scripts\\train_discount_model.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path wiring
# ---------------------------------------------------------------------------
# This script lives at ``data/scripts/train_discount_model.py`` alongside
# ``ingest.py``. Putting its own directory on ``sys.path`` lets us ``import
# ingest``; importing that module in turn appends ``backend/`` to ``sys.path``
# (see its "Path wiring" block), which is what makes the ``app.*`` packages
# importable below. This mirrors how the other data scripts bootstrap the app.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import ingest  # noqa: E402  -- side effect: puts backend/ on sys.path

# Now that ``backend/`` is importable, pull in the frozen feature contract and
# the shared transform/labeling helpers. Importing this module does NOT open a
# database connection (it depends only on the stdlib + a lazy pandas import), so
# it is safe to import before the training database is configured.
from app.ml.discount_model import (  # noqa: E402
    FEATURE_NAMES,
    LABEL_GENUINE,
    LABEL_INFLATED,
    MODEL_PATH,
    engineer_features,
    engineer_features_frame,
    features_to_vector,
    label_frame,
    label_row,
)

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402
from sqlalchemy import text  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
# Synthetic fixtures to train on (task 3.1). ``ingest`` exposes the repo root it
# computed, so reuse it rather than recomputing the ``parents[...]`` walk.
FIXTURES_DIR = ingest.PROJECT_ROOT / "data" / "raw" / "fixtures"

# The trained model is written next to the other model artefacts; the throwaway
# training database lives in the same directory and is deleted on exit. Using
# ``MODEL_PATH.parent`` keeps both in lock-step with wherever the model lives.
MODELS_DIR = MODEL_PATH.parent
TRAIN_DB_PATH = MODELS_DIR / "_train.db"
# SQLAlchemy SQLite URL for an absolute path: ``sqlite:///`` + posix-style path.
TRAIN_DB_URL = f"sqlite:///{TRAIN_DB_PATH.as_posix()}"

# Reproducibility: one seed drives both the split and the booster.
RANDOM_STATE = 42
TEST_SIZE = 0.20

# The join that reconstructs the training rows. ``category`` comes from
# ``products`` (the model is category-relative), the price / review signals from
# ``price_snapshots``.
_TRAINING_ROWS_SQL = """
    SELECT p.category            AS category,
           ps.displayed_price    AS displayed_price,
           ps.reference_price    AS reference_price,
           ps.rating             AS rating,
           ps.rating_count       AS rating_count
    FROM price_snapshots AS ps
    JOIN products AS p ON ps.product_id = p.id
"""

_CATEGORY_STATS_SQL = "SELECT * FROM category_price_stats"


# ---------------------------------------------------------------------------
# Temp-database lifecycle
# ---------------------------------------------------------------------------
def _remove_train_db() -> None:
    """Delete the throwaway SQLite database and any WAL/journal siblings.

    Called both before ingestion (start clean, so a stale schema can never leak
    in) and in the ``finally`` block (leave nothing behind). The engine bound to
    this file must already be disposed on Windows, or the file would still be
    locked; :func:`main` disposes it before calling this on the way out.
    """

    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(TRAIN_DB_PATH) + suffix)
        try:
            candidate.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - diagnostics only
            print(f"  warning: could not remove {candidate.name}: {exc}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_training_frame(engine) -> pd.DataFrame:
    """Read the joined ``price_snapshots`` x ``products`` rows into a DataFrame."""

    with engine.connect() as conn:
        return pd.read_sql(text(_TRAINING_ROWS_SQL), conn)


def _load_stats_by_category(engine) -> dict[str, dict]:
    """Read ``category_price_stats`` into a ``category -> {stat fields}`` mapping.

    The feature transform accepts either an ORM row or a plain mapping per
    category (it reads fields via ``.get``/``getattr``), so a dict-of-dicts is a
    valid ``stats_by_category`` and keeps this script free of the ORM.
    """

    with engine.connect() as conn:
        stats_df = pd.read_sql(text(_CATEGORY_STATS_SQL), conn)
    # ``to_dict("index")`` with the category as index yields exactly
    # ``{category: {column: value, ...}}``.
    return stats_df.set_index("category").to_dict("index")


# ---------------------------------------------------------------------------
# Post-training sanity check
# ---------------------------------------------------------------------------
def _sample_probabilities(model, stats_by_category: dict[str, dict]) -> None:
    """Score one genuine-looking and one inflated-looking listing and report.

    Builds two synthetic listings from a real category's statistics so the check
    is grounded in the same distribution the model trained on:

    * **genuine** - the displayed price sits clearly *below* the category norm
      with a reference price in the normal band (a real markdown);
    * **inflated** - the displayed price sits *at* the category norm while the
      reference/"original" is a high outlier (a manufactured discount).

    A correctly-trained model should assign the genuine listing a higher
    ``p(genuine)`` than the inflated one. The features are engineered through the
    shared transform and ordered via :func:`features_to_vector`, so this is a
    faithful stand-in for a task-4.4 inference call.
    """

    # Pick a category with real price spread so the constructed z-scores are
    # meaningful (a zero-std category would collapse every z-score to 0).
    spread_categories = {
        cat: s for cat, s in stats_by_category.items() if float(s.get("std_price") or 0.0) > 0
    }
    if not spread_categories:
        print("  (no category with non-zero price spread; skipping sample scoring)")
        return
    category = max(spread_categories, key=lambda c: float(spread_categories[c]["std_price"]))
    stats = spread_categories[category]

    mean_price = float(stats["mean_price"])
    std_price = float(stats["std_price"])

    # Genuine: displayed well below the mean; reference modestly above displayed
    # but still inside the normal band.
    genuine_displayed = max(mean_price - 1.2 * std_price, 1.0)
    genuine_reference = mean_price + 0.3 * std_price
    # Inflated: displayed at the mean (near the norm); reference a clear outlier
    # far above the distribution.
    inflated_displayed = mean_price
    inflated_reference = mean_price + 3.5 * std_price

    genuine_feats = engineer_features(genuine_displayed, genuine_reference, stats)
    inflated_feats = engineer_features(inflated_displayed, inflated_reference, stats)

    sample_x = pd.DataFrame(
        [features_to_vector(genuine_feats), features_to_vector(inflated_feats)],
        columns=list(FEATURE_NAMES),
    )
    # classes_ == [inflated(0), genuine(1)], so column 1 is p(genuine).
    proba = model.predict_proba(sample_x)
    genuine_idx = list(model.classes_).index(LABEL_GENUINE)

    p_genuine_for_genuine = float(proba[0][genuine_idx])
    p_genuine_for_inflated = float(proba[1][genuine_idx])

    # The weak-supervision rule's own verdict on each constructed listing, shown
    # so the probabilities can be read against the intended label.
    genuine_label = label_row(genuine_displayed, genuine_reference, stats)
    inflated_label = label_row(inflated_displayed, inflated_reference, stats)

    print("-" * 70)
    print("SAMPLE predict_proba (reloaded model)")
    print(f"  reference category      : {category}")
    print(
        f"    (mean_price={mean_price:.2f}, std_price={std_price:.2f}, "
        f"p75_price={float(stats.get('p75_price') or 0.0):.2f})"
    )
    print(
        f"  genuine-looking listing : displayed={genuine_displayed:.2f} "
        f"reference={genuine_reference:.2f} "
        f"weak_label={_label_name(genuine_label)} "
        f"-> p(genuine)={p_genuine_for_genuine:.4f}"
    )
    print(
        f"  inflated-looking listing: displayed={inflated_displayed:.2f} "
        f"reference={inflated_reference:.2f} "
        f"weak_label={_label_name(inflated_label)} "
        f"-> p(genuine)={p_genuine_for_inflated:.4f}"
    )
    if p_genuine_for_genuine > p_genuine_for_inflated:
        print(
            "  OK: the genuine-looking listing scores a higher p(genuine) than "
            "the inflated-looking one."
        )
    else:
        print(
            "  WARNING: the genuine-looking listing did NOT score higher than the "
            "inflated-looking one."
        )


def _label_name(label: int) -> str:
    """Human-readable name for a weak-supervision label integer."""

    if label == LABEL_GENUINE:
        return "genuine"
    if label == LABEL_INFLATED:
        return "inflated"
    return f"unknown({label})"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("PRICE TRUTH - TRAIN DISCOUNT MODEL (task 4.3, Req 2.3)")
    print("=" * 70)

    # data/models/ holds both the throwaway training DB and the model artefact.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Start from a clean slate so a stale temp DB can never contaminate a run.
    _remove_train_db()

    print(f"Fixtures directory : {FIXTURES_DIR}")
    print(f"Training database  : {TRAIN_DB_URL}")
    print(f"Model output       : {MODEL_PATH}")
    print("-" * 70)

    engine = None
    try:
        # 1. Populate the throwaway training database from the synthetic fixtures.
        print("Ingesting synthetic fixtures into the training database ...")
        counts = ingest.ingest(path=FIXTURES_DIR, database_url=TRAIN_DB_URL)
        print(
            f"  ingested: {counts.products} products, "
            f"{counts.price_snapshots} price snapshots, "
            f"{counts.category_price_stats} category-stat rows"
        )

        # ``ingest`` imported app.db.session with DATABASE_URL pointing at the
        # temp DB, so its engine is now bound to that file. Reuse it for reads
        # (one engine to dispose before the file can be deleted on Windows).
        from app.db.session import engine  # noqa: E402  (bound to the temp DB)

        # 2. Read the training rows and the per-category stats back out.
        df = _load_training_frame(engine)
        stats_by_category = _load_stats_by_category(engine)
        print(f"  loaded {len(df)} training rows across {len(stats_by_category)} categories")

        if df.empty:
            print("ERROR: no training rows were loaded; aborting.")
            return 1

        # 3. Engineer features (frozen FEATURE_NAMES order) and weak-supervision
        #    labels through the shared helpers.
        x = engineer_features_frame(df, stats_by_category)
        y = label_frame(df, stats_by_category)

        n_genuine = int((y == LABEL_GENUINE).sum())
        n_inflated = int((y == LABEL_INFLATED).sum())
        print("-" * 70)
        print(f"Feature matrix     : {x.shape[0]} rows x {x.shape[1]} features")
        print(f"Feature order      : {list(x.columns)}")
        print(
            f"Label distribution : genuine(1)={n_genuine}, inflated(0)={n_inflated}"
        )

        # 4. Stratified 80/20 split (fixed seed). Fall back to an unstratified
        #    split only in the degenerate case of a single-member class.
        stratify = y if min(n_genuine, n_inflated) >= 2 else None
        if stratify is None:
            print(
                "  note: a class has fewer than 2 members; using an unstratified split."
            )
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
        print(f"Train / test split : {len(x_train)} train, {len(x_test)} test")

        # 4b. Train the XGBoost binary classifier.
        print("-" * 70)
        print("Training XGBoost classifier ...")
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=4,
        )
        model.fit(x_train, y_train)

        # 4c. Evaluate on the held-out test split (positive class = genuine(1)).
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=LABEL_GENUINE, zero_division=0)
        recall = recall_score(y_test, y_pred, pos_label=LABEL_GENUINE, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=LABEL_GENUINE, zero_division=0)

        print("-" * 70)
        print("TEST-SPLIT METRICS (positive class = genuine)")
        print(f"  accuracy : {accuracy:.4f}")
        print(f"  precision: {precision:.4f}")
        print(f"  recall   : {recall:.4f}")
        print(f"  f1       : {f1:.4f}")
        print(
            "  note: labels come from the same feature-derived weak-supervision "
            "rule the model\n"
            "        trains on, so near-perfect metrics on the synthetic fixtures "
            "are EXPECTED\n"
            "        (the model is faithfully learning the disclosed heuristic, "
            "not ground truth)."
        )

        # 5. Serialize the fitted model to the path task 4.4 loads.
        joblib.dump(model, MODEL_PATH)
        size_kb = MODEL_PATH.stat().st_size / 1024.0
        print("-" * 70)
        print(f"Model persisted    : {MODEL_PATH} ({size_kb:.1f} KB)")

        # Reload it (as task 4.4 will) and run the sample sanity check.
        reloaded = joblib.load(MODEL_PATH)
        _sample_probabilities(reloaded, stats_by_category)

        print("=" * 70)
        print("DONE")
        print("=" * 70)
        return 0
    finally:
        # Dispose the engine so the SQLite file is unlocked (Windows), then
        # remove the throwaway database so the script stays re-runnable.
        if engine is not None:
            engine.dispose()
        _remove_train_db()


if __name__ == "__main__":
    raise SystemExit(main())
