"""
dashboard/app.py

Flask dashboard for the IoT Intrusion Detection project. Provides:

    - A homepage summarizing model comparison results (GNNs + baselines).
    - JSON API endpoints for the comparison table, class distribution, and
      per-model training history (consumed by static/js/dashboard.js to
      render charts client-side).
    - A route that serves saved figures from results/figures/.
    - A live single-flow prediction demo: paste in feature values for one
      network flow, get back a predicted attack category. Uses the trained
      baseline ML models (Random Forest / XGBoost / Logistic Regression)
      rather than the GNNs, since a single isolated flow has no graph
      neighbors for a GNN to aggregate over -- the baselines are the right
      tool for real-time, single-record inference.

Run with:
    cd iot-intrusion-detection
    python dashboard/app.py
Then open http://127.0.0.1:5000 in a browser.
"""

import os
import sys

# Allow running this file directly (`python dashboard/app.py`) by adding the
# project root to sys.path, since dashboard/ sits outside the src package.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, send_from_directory, request

from src import config
from src.data.feature_selection import get_selected_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded, cached resources (loaded once on first use, not at import
# time, so the dashboard still starts even if some artifacts don't exist
# yet -- e.g. before training has been run).
# ---------------------------------------------------------------------------
_scaler_cache = None
_baseline_model_cache = {}

BASELINE_MODEL_FILES = {
    "RandomForest": "randomforest_baseline.joblib",
    "XGBoost": "xgboost_baseline.joblib",
    "LogisticRegression": "logisticregression_baseline.joblib",
}


def get_scaler():
    """Load and cache the fitted StandardScaler saved by preprocessing.py."""
    global _scaler_cache
    if _scaler_cache is None:
        scaler_path = os.path.join(config.PROCESSED_DATA_DIR, "feature_scaler.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. Run "
                f"src/data/preprocessing.py first."
            )
        _scaler_cache = joblib.load(scaler_path)
    return _scaler_cache


def get_baseline_model(model_key: str):
    """Load and cache a trained baseline model by its display name."""
    if model_key not in BASELINE_MODEL_FILES:
        raise ValueError(
            f"Unknown baseline model '{model_key}'. Must be one of "
            f"{list(BASELINE_MODEL_FILES.keys())}."
        )

    if model_key not in _baseline_model_cache:
        model_path = os.path.join(config.MODELS_DIR, BASELINE_MODEL_FILES[model_key])
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run "
                f"src/models/baseline_ml.py first."
            )
        _baseline_model_cache[model_key] = joblib.load(model_path)

    return _baseline_model_cache[model_key]


# ---------------------------------------------------------------------------
# PAGE ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Homepage: model comparison table + links to charts/figures."""
    comparison_path = os.path.join(config.RESULTS_DIR, "comparison.csv")
    comparison_records = []

    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        comparison_records = comparison_df.to_dict(orient="records")

    return render_template(
        "index.html",
        comparison_records=comparison_records,
        class_names=config.CLASS_NAMES,
        baseline_model_names=list(BASELINE_MODEL_FILES.keys()),
    )


# ---------------------------------------------------------------------------
# JSON API ROUTES (consumed by static/js/dashboard.js)
# ---------------------------------------------------------------------------

@app.route("/api/comparison")
def api_comparison():
    """Return results/comparison.csv as JSON records."""
    comparison_path = os.path.join(config.RESULTS_DIR, "comparison.csv")
    if not os.path.exists(comparison_path):
        return jsonify({"error": "comparison.csv not found. Run evaluation first."}), 404

    df = pd.read_csv(comparison_path)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/class-distribution")
def api_class_distribution():
    """Return class counts from the processed training set as JSON."""
    if not os.path.exists(config.TRAIN_CSV_PATH):
        return jsonify({"error": "train.csv not found. Run preprocessing first."}), 404

    df = pd.read_csv(config.TRAIN_CSV_PATH, usecols=["Category"])
    counts = df["Category"].value_counts().to_dict()
    return jsonify(counts)


@app.route("/api/training-history/<model_name>")
def api_training_history(model_name):
    """Return a GNN model's per-epoch training history as JSON."""
    history_path = os.path.join(
        config.METRICS_DIR, f"{model_name.lower()}_training_history.csv"
    )
    if not os.path.exists(history_path):
        return jsonify({"error": f"No training history found for '{model_name}'."}), 404

    df = pd.read_csv(history_path)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/confusion-matrix/<model_name>")
def api_confusion_matrix(model_name):
    """Return a model's confusion matrix (CSV) as JSON."""
    cm_path = os.path.join(
        config.METRICS_DIR, f"{model_name.lower()}_confusion_matrix.csv"
    )
    if not os.path.exists(cm_path):
        return jsonify({"error": f"No confusion matrix found for '{model_name}'."}), 404

    df = pd.read_csv(cm_path, index_col=0)
    return jsonify(
        {
            "labels": df.columns.tolist(),
            "matrix": df.values.tolist(),
        }
    )


@app.route("/api/feature-names")
def api_feature_names():
    """Return the selected feature list, used to build the prediction form."""
    return jsonify(get_selected_features())


# ---------------------------------------------------------------------------
# FIGURE SERVING
# ---------------------------------------------------------------------------

@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serve a saved plot image from results/figures/."""
    return send_from_directory(config.FIGURES_DIR, filename)


# ---------------------------------------------------------------------------
# LIVE PREDICTION DEMO
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a single flow record's feature values and return a predicted
    attack category using one of the trained baseline ML models.

    Expected JSON body:
        {
            "model": "RandomForest",   // one of BASELINE_MODEL_FILES keys
            "features": {"flow_duration": 0.12, "Rate": 3.4, ...}
        }

    Missing feature values default to 0.0 (post-scaling this represents
    the training-set mean, since StandardScaler centers features at 0).
    """
    try:
        payload = request.get_json(force=True)
        model_key = payload.get("model", "RandomForest")
        raw_features = payload.get("features", {})

        selected_features = get_selected_features()
        feature_vector = [
            float(raw_features.get(name, 0.0)) for name in selected_features
        ]
        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        scaler = get_scaler()
        # NOTE: the scaler was fit on the FULL config.FEATURE_COLUMNS set,
        # while `selected_features` may be a reduced subset from
        # feature_selection.py. If a reduced set is in use, this assumes
        # selected_features is a subset in the same relative order as
        # config.FEATURE_COLUMNS; for a strict pipeline, prediction-time
        # scaling should select the matching scaler sub-parameters.
        if len(selected_features) == len(config.FEATURE_COLUMNS):
            X_scaled = scaler.transform(X)
        else:
            logger.warning(
                "Selected feature count differs from the scaler's original "
                "fit set; skipping scaling for this prediction. Retrain "
                "with matching feature sets for fully consistent scaling."
            )
            X_scaled = X

        model = get_baseline_model(model_key)
        predicted_label = int(model.predict(X_scaled)[0])
        predicted_class = config.INDEX_TO_LABEL[predicted_label]

        response = {
            "predicted_class": predicted_class,
            "model_used": model_key,
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_scaled)[0]
            response["probabilities"] = {
                config.INDEX_TO_LABEL[i]: float(p) for i, p in enumerate(probabilities)
            }

        return jsonify(response)

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 404
    except Exception as exc:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
