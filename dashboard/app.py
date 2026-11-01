
"""
dashboard/app.py

Flask backend for the IoT Intrusion Detection dashboard.

Provides:
- Dashboard homepage
- Model comparison API
- Class distribution API
- Model training history API
- Model confusion matrix API
- Selected feature names API
- Dynamic graph explorer APIs
- Figure serving
- Single-flow prediction using trained baseline ML models

Run from the project root:
    python dashboard/app.py

Dashboard:
    http://127.0.0.1:5000
"""

import os
import sys
import glob
import logging

import numpy as np
import pandas as pd
import torch
import joblib

from sklearn.metrics import confusion_matrix
from flask import (
    Flask,
    render_template,
    jsonify,
    send_from_directory,
    request,
)

# ---------------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.data.feature_selection import get_selected_features


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# CACHES AND MODEL CONFIGURATION
# ---------------------------------------------------------------------------

_scaler_cache = None
_baseline_model_cache = {}
_baseline_confusion_cache = {}

BASELINE_MODEL_FILES = {
    "RandomForest": "randomforest_baseline.joblib",
    "XGBoost": "xgboost_baseline.joblib",
    "LogisticRegression": "logisticregression_baseline.joblib",
}


# ---------------------------------------------------------------------------
# RESOURCE LOADING
# ---------------------------------------------------------------------------

def get_scaler():
    """Load and cache the fitted feature scaler."""

    global _scaler_cache

    if _scaler_cache is None:

        scaler_path = os.path.join(
            config.PROCESSED_DATA_DIR,
            "feature_scaler.joblib",
        )

        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(
                f"Scaler not found at {scaler_path}. "
                "Run the preprocessing pipeline first."
            )

        _scaler_cache = joblib.load(scaler_path)

        logger.info("Feature scaler loaded successfully.")

    return _scaler_cache


def get_baseline_model(model_key):
    """Load and cache a trained baseline model."""

    if model_key not in BASELINE_MODEL_FILES:
        raise ValueError(
            f"Unknown model '{model_key}'. Available models: "
            f"{list(BASELINE_MODEL_FILES.keys())}"
        )

    if model_key not in _baseline_model_cache:

        model_path = os.path.join(
            config.MODELS_DIR,
            BASELINE_MODEL_FILES[model_key],
        )

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run src/models/baseline_ml.py first."
            )

        _baseline_model_cache[model_key] = joblib.load(model_path)

        logger.info("Loaded baseline model: %s", model_key)

    return _baseline_model_cache[model_key]


def scale_prediction_features(X, selected_features):
    """
    Scale prediction features using the matching scaler parameters.

    Supports:
    1. A scaler fitted on the complete feature set.
    2. A scaler fitted on a named feature subset.
    3. A scaler fitted on the selected features in their current order.

    Raises an error if feature alignment cannot be established safely.
    """

    scaler = get_scaler()

    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError(
            "The saved scaler does not contain mean_ and scale_ parameters."
        )

    # Prefer feature names saved by sklearn, when available.
    if hasattr(scaler, "feature_names_in_"):
        scaler_features = list(scaler.feature_names_in_)
    else:
        scaler_features = list(config.FEATURE_COLUMNS)

    if len(scaler_features) != len(scaler.mean_):
        raise ValueError(
            "Scaler feature metadata does not match its fitted parameters."
        )

    feature_indices = []

    for feature in selected_features:

        if feature not in scaler_features:
            raise ValueError(
                f"Feature '{feature}' is not present in the saved scaler."
            )

        feature_indices.append(scaler_features.index(feature))

    means = np.asarray(scaler.mean_)[feature_indices]
    scales = np.asarray(scaler.scale_)[feature_indices]

    if len(means) != X.shape[1]:
        raise ValueError(
            "Prediction feature count does not match scaler parameters."
        )

    # StandardScaler uses scale_=1 for constant features.
    X_scaled = (X - means) / scales

    return X_scaled.astype(np.float32)


# ---------------------------------------------------------------------------
# PAGE ROUTES
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Render the original Flask dashboard homepage."""

    comparison_path = os.path.join(
        config.RESULTS_DIR,
        "comparison.csv",
    )

    comparison_records = []

    if os.path.isfile(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        comparison_records = comparison_df.to_dict(orient="records")

    return render_template(
        "index.html",
        comparison_records=comparison_records,
        class_names=config.CLASS_NAMES,
        baseline_model_names=list(BASELINE_MODEL_FILES.keys()),
    )


# ---------------------------------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------------------------------

@app.route("/api/comparison")
def api_comparison():
    """Return model comparison records."""

    comparison_path = os.path.join(
        config.RESULTS_DIR,
        "comparison.csv",
    )

    if not os.path.isfile(comparison_path):
        return jsonify({
            "error": "comparison.csv not found. Run evaluation first."
        }), 404

    df = pd.read_csv(comparison_path)

    return jsonify(df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------------------------

@app.route("/api/class-distribution")
def api_class_distribution():
    """Return category counts from the processed training dataset."""

    if not os.path.isfile(config.TRAIN_CSV_PATH):
        return jsonify({
            "error": "train.csv not found. Run preprocessing first."
        }), 404

    df = pd.read_csv(
        config.TRAIN_CSV_PATH,
        usecols=["Category"],
    )

    counts = df["Category"].value_counts().to_dict()

    return jsonify(counts)


# ---------------------------------------------------------------------------
# TRAINING HISTORY
# ---------------------------------------------------------------------------

@app.route("/api/training-history/<model_name>")
def api_training_history(model_name):
    """Return per-epoch training history for a model."""

    history_path = os.path.join(
        config.METRICS_DIR,
        f"{model_name.lower()}_training_history.csv",
    )

    if not os.path.isfile(history_path):
        return jsonify({
            "error": f"No training history found for '{model_name}'."
        }), 404

    df = pd.read_csv(history_path)

    return jsonify(df.to_dict(orient="records"))


# ---------------------------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------------------------

@app.route("/api/confusion-matrix/<model_name>")
def api_confusion_matrix(model_name):
    """
    Return a saved confusion matrix.

    GNN:
        Read the saved confusion matrix CSV.

    Baseline models:
        If the CSV does not exist, calculate it using the saved model
        and processed test dataset. No retraining is performed.
    """

    model_key = next(
        (
            name
            for name in BASELINE_MODEL_FILES
            if name.lower() == model_name.lower()
        ),
        None,
    )

    cm_path = os.path.join(
        config.METRICS_DIR,
        f"{model_name.lower()}_confusion_matrix.csv",
    )

    # First check whether a saved CSV exists.
    if os.path.isfile(cm_path):

        try:
            df = pd.read_csv(cm_path, index_col=0)

            return jsonify({
                "labels": df.columns.tolist(),
                "matrix": df.values.tolist(),
            })

        except Exception:
            logger.exception(
                "Could not read confusion matrix CSV for %s",
                model_name,
            )

    # Return 404 for missing GNN matrices.
    if model_key is None:
        return jsonify({
            "error": f"No confusion matrix found for '{model_name}'."
        }), 404

    # Return cached baseline matrix.
    if model_key in _baseline_confusion_cache:
        return jsonify(_baseline_confusion_cache[model_key])

    try:

        logger.info(
            "Generating confusion matrix for %s",
            model_key,
        )

        model = get_baseline_model(model_key)

        selected_features = get_selected_features()

        test_df = pd.read_csv(config.TEST_CSV_PATH)

        X_test = test_df[selected_features].values.astype(np.float32)
        y_test = test_df["Label"].values.astype(np.int64)

        y_pred = model.predict(X_test)

        labels = list(range(len(config.CLASS_NAMES)))

        matrix = confusion_matrix(
            y_test,
            y_pred,
            labels=labels,
        )

        result = {
            "labels": config.CLASS_NAMES,
            "matrix": matrix.tolist(),
        }

        os.makedirs(
            config.METRICS_DIR,
            exist_ok=True,
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=config.CLASS_NAMES,
            columns=config.CLASS_NAMES,
        )

        matrix_df.index.name = "Actual"

        matrix_df.to_csv(cm_path)

        _baseline_confusion_cache[model_key] = result

        logger.info(
            "Generated and saved confusion matrix for %s",
            model_key,
        )

        return jsonify(result)

    except FileNotFoundError as exc:

        logger.exception(
            "Required artifact not found for confusion matrix."
        )

        return jsonify({
            "error": str(exc)
        }), 404

    except Exception as exc:

        logger.exception(
            "Could not generate confusion matrix for %s",
            model_key,
        )

        return jsonify({
            "error": f"Could not generate confusion matrix: {str(exc)}"
        }), 500


# ---------------------------------------------------------------------------
# FEATURE NAMES
# ---------------------------------------------------------------------------

@app.route("/api/feature-names")
def api_feature_names():
    """Return the selected features used by the prediction model."""

    return jsonify(get_selected_features())


# ---------------------------------------------------------------------------
# DYNAMIC GRAPH EXPLORER API
# ---------------------------------------------------------------------------

@app.route("/api/graphs")
def api_graph_snapshots():
    """List available graph snapshots."""

    snapshots = []

    for split in ["train", "validation", "test"]:

        pattern = os.path.join(
            config.GRAPHS_DIR,
            f"{split}_graph_*.pt",
        )

        files = sorted(glob.glob(pattern))

        for file_path in files:

            filename = os.path.basename(file_path)

            index = int(
                filename.replace(
                    f"{split}_graph_", ""
                ).replace(".pt", "")
            )

            snapshots.append({
                "split": split,
                "index": index,
                "name": f"{split.capitalize()} Graph {index + 1}",
                "filename": filename,
            })

    return jsonify({
        "total_snapshots": len(snapshots),
        "snapshots": snapshots,
    })


@app.route("/api/graphs/<split>/<int:index>")
def api_graph_detail(split, index):
    """
    Return a limited representation of a graph snapshot.

    The API returns at most 180 nodes for visualization.
    """

    if split not in ["train", "validation", "test"]:
        return jsonify({
            "error": "Invalid dataset split."
        }), 400

    filename = f"{split}_graph_{index:04d}.pt"

    graph_path = os.path.join(
        config.GRAPHS_DIR,
        filename,
    )

    if not os.path.isfile(graph_path):
        return jsonify({
            "error": f"Graph snapshot not found: {filename}"
        }), 404

    try:

        graph = torch.load(
            graph_path,
            map_location="cpu",
            weights_only=False,
        )

        max_nodes = 180

        total_nodes = int(graph.x.shape[0])
        total_edges = int(graph.edge_index.shape[1])

        visible_nodes = min(
            total_nodes,
            max_nodes,
        )

        x = graph.x.detach().cpu().numpy()
        y = graph.y.detach().cpu().numpy()
        edge_index = graph.edge_index.detach().cpu().numpy()

        edge_weights = None

        if getattr(graph, "edge_attr", None) is not None:
            edge_weights = (
                graph.edge_attr
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        try:
            feature_names = get_selected_features()
        except Exception:
            feature_names = []

        nodes = []

        for node_id in range(visible_nodes):

            label_id = int(y[node_id])

            feature_values = {}

            for feature_index in range(
                min(len(x[node_id]), 8)
            ):

                if feature_index < len(feature_names):
                    feature_name = feature_names[feature_index]
                else:
                    feature_name = f"Feature {feature_index + 1}"

                feature_values[feature_name] = round(
                    float(x[node_id][feature_index]),
                    5,
                )

            nodes.append({
                "id": node_id,
                "class_id": label_id,
                "class_name": config.INDEX_TO_LABEL.get(
                    label_id,
                    f"Class {label_id}",
                ),
                "features": feature_values,
            })

        edges = []
        seen_edges = set()

        for edge_position in range(edge_index.shape[1]):

            source = int(edge_index[0][edge_position])
            target = int(edge_index[1][edge_position])

            if source >= visible_nodes or target >= visible_nodes:
                continue

            edge_key = (
                min(source, target),
                max(source, target),
            )

            if source == target or edge_key in seen_edges:
                continue

            seen_edges.add(edge_key)

            weight = (
                float(edge_weights[edge_position])
                if edge_weights is not None
                else None
            )

            edges.append({
                "source": edge_key[0],
                "target": edge_key[1],
                "weight": (
                    round(weight, 5)
                    if weight is not None
                    else None
                ),
            })

        return jsonify({
            "split": split,
            "snapshot_index": index,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "visible_nodes": visible_nodes,
            "visible_edges": len(edges),
            "truncated": total_nodes > visible_nodes,
            "nodes": nodes,
            "edges": edges,
        })

    except Exception as exc:

        logger.exception("Could not load graph snapshot.")

        return jsonify({
            "error": f"Could not load graph snapshot: {str(exc)}"
        }), 500


# ---------------------------------------------------------------------------
# FIGURE SERVING
# ---------------------------------------------------------------------------

@app.route("/figures/<path:filename>")
def serve_figure(filename):
    """Serve saved plots and figures."""

    return send_from_directory(
        config.FIGURES_DIR,
        filename,
    )


# ---------------------------------------------------------------------------
# SAMPLE FLOW API
# ---------------------------------------------------------------------------

@app.route("/api/sample-flow")
def api_sample_flow():
    """
    Return one real network-flow record from the processed test dataset.

    The actual category is included so Prediction Lab can compare
    the model prediction against the dataset label.
    """

    try:
        selected_features = get_selected_features()

        required_columns = selected_features + ["Label"]

        if not os.path.isfile(config.TEST_CSV_PATH):
            return jsonify({
                "error": "Processed test dataset was not found."
            }), 404

        test_df = pd.read_csv(
            config.TEST_CSV_PATH,
            usecols=required_columns
        )

        if test_df.empty:
            return jsonify({
                "error": "The processed test dataset is empty."
            }), 404

        # Deterministic sample, so the same record can be reproduced.
        sample = test_df.sample(
            n=1,
            random_state=42
        ).iloc[0]

        features = {
            feature: float(sample[feature])
            for feature in selected_features
        }

        actual_label = int(sample["Label"])

        actual_class = config.INDEX_TO_LABEL.get(
            actual_label,
            f"Class {actual_label}"
        )

        return jsonify({
            "features": features,
            "actual_class": actual_class,
            "source": "Processed CICIoT2023 test dataset"
        })

    except Exception as exc:
        logger.exception("Could not load sample network flow.")

        return jsonify({
            "error": f"Could not load sample flow: {str(exc)}"
        }), 500

# ---------------------------------------------------------------------------
# SINGLE-FLOW PREDICTION
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the attack category for one network flow.

    Expected JSON:

    {
        "model": "RandomForest",
        "features": {
            "Header_Length": 0.5,
            "Protocol Type": 6,
            "Time_To_Live": 64
        }
    }

    Missing features default to 0.0, matching the existing API contract.

    Returns:
        predicted_class
        model_used
        confidence
        probabilities (when supported by the model)
    """

    try:

        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Request body must contain valid JSON."
            }), 400

        model_key = payload.get(
            "model",
            "RandomForest",
        )

        raw_features = payload.get(
            "features",
            {},
        )

        if model_key not in BASELINE_MODEL_FILES:
            return jsonify({
                "error": (
                    f"Invalid model '{model_key}'. "
                    f"Available models: {list(BASELINE_MODEL_FILES.keys())}"
                )
            }), 400

        if not isinstance(raw_features, dict):
            return jsonify({
                "error": "'features' must be a JSON object."
            }), 400

        selected_features = get_selected_features()

        if not selected_features:
            return jsonify({
                "error": "No selected features are configured."
            }), 500

        # Construct feature vector in the exact selected-feature order.
        feature_values = []

        for feature_name in selected_features:

            raw_value = raw_features.get(
                feature_name,
                0.0,
            )

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return jsonify({
                    "error": (
                        f"Invalid numeric value for feature "
                        f"'{feature_name}'."
                    )
                }), 400

            if not np.isfinite(value):
                return jsonify({
                    "error": (
                        f"Feature '{feature_name}' must contain "
                        "a finite numeric value."
                    )
                }), 400

            feature_values.append(value)

        X = np.asarray(
            feature_values,
            dtype=np.float32,
        ).reshape(1, -1)

        # The baseline models were trained on the processed feature data.
        # Do not apply StandardScaler again to these already-processed inputs.
        X_model = X

        model = get_baseline_model(model_key)
        predicted_label = int(model.predict(X_model)[0])

        if predicted_label not in config.INDEX_TO_LABEL:
            raise ValueError(
                f"Unknown predicted class ID: {predicted_label}"
            )

        predicted_class = config.INDEX_TO_LABEL[predicted_label]

        response = {
            "predicted_class": predicted_class,
            "model_used": model_key,
            "feature_count": len(selected_features),
        }

        # Return class probabilities when supported.
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X_model)[0]

            class_probabilities = {}

            model_classes = getattr(
                model,
                "classes_",
                range(len(probabilities)),
            )

            for class_id, probability in zip(
                model_classes,
                probabilities,
            ):

                class_id = int(class_id)

                class_name = config.INDEX_TO_LABEL.get(
                    class_id,
                    f"Class {class_id}",
                )

                class_probabilities[class_name] = float(probability)

            response["probabilities"] = class_probabilities

            response["confidence"] = float(
                max(probabilities)
            )

        logger.info(
            "Prediction completed: model=%s, class=%s",
            model_key,
            predicted_class,
        )

        return jsonify(response), 200

    except FileNotFoundError as exc:

        logger.exception("Prediction artifact not found.")

        return jsonify({
            "error": str(exc)
        }), 404

    except ValueError as exc:

        logger.exception("Invalid prediction input or configuration.")

        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:

        logger.exception("Prediction failed.")

        return jsonify({
            "error": f"Prediction failed: {str(exc)}"
        }), 500


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )
