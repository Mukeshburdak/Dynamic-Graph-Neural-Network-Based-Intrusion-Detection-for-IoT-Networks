"""
baseline_ml.py

Classical (non-graph) machine learning baselines trained directly on the
tabular flow features, used as a comparison point against the three GNN
architectures (GCN, GraphSAGE, GAT). This answers the key research
question: "does modeling flows as a similarity graph actually improve
detection over treating each flow independently?"

Baselines included:
    - Random Forest        (robust, standard IDS baseline in literature)
    - XGBoost               (typically the strongest tabular baseline;
                              falls back to sklearn's GradientBoosting if
                              the xgboost package is not installed)
    - Logistic Regression   (simple linear baseline, useful lower bound)

These operate on config.TRAIN_CSV_PATH / VAL_CSV_PATH / TEST_CSV_PATH
directly (the same preprocessed, balanced, scaled data used to build
graphs) -- NOT on the graph snapshot .pt files, since baselines don't use
graph structure.
"""

import os
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src import config
from src.data.feature_selection import get_selected_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Try to use XGBoost if it's installed (generally the strongest tabular
# baseline); otherwise fall back to sklearn's GradientBoostingClassifier so
# the pipeline still runs without an extra dependency.
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.warning(
        "xgboost is not installed; falling back to "
        "sklearn.GradientBoostingClassifier for the 'XGBoost' baseline. "
        "Run `pip install xgboost` for the actual XGBoost implementation."
    )


# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------

def load_split(csv_path: str, feature_columns: list = None):
    """
    Load a processed CSV split and return (X, y) numpy arrays ready for
    scikit-learn / XGBoost.

    Parameters
    ----------
    csv_path : str
        Path to train.csv / validation.csv / test.csv.
    feature_columns : list, optional
        If None, uses feature_selection.get_selected_features().

    Returns
    -------
    X : np.ndarray, shape (num_rows, num_features)
    y : np.ndarray, shape (num_rows,)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run src/data/preprocessing.py first."
        )

    if feature_columns is None:
        feature_columns = get_selected_features()

    df = pd.read_csv(csv_path)
    X = df[feature_columns].values.astype(np.float32)
    y = df["Label"].values.astype(np.int64)
    return X, y


# ---------------------------------------------------------------------------
# 2. MODEL DEFINITIONS
# ---------------------------------------------------------------------------

def get_baseline_models() -> dict:
    """
    Return a dictionary of {name: unfitted sklearn/xgboost estimator} for
    every baseline model to be trained and compared.
    """
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=config.RANDOM_SEED,
        ),
    }

    if _XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="multi:softmax",
            num_class=config.NUM_CLASSES,
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=config.RANDOM_SEED,
        )
    else:
        models["XGBoost"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            random_state=config.RANDOM_SEED,
        )

    return models


# ---------------------------------------------------------------------------
# 3. TRAIN / EVALUATE A SINGLE BASELINE
# ---------------------------------------------------------------------------

def train_and_evaluate_baseline(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_model: bool = True,
) -> dict:
    """
    Fit `model` on the training set, evaluate it on the test set, and
    optionally save the fitted model to disk.

    Returns
    -------
    dict
        Metrics: accuracy, macro precision/recall/F1, and the full
        per-class classification report (as a string) and confusion matrix
        (as a list of lists, for JSON/CSV-friendly storage).
    """
    logger.info(f"Training baseline: {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=config.CLASS_NAMES,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred).tolist()
        # Save the confusion matrix for the React dashboard.
    os.makedirs(config.METRICS_DIR, exist_ok=True)

    labels = list(range(len(config.CLASS_NAMES)))

    cm_array = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm_array,
        index=config.CLASS_NAMES,
        columns=config.CLASS_NAMES
    )

    cm_path = os.path.join(
        config.METRICS_DIR,
        f"{name.lower()}_confusion_matrix.csv"
    )

    cm_df.index.name = "Actual"
    cm_df.to_csv(cm_path)

    logger.info("Saved confusion matrix to: %s", cm_path)

    cm = cm_array.tolist()
    logger.info(
        f"{name} -> accuracy={accuracy:.4f}, macro_precision={precision:.4f}, "
        f"macro_recall={recall:.4f}, macro_f1={f1:.4f}"
    )

    if save_model:
        model_path = os.path.join(config.MODELS_DIR, f"{name.lower()}_baseline.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Saved {name} model to: {model_path}")

    return {
        "model_name": name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# 4. FULL BASELINE COMPARISON PIPELINE
# ---------------------------------------------------------------------------

def run_baseline_comparison(save_results: bool = True) -> pd.DataFrame:
    """
    Train and evaluate every baseline model on the same train/test splits
    used by the GNN pipeline, and return a summary comparison table.

    Returns
    -------
    pd.DataFrame
        One row per baseline model with accuracy/precision/recall/F1
        columns, suitable for direct comparison against GNN results in
        results/comparison.csv.
    """
    feature_columns = get_selected_features()
    logger.info(f"Using {len(feature_columns)} feature(s) for baseline models.")

    logger.info("Loading train/test splits...")
    X_train, y_train = load_split(config.TRAIN_CSV_PATH, feature_columns)
    X_test, y_test = load_split(config.TEST_CSV_PATH, feature_columns)

    models = get_baseline_models()
    results = []

    for name, model in models.items():
        metrics = train_and_evaluate_baseline(
            name, model, X_train, y_train, X_test, y_test
        )
        results.append(metrics)

        # Save the detailed per-class report separately for each model.
        report_path = os.path.join(
            config.METRICS_DIR, f"{name.lower()}_classification_report.txt"
        )
        with open(report_path, "w") as f:
            f.write(metrics["classification_report"])

    comparison_df = pd.DataFrame(
        [
            {
                "model_name": r["model_name"],
                "accuracy": r["accuracy"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "macro_f1": r["macro_f1"],
            }
            for r in results
        ]
    ).sort_values("macro_f1", ascending=False)

    if save_results:
        out_path = os.path.join(config.RESULTS_DIR, "baseline_comparison.csv")
        comparison_df.to_csv(out_path, index=False)
        logger.info(f"Saved baseline comparison table to: {out_path}")

    logger.info(f"\n{comparison_df.to_string(index=False)}")
    return comparison_df


# ---------------------------------------------------------------------------
# Manual run: `python -m src.models.baseline_ml` from the project root.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_baseline_comparison()
