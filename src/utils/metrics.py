"""
metrics.py

Centralized, reusable classification metric utilities. While
evaluate.py and baseline_ml.py already compute their own core metrics
inline (accuracy, macro P/R/F1) to keep those modules self-contained, this
module provides additional reusable helpers for deeper analysis -- per-class
metric tables, multiclass ROC-AUC, and JSON export -- meant to be used from
notebooks, the dashboard app, or any future analysis script without
duplicating logic.
"""

import json
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. FULL METRIC BUNDLE
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: list = None,
) -> dict:
    """
    Compute a full bundle of classification metrics for a set of
    predictions, optionally including multiclass ROC-AUC if predicted
    class probabilities are supplied.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Ground-truth integer class labels.
    y_pred : np.ndarray, shape (n_samples,)
        Predicted integer class labels.
    y_proba : np.ndarray, shape (n_samples, n_classes), optional
        Predicted class probabilities (e.g. softmax output). Required for
        ROC-AUC; if omitted, "roc_auc_macro" and "roc_auc_per_class" will
        be None in the returned dict.
    class_names : list, optional
        Human-readable class names, defaults to config.CLASS_NAMES.

    Returns
    -------
    dict with keys:
        accuracy, macro_precision, macro_recall, macro_f1,
        weighted_precision, weighted_recall, weighted_f1,
        roc_auc_macro, roc_auc_per_class (dict or None),
        confusion_matrix (np.ndarray)
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    accuracy = accuracy_score(y_true, y_pred)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    roc_auc_macro = None
    roc_auc_per_class = None

    if y_proba is not None:
        try:
            roc_auc_macro = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
            per_class_scores = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average=None
            )
            roc_auc_per_class = dict(zip(class_names, per_class_scores))
        except ValueError as exc:
            # Can happen if a class is entirely absent from y_true in a
            # given evaluation batch; ROC-AUC is undefined for that class.
            logger.warning(f"Could not compute ROC-AUC: {exc}")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": accuracy,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_per_class": roc_auc_per_class,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# 2. PER-CLASS METRIC TABLE
# ---------------------------------------------------------------------------

def get_per_class_metrics_df(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list = None
) -> pd.DataFrame:
    """
    Build a per-class precision/recall/F1/support table as a DataFrame,
    convenient for display in notebooks or the dashboard.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    ).sort_values("f1_score", ascending=True)


# ---------------------------------------------------------------------------
# 3. JSON EXPORT (for the dashboard to consume without re-running models)
# ---------------------------------------------------------------------------

def _json_safe(obj):
    """Recursively convert numpy types to native Python types for json.dump."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def save_metrics_json(metrics: dict, path: str):
    """Save a metrics dict (from compute_classification_metrics) to JSON."""
    with open(path, "w") as f:
        json.dump(_json_safe(metrics), f, indent=2)
    logger.info(f"Saved metrics JSON to: {path}")


def load_metrics_json(path: str) -> dict:
    """Load a previously saved metrics JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 4. MODEL COMPARISON HELPER
# ---------------------------------------------------------------------------

def summarize_comparison(comparison_df: pd.DataFrame) -> str:
    """
    Return a short human-readable summary string highlighting the
    best-performing model in a comparison table (such as
    results/comparison.csv), useful for printing in notebooks/CLI output.
    """
    if comparison_df.empty:
        return "Comparison table is empty."

    best_row = comparison_df.sort_values("macro_f1", ascending=False).iloc[0]
    return (
        f"Best model: {best_row['model_name']} "
        f"(accuracy={best_row['accuracy']:.4f}, "
        f"macro_f1={best_row['macro_f1']:.4f})"
    )
