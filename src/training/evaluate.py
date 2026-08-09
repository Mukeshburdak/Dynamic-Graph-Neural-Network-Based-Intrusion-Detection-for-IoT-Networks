"""
evaluate.py

Loads each trained GNN checkpoint (saved by src/training/train.py),
evaluates it on the held-out test graph snapshots, and produces:
    - Per-model classification report (results/metrics/{model}_classification_report.txt)
    - Per-model confusion matrix, both as CSV (results/metrics/) and a
      plotted heatmap (results/figures/confusion_matrix_{model}.png)
    - A single unified comparison table across all 3 GNNs + the classical
      ML baselines from baseline_ml.py, saved to results/comparison.csv

Usage
-----
    python -m src.training.evaluate
"""

import os
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src import config
from src.graph.dynamic_graph import IoTGraphDataset
from src.data.feature_selection import get_selected_features
from src.training.train import build_model, MODEL_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. LOAD A TRAINED CHECKPOINT
# ---------------------------------------------------------------------------

def load_trained_model(model_name: str, in_channels: int) -> torch.nn.Module:
    """
    Rebuild a GNN model's architecture and load its trained weights from
    models/{model_name}_model.pt (saved by train.py during training).
    """
    checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint_path}. Train the model "
            f"first with: python -m src.training.train --model {model_name}"
        )

    model = build_model(model_name, in_channels)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    logger.info(f"Loaded trained '{model_name}' checkpoint from: {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# 2. RUN INFERENCE ON THE TEST SET
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_on_test(model: torch.nn.Module, test_loader: DataLoader):
    """
    Run the model over every batch in test_loader and collect predictions.

    Returns
    -------
    y_true : np.ndarray, shape (num_test_nodes,)
    y_pred : np.ndarray, shape (num_test_nodes,)
    """
    model.eval()
    all_preds, all_targets = [], []

    for batch in test_loader:
        batch = batch.to(config.DEVICE)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    return np.concatenate(all_targets), np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# 3. METRICS COMPUTATION
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute accuracy, macro precision/recall/F1, full per-class
    classification report, and confusion matrix for a set of predictions.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, target_names=config.CLASS_NAMES, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def save_confusion_matrix_plot(cm: np.ndarray, model_name: str):
    """Save a labeled confusion matrix heatmap to results/figures/."""
    save_path = os.path.join(config.FIGURES_DIR, f"confusion_matrix_{model_name}.png")

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name.upper()}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    logger.info(f"Saved confusion matrix plot to: {save_path}")


def save_model_metrics(model_name: str, metrics: dict):
    """
    Save a model's classification report (.txt) and confusion matrix
    (.csv) to results/metrics/, and its confusion matrix heatmap to
    results/figures/.
    """
    report_path = os.path.join(
        config.METRICS_DIR, f"{model_name}_classification_report.txt"
    )
    with open(report_path, "w") as f:
        f.write(metrics["classification_report"])
    logger.info(f"Saved classification report to: {report_path}")

    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=config.CLASS_NAMES,
        columns=config.CLASS_NAMES,
    )
    cm_path = os.path.join(config.METRICS_DIR, f"{model_name}_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    logger.info(f"Saved confusion matrix CSV to: {cm_path}")

    save_confusion_matrix_plot(metrics["confusion_matrix"], model_name)


# ---------------------------------------------------------------------------
# 4. EVALUATE ALL THREE GNN MODELS
# ---------------------------------------------------------------------------

def evaluate_all_gnn_models(batch_size: int = config.BATCH_SIZE) -> pd.DataFrame:
    """
    Evaluate every model in MODEL_REGISTRY (gcn, graphsage, gat) on the test
    graph snapshots. Skips any model whose checkpoint doesn't exist yet
    (with a warning) rather than crashing, so partial runs still work.

    Returns
    -------
    pd.DataFrame
        One row per successfully evaluated GNN model with accuracy/
        precision/recall/F1 columns.
    """
    test_dataset = IoTGraphDataset(split_name="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    in_channels = len(get_selected_features())
    results = []

    for model_name in MODEL_REGISTRY.keys():
        checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}_model.pt")
        if not os.path.exists(checkpoint_path):
            logger.warning(
                f"Skipping '{model_name}': no checkpoint found at "
                f"{checkpoint_path}. Train it first."
            )
            continue

        model = load_trained_model(model_name, in_channels).to(config.DEVICE)
        y_true, y_pred = predict_on_test(model, test_loader)
        metrics = compute_metrics(y_true, y_pred)

        logger.info(
            f"[TEST] {model_name} -> accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )

        save_model_metrics(model_name, metrics)

        results.append(
            {
                "model_name": model_name.upper(),
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. FINAL UNIFIED COMPARISON TABLE (GNNs + baselines)
# ---------------------------------------------------------------------------

def build_final_comparison(gnn_results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge GNN evaluation results with the classical ML baseline results
    (from baseline_ml.run_baseline_comparison(), if it has already been
    run) into one final comparison table, saved to results/comparison.csv.
    """
    baseline_path = os.path.join(config.RESULTS_DIR, "baseline_comparison.csv")

    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        combined_df = pd.concat([gnn_results_df, baseline_df], ignore_index=True)
        logger.info("Merged GNN results with existing baseline_comparison.csv.")
    else:
        combined_df = gnn_results_df
        logger.warning(
            f"{baseline_path} not found. Comparison table will only "
            f"include GNN models. Run "
            f"`python -m src.models.baseline_ml` to include ML baselines."
        )

    combined_df = combined_df.sort_values("macro_f1", ascending=False).reset_index(
        drop=True
    )

    out_path = os.path.join(config.RESULTS_DIR, "comparison.csv")
    combined_df.to_csv(out_path, index=False)
    logger.info(f"Saved final comparison table to: {out_path}")
    logger.info(f"\n{combined_df.to_string(index=False)}")

    return combined_df


# ---------------------------------------------------------------------------
# Manual run: `python -m src.training.evaluate` from the project root.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gnn_results = evaluate_all_gnn_models()
    if gnn_results.empty:
        logger.warning(
            "No GNN models were evaluated (no checkpoints found). "
            "Train at least one model first."
        )
    else:
        build_final_comparison(gnn_results)
