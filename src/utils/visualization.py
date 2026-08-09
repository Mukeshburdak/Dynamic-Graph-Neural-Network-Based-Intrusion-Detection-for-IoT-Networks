"""
visualization.py

Reusable plotting functions used across notebooks, evaluate.py, and the
dashboard app. Every function saves its figure to results/figures/ (or a
custom path) and also returns the matplotlib Figure object in case the
caller wants to display it inline (e.g. in a Jupyter notebook) instead of
just saving to disk.
"""

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from torch_geometric.utils import to_networkx

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# 1. CLASS DISTRIBUTION
# ---------------------------------------------------------------------------

def plot_class_distribution(
    labels: pd.Series,
    save_path: str = None,
    title: str = "Class Distribution",
):
    """
    Bar chart of class counts. Accepts either the string 'Category' column
    or the integer 'Label' column (values are shown as-is on the x-axis).

    Parameters
    ----------
    labels : pd.Series
        Column of class labels (string categories or integer indices).
    save_path : str, optional
        Defaults to results/figures/class_distribution.png.
    """
    if save_path is None:
        save_path = os.path.join(config.FIGURES_DIR, "class_distribution.png")

    counts = labels.value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="viridis")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved class distribution plot to: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 2. CONFUSION MATRIX
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    save_path: str = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
):
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
    class_names : list, optional
        Defaults to config.CLASS_NAMES.
    save_path : str, optional
        Defaults to results/figures/confusion_matrix.png.
    normalize : bool
        If True, rows are normalized to show percentages instead of raw
        counts -- useful when class sizes differ a lot.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    if save_path is None:
        save_path = os.path.join(config.FIGURES_DIR, "confusion_matrix.png")

    cm_display = cm.astype(np.float64)
    fmt = "d"
    if normalize:
        row_sums = cm_display.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_display = cm_display / row_sums
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved confusion matrix plot to: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 3. ROC CURVES (multiclass, one-vs-rest)
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list = None,
    save_path: str = None,
    title: str = "ROC Curves (One-vs-Rest)",
):
    """
    Plot one-vs-rest ROC curves for every class on a single figure.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_samples,)
        Integer class labels.
    y_proba : np.ndarray, shape (n_samples, n_classes)
        Predicted class probabilities (e.g. softmax output).
    class_names : list, optional
        Defaults to config.CLASS_NAMES.
    save_path : str, optional
        Defaults to results/figures/roc_curve.png.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    if save_path is None:
        save_path = os.path.join(config.FIGURES_DIR, "roc_curve.png")

    n_classes = len(class_names)
    y_true_binarized = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        # Skip classes with no positive samples in y_true (ROC undefined).
        if y_true_binarized[:, i].sum() == 0:
            logger.warning(f"Skipping ROC curve for '{class_name}': no positive samples.")
            continue

        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved ROC curve plot to: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 4. TRAINING CURVES (loss / accuracy / F1 over epochs)
# ---------------------------------------------------------------------------

def plot_training_curves(
    history_df: pd.DataFrame,
    model_name: str = "",
    save_path: str = None,
):
    """
    Plot training/validation loss, accuracy, and validation macro-F1 over
    epochs, from a history DataFrame produced by
    src/training/train.py::train_model().

    Parameters
    ----------
    history_df : pd.DataFrame
        Must contain columns: epoch, train_loss, val_loss, train_accuracy,
        val_accuracy, val_macro_f1.
    model_name : str
        Used in the plot title and default filename.
    save_path : str, optional
        Defaults to results/figures/{model_name}_training_curves.png.
    """
    if save_path is None:
        suffix = f"{model_name}_" if model_name else ""
        save_path = os.path.join(config.FIGURES_DIR, f"{suffix}training_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], label="Train Acc")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    axes[2].plot(history_df["epoch"], history_df["val_macro_f1"], label="Val Macro-F1", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro F1")
    axes[2].set_title("Validation Macro-F1")
    axes[2].legend()

    fig.suptitle(f"Training Curves{' - ' + model_name.upper() if model_name else ''}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved training curves plot to: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# 5. GRAPH SNAPSHOT VISUALIZATION
# ---------------------------------------------------------------------------

def plot_graph_snapshot(
    graph_data,
    save_path: str = None,
    max_nodes: int = 150,
    title: str = "Dynamic Graph Snapshot",
):
    """
    Visualize a single k-NN similarity graph snapshot (a torch_geometric
    Data object, e.g. loaded from data/graphs/train_graph_0000.pt), with
    nodes colored by their true class label. Useful for a report/thesis
    figure showing what the "dynamic graph" actually looks like.

    Parameters
    ----------
    graph_data : torch_geometric.data.Data
        A single graph snapshot with .x, .edge_index, and .y attributes.
    save_path : str, optional
        Defaults to results/figures/graph_visualization.png.
    max_nodes : int
        If the graph has more nodes than this, a random subset is
        visualized instead (full similarity graphs can be too dense to
        read visually).
    """
    if save_path is None:
        save_path = os.path.join(config.FIGURES_DIR, "graph_visualization.png")

    nx_graph = to_networkx(graph_data, to_undirected=True)
    labels = graph_data.y.cpu().numpy()

    if nx_graph.number_of_nodes() > max_nodes:
        rng = np.random.default_rng(config.RANDOM_SEED)
        keep_nodes = rng.choice(
            nx_graph.number_of_nodes(), size=max_nodes, replace=False
        )
        nx_graph = nx_graph.subgraph(keep_nodes).copy()
        labels = labels[keep_nodes]
        node_list = list(nx_graph.nodes())
    else:
        node_list = list(nx_graph.nodes())

    # Assign a distinct color per class present in this subgraph.
    palette = sns.color_palette("tab10", n_colors=config.NUM_CLASSES)
    node_colors = [palette[labels[i]] for i in range(len(node_list))]

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(nx_graph, seed=config.RANDOM_SEED)
    nx.draw_networkx_nodes(
        nx_graph, pos, nodelist=node_list, node_color=node_colors, node_size=60, ax=ax
    )
    nx.draw_networkx_edges(nx_graph, pos, alpha=0.2, ax=ax)

    # Build a legend mapping color -> class name for classes present here.
    present_classes = sorted(set(labels.tolist()))
    legend_handles = [
        plt.Line2D(
            [0], [0],
            marker="o", color="w",
            markerfacecolor=palette[c], markersize=10,
            label=config.CLASS_NAMES[c],
        )
        for c in present_classes
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"Saved graph snapshot visualization to: {save_path}")
    return fig
