"""
train.py

Shared training loop for all three GNN architectures (GCN, GraphSAGE,
GAT). Since every model implements the same forward(x, edge_index,
edge_weight) interface, a single training routine can train any of them by
name.

Usage
-----
    python -m src.training.train --model gcn
    python -m src.training.train --model graphsage
    python -m src.training.train --model gat
    python -m src.training.train --model all      # trains all three in sequence
"""

import os
import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from src import config
from src.graph.dynamic_graph import IoTGraphDataset
from src.data.feature_selection import get_selected_features
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "gcn": GCN,
    "graphsage": GraphSAGE,
    "gat": GAT,
}


# ---------------------------------------------------------------------------
# 1. MODEL FACTORY
# ---------------------------------------------------------------------------

def build_model(model_name: str, in_channels: int) -> nn.Module:
    """
    Instantiate a GNN model by name using the shared hyperparameters from
    config.py. Raises a clear error for unknown model names.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_name '{model_name}'. Must be one of "
            f"{list(MODEL_REGISTRY.keys())}."
        )

    model_class = MODEL_REGISTRY[model_name]
    model = model_class(
        in_channels=in_channels,
        hidden_channels=config.HIDDEN_CHANNELS,
        out_channels=config.NUM_CLASSES,
        num_layers=config.NUM_GNN_LAYERS,
        dropout=config.DROPOUT,
    )
    return model.to(config.DEVICE)


# ---------------------------------------------------------------------------
# 2. CLASS WEIGHTS (extra imbalance correction on top of row-level balancing)
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: IoTGraphDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights across every node label in the
    training dataset, for use in a weighted CrossEntropyLoss. This provides
    an additional safeguard against residual imbalance at the graph level
    (individual windows may not have perfectly even class distributions
    even though preprocessing.py balanced the overall dataset).

    Returns
    -------
    torch.Tensor, shape (config.NUM_CLASSES,)
        Weight per class, normalized so weights sum to config.NUM_CLASSES.
    """
    all_labels = []
    for graph in dataset:
        all_labels.append(graph.y.numpy())
    all_labels = np.concatenate(all_labels)

    counts = np.bincount(all_labels, minlength=config.NUM_CLASSES).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid division by zero for any absent class

    inverse_freq = 1.0 / counts
    weights = inverse_freq / inverse_freq.sum() * config.NUM_CLASSES

    logger.info(f"Computed class weights: {dict(zip(config.CLASS_NAMES, weights.round(3)))}")
    return torch.tensor(weights, dtype=torch.float, device=config.DEVICE)


# ---------------------------------------------------------------------------
# 3. TRAIN / EVALUATE ONE EPOCH
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple:
    """Run one full training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(logits, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    accuracy = accuracy_score(all_targets, all_preds)

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    """Run evaluation (no gradient updates). Returns (avg_loss, accuracy, macro_f1)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(logits, batch.y)

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    accuracy = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1


# ---------------------------------------------------------------------------
# 4. FULL TRAINING ROUTINE (with early stopping + checkpointing)
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    num_epochs: int = config.NUM_EPOCHS,
    patience: int = config.EARLY_STOPPING_PATIENCE,
    batch_size: int = config.BATCH_SIZE,
    lr: float = config.LEARNING_RATE,
    weight_decay: float = config.WEIGHT_DECAY,
    use_class_weights: bool = True,
) -> pd.DataFrame:
    """
    Train a single GNN model end-to-end: load graph datasets, build the
    model, train with early stopping on validation macro-F1, save the best
    checkpoint, and return the full training history.

    Parameters
    ----------
    model_name : str
        One of "gcn", "graphsage", "gat".
    num_epochs, patience, batch_size, lr, weight_decay : see config.py defaults.
    use_class_weights : bool
        Whether to weight the loss by inverse class frequency (see
        compute_class_weights()).

    Returns
    -------
    pd.DataFrame
        One row per epoch with train/val loss, accuracy, and macro-F1.
        Also saved to results/metrics/{model_name}_training_history.csv.
    """
    logger.info(f"===== Training model: {model_name.upper()} =====")

    # ---- Load datasets ----
    train_dataset = IoTGraphDataset(split_name="train")
    val_dataset = IoTGraphDataset(split_name="validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---- Build model ----
    in_channels = len(get_selected_features())
    model = build_model(model_name, in_channels)
    logger.info(f"Model architecture:\n{model}")

    # ---- Loss, optimizer ----
    if use_class_weights:
        class_weights = compute_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- Training loop with early stopping ----
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}_model.pt")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, config.DEVICE
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
            }
        )

        logger.info(
            f"[{model_name}] Epoch {epoch:03d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                f"  -> New best val_macro_f1={val_f1:.4f}, saved checkpoint to "
                f"{checkpoint_path}"
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"({patience} epochs without improvement)."
            )
            break

    history_df = pd.DataFrame(history)
    history_path = os.path.join(
        config.METRICS_DIR, f"{model_name}_training_history.csv"
    )
    history_df.to_csv(history_path, index=False)
    logger.info(f"Saved training history to: {history_path}")
    logger.info(f"Best validation macro-F1 for {model_name}: {best_val_f1:.4f}")

    return history_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GNN model(s) for IoT intrusion detection.")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["gcn", "graphsage", "gat", "all"],
        help="Which model to train ('all' trains gcn, graphsage, and gat in sequence).",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    models_to_train = (
        ["gcn", "graphsage", "gat"] if args.model == "all" else [args.model]
    )

    for model_name in models_to_train:
        train_model(
            model_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
