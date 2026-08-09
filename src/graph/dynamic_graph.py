"""
dynamic_graph.py

Builds the full sequence of dynamic graph snapshots for a dataset split
(train/validation/test) by sliding a window over the flow records and
calling graph_builder.build_knn_graph() on each window. The resulting
sequence of graphs represents the "dynamic" aspect of the project: as new
traffic arrives, a new graph snapshot is built, so the model can eventually
be extended to process graphs as an evolving stream rather than one static
graph.

Also provides IoTGraphDataset, a simple PyTorch Dataset over the saved
snapshot files, ready to be used with torch_geometric's DataLoader for
mini-batch training.
"""

import os
import glob
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src import config
from src.graph.graph_builder import build_knn_graph, graph_summary
from src.data.feature_selection import get_selected_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. WINDOW GENERATION
# ---------------------------------------------------------------------------

def _iter_windows(num_rows: int, window_size: int, stride: int):
    """
    Yield (start, end) index pairs sliding over a dataset of length
    num_rows. The final window is dropped if it would be smaller than 2
    rows (too small to build a meaningful graph).
    """
    start = 0
    while start < num_rows:
        end = min(start + window_size, num_rows)
        if end - start >= 2:
            yield start, end
        start += stride


def generate_graph_snapshots(
    df: pd.DataFrame,
    feature_columns: list,
    window_size: int = config.GRAPH_WINDOW_SIZE,
    stride: int = config.GRAPH_WINDOW_STRIDE,
    max_nodes_per_graph: int = config.MAX_NODES_PER_GRAPH,
    k_neighbors: int = config.K_NEIGHBORS,
    random_state: int = config.RANDOM_SEED,
) -> list:
    """
    Slide a window over df and build one k-NN similarity graph per window.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain feature_columns and a 'Label' column (integer class
        index). Typically one of train.csv / validation.csv / test.csv,
        already cleaned, balanced, and scaled by preprocessing.py.
    feature_columns : list
        Column names to use as node features (from feature_selection.py,
        or config.FEATURE_COLUMNS as a fallback).
    window_size : int
        Number of consecutive rows grouped into one graph snapshot.
    stride : int
        Step size between window starts. Equal to window_size means
        non-overlapping windows; smaller creates overlap.
    max_nodes_per_graph : int
        If a window has more rows than this, it is randomly subsampled down
        (keeps k-NN computation and GNN memory bounded).
    k_neighbors : int
        Passed through to graph_builder.build_knn_graph().
    random_state : int
        Seed for reproducible subsampling.

    Returns
    -------
    list of torch_geometric.data.Data
        One graph object per window.
    """
    rng = np.random.default_rng(random_state)
    graphs = []

    num_rows = len(df)
    windows = list(_iter_windows(num_rows, window_size, stride))
    logger.info(f"Building {len(windows)} graph snapshot(s) from {num_rows:,} rows...")

    for window_idx, (start, end) in enumerate(windows):
        window_df = df.iloc[start:end]

        if len(window_df) > max_nodes_per_graph:
            sampled_positions = rng.choice(
                len(window_df), size=max_nodes_per_graph, replace=False
            )
            window_df = window_df.iloc[sampled_positions]

        features = window_df[feature_columns].values.astype(np.float32)
        labels = window_df["Label"].values.astype(np.int64)

        try:
            graph = build_knn_graph(features, labels, k=k_neighbors)
        except ValueError as exc:
            logger.warning(f"Skipping window {window_idx} ({start}:{end}): {exc}")
            continue

        graphs.append(graph)

        if window_idx % 50 == 0:
            logger.info(f"  [{window_idx}/{len(windows)}] {graph_summary(graph)}")

    logger.info(f"Finished building {len(graphs)} graph snapshot(s).")
    return graphs


# ---------------------------------------------------------------------------
# 2. SAVE / LOAD GRAPH SNAPSHOTS
# ---------------------------------------------------------------------------

def save_graph_snapshots(graphs: list, split_name: str, graphs_dir: str = config.GRAPHS_DIR):
    """
    Save each graph in `graphs` to its own .pt file, named
    "{split_name}_graph_0000.pt", "{split_name}_graph_0001.pt", etc.

    Parameters
    ----------
    graphs : list of torch_geometric.data.Data
    split_name : str
        One of "train", "validation", "test" -- used as the filename prefix.
    graphs_dir : str
        Output directory (defaults to config.GRAPHS_DIR).
    """
    os.makedirs(graphs_dir, exist_ok=True)

    for idx, graph in enumerate(graphs):
        filename = f"{split_name}_graph_{idx:04d}.pt"
        filepath = os.path.join(graphs_dir, filename)
        torch.save(graph, filepath)

    logger.info(f"Saved {len(graphs)} '{split_name}' graph snapshot(s) to: {graphs_dir}")


def build_and_save_graphs_for_split(
    csv_path: str,
    split_name: str,
    feature_columns: list = None,
) -> int:
    """
    End-to-end helper: load a processed CSV split, build its graph
    snapshots, and save them to disk.

    Parameters
    ----------
    csv_path : str
        Path to train.csv / validation.csv / test.csv.
    split_name : str
        Filename prefix used when saving ("train", "validation", "test").
    feature_columns : list, optional
        If None, uses feature_selection.get_selected_features() (which
        falls back to the full config.FEATURE_COLUMNS if feature selection
        hasn't been run yet).

    Returns
    -------
    int
        Number of graph snapshots created.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run src/data/preprocessing.py first."
        )

    if feature_columns is None:
        feature_columns = get_selected_features()

    logger.info(f"Loading '{split_name}' split from: {csv_path}")
    df = pd.read_csv(csv_path)

    graphs = generate_graph_snapshots(df, feature_columns)
    save_graph_snapshots(graphs, split_name)

    return len(graphs)


# ---------------------------------------------------------------------------
# 3. PYTORCH DATASET FOR LOADING SAVED GRAPHS
# ---------------------------------------------------------------------------

class IoTGraphDataset(Dataset):
    """
    A simple PyTorch Dataset that lazily loads previously saved graph
    snapshot .pt files for a given split. Compatible with
    torch_geometric.loader.DataLoader, which knows how to batch a list of
    Data objects into a single disjoint-union mini-batch.

    Example
    -------
    >>> from torch_geometric.loader import DataLoader
    >>> train_dataset = IoTGraphDataset(split_name="train")
    >>> train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
    ...                            shuffle=True)
    """

    def __init__(self, split_name: str, graphs_dir: str = config.GRAPHS_DIR):
        self.split_name = split_name
        self.graphs_dir = graphs_dir
        self.file_paths = sorted(
            glob.glob(os.path.join(graphs_dir, f"{split_name}_graph_*.pt"))
        )

        if not self.file_paths:
            raise FileNotFoundError(
                f"No graph snapshot files found for split '{split_name}' in "
                f"{graphs_dir}. Run "
                f"dynamic_graph.build_and_save_graphs_for_split() first."
            )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        # weights_only=False because Data objects are custom classes, not
        # plain tensors; safe here since these files are generated locally
        # by our own pipeline, not downloaded from an untrusted source.
        return torch.load(self.file_paths[idx], weights_only=False)


# ---------------------------------------------------------------------------
# Manual run: `python -m src.graph.dynamic_graph` from the project root.
# Builds and saves graph snapshots for all three splits using whatever
# train/validation/test CSVs currently exist in data/processed/.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    splits = [
        (config.TRAIN_CSV_PATH, "train"),
        (config.VAL_CSV_PATH, "validation"),
        (config.TEST_CSV_PATH, "test"),
    ]

    for csv_path, split_name in splits:
        if os.path.exists(csv_path):
            count = build_and_save_graphs_for_split(csv_path, split_name)
            logger.info(f"'{split_name}': {count} graph snapshot(s) created.\n")
        else:
            logger.warning(f"Skipping '{split_name}': {csv_path} not found.")
