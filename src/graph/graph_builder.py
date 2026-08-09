"""
graph_builder.py

Core logic for converting a chunk (window) of flow records into a single
PyTorch Geometric graph, since the public CICIoT2023 CSV release contains
no Source/Destination IP addresses (no real network topology available).

Graph construction strategy (similarity graph):
    - Each row (flow record) in the window becomes one graph NODE.
    - Node features = the row's scaled feature vector
      (config.FEATURE_COLUMNS or the reduced set from feature_selection.py).
    - Node label = the row's individual 'Label' (used for node-level
      classification -- the GNN predicts a class for every node using
      information aggregated from its most similar neighboring flows).
    - EDGES connect each node to its K nearest neighbors in feature space
      (Euclidean distance), making the graph undirected. Edge weight is
      stored as a similarity score (1 / (1 + distance)) for optional use by
      attention-based models (GAT).
"""

import logging

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_knn_graph(
    features: np.ndarray,
    labels: np.ndarray,
    k: int = config.K_NEIGHBORS,
) -> Data:
    """
    Build a single k-NN similarity graph from a block of flow records.

    Parameters
    ----------
    features : np.ndarray, shape (num_nodes, num_features)
        Scaled numeric feature matrix for this window/chunk of flows.
    labels : np.ndarray, shape (num_nodes,)
        Integer class label (config.LABEL_TO_INDEX values) for each row,
        used as the per-node target for node classification.
    k : int
        Number of nearest neighbors to connect each node to. Automatically
        reduced if the window has fewer than k+1 rows.

    Returns
    -------
    torch_geometric.data.Data
        A single graph with:
          x          : (num_nodes, num_features) node feature matrix
          edge_index : (2, num_edges) edge list in COO format, undirected
          edge_attr  : (num_edges, 1) similarity weight per edge
          y          : (num_nodes,) integer class label per node
    """
    num_nodes = features.shape[0]

    if num_nodes < 2:
        raise ValueError(
            f"Cannot build a graph with fewer than 2 nodes (got {num_nodes})."
        )

    # k can't exceed (num_nodes - 1), since a node can't be its own neighbor.
    effective_k = min(k, num_nodes - 1)
    if effective_k < k:
        logger.warning(
            f"Window has only {num_nodes} nodes; reducing k from {k} to "
            f"{effective_k} for this graph."
        )

    # +1 because NearestNeighbors counts each point as its own nearest
    # neighbor (distance 0); we drop that self-match below.
    nbrs = NearestNeighbors(n_neighbors=effective_k + 1, algorithm="auto")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Drop the first column (self-match) from both arrays.
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    source_nodes = []
    target_nodes = []
    edge_weights = []

    for node_idx in range(num_nodes):
        for neighbor_pos in range(effective_k):
            neighbor_idx = indices[node_idx, neighbor_pos]
            dist = distances[node_idx, neighbor_pos]
            similarity = 1.0 / (1.0 + dist)  # bounded (0, 1], higher = more similar

            # Add edge in both directions to make the graph undirected,
            # since GCN/GraphSAGE/GAT expect symmetric adjacency for
            # standard message passing.
            source_nodes.extend([node_idx, neighbor_idx])
            target_nodes.extend([neighbor_idx, node_idx])
            edge_weights.extend([similarity, similarity])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

    # torch_geometric.utils.coalesce would also remove duplicate edges from
    # nodes that mutually selected each other as neighbors; done here
    # manually via torch_geometric's built-in Data validation is skipped
    # for speed, duplicate edges are harmless for message passing (they
    # just slightly up-weight that connection).

    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return graph


def graph_summary(graph: Data) -> str:
    """Return a short human-readable summary string for logging/debugging."""
    return (
        f"Graph(nodes={graph.num_nodes}, edges={graph.num_edges}, "
        f"features={graph.num_node_features}, "
        f"classes_present={sorted(set(graph.y.tolist()))})"
    )
