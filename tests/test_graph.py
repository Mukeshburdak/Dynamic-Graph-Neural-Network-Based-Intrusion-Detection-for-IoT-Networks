"""
tests/test_graph.py

Unit tests for src/graph/graph_builder.py and src/graph/dynamic_graph.py.
Uses small synthetic feature/label arrays instead of real flow data.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.graph.graph_builder import build_knn_graph
from src.graph.dynamic_graph import generate_graph_snapshots


# ---------------------------------------------------------------------------
# build_knn_graph()
# ---------------------------------------------------------------------------

def test_build_knn_graph_output_shapes():
    num_nodes = 20
    num_features = 6
    k = 5

    rng = np.random.default_rng(config.RANDOM_SEED)
    features = rng.normal(size=(num_nodes, num_features)).astype(np.float32)
    labels = rng.integers(0, config.NUM_CLASSES, size=num_nodes).astype(np.int64)

    graph = build_knn_graph(features, labels, k=k)

    assert graph.x.shape == (num_nodes, num_features)
    assert graph.y.shape == (num_nodes,)
    # Each node connects to k neighbors, and edges are added in BOTH
    # directions -> num_nodes * k * 2 total directed edges.
    assert graph.edge_index.shape == (2, num_nodes * k * 2)
    assert graph.edge_attr.shape == (num_nodes * k * 2, 1)


def test_build_knn_graph_reduces_k_for_small_window():
    num_nodes = 3
    features = np.random.randn(num_nodes, 4).astype(np.float32)
    labels = np.array([0, 1, 0], dtype=np.int64)

    # Requesting k=8 neighbors with only 3 nodes should silently reduce
    # effective_k to (num_nodes - 1) = 2 instead of raising an error.
    graph = build_knn_graph(features, labels, k=8)

    effective_k = num_nodes - 1
    assert graph.edge_index.shape == (2, num_nodes * effective_k * 2)


def test_build_knn_graph_raises_on_single_node():
    features = np.random.randn(1, 4).astype(np.float32)
    labels = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError):
        build_knn_graph(features, labels, k=5)


def test_build_knn_graph_similarity_weights_are_positive():
    num_nodes = 10
    features = np.random.randn(num_nodes, 4).astype(np.float32)
    labels = np.zeros(num_nodes, dtype=np.int64)

    graph = build_knn_graph(features, labels, k=3)

    # Similarity = 1 / (1 + distance); distance >= 0, so this must always
    # be in the (0, 1] range.
    assert (graph.edge_attr > 0).all()
    assert (graph.edge_attr <= 1).all()


# ---------------------------------------------------------------------------
# generate_graph_snapshots()
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_flow_df():
    feature_columns = config.FEATURE_COLUMNS[:4]
    n_rows = 55  # deliberately not an exact multiple of window_size

    rng = np.random.default_rng(config.RANDOM_SEED)
    data = {col: rng.normal(size=n_rows) for col in feature_columns}
    data["Label"] = rng.integers(0, config.NUM_CLASSES, size=n_rows)

    return pd.DataFrame(data), feature_columns


def test_generate_graph_snapshots_window_count(synthetic_flow_df):
    df, feature_columns = synthetic_flow_df

    # window_size=20, stride=20 over 55 rows -> windows at [0:20], [20:40],
    # [40:55] (last window smaller but still >= 2 rows, so it's kept).
    graphs = generate_graph_snapshots(
        df,
        feature_columns,
        window_size=20,
        stride=20,
        max_nodes_per_graph=100,  # larger than any window, so no subsampling
        k_neighbors=3,
    )

    assert len(graphs) == 3
    assert graphs[0].x.shape[0] == 20
    assert graphs[1].x.shape[0] == 20
    assert graphs[2].x.shape[0] == 15  # remaining rows


def test_generate_graph_snapshots_respects_max_nodes(synthetic_flow_df):
    df, feature_columns = synthetic_flow_df

    graphs = generate_graph_snapshots(
        df,
        feature_columns,
        window_size=50,
        stride=50,
        max_nodes_per_graph=10,  # forces subsampling within each window
        k_neighbors=3,
    )

    for graph in graphs:
        assert graph.x.shape[0] <= 10


def test_generate_graph_snapshots_feature_dimension_matches(synthetic_flow_df):
    df, feature_columns = synthetic_flow_df

    graphs = generate_graph_snapshots(
        df, feature_columns, window_size=20, stride=20, k_neighbors=3
    )

    for graph in graphs:
        assert graph.x.shape[1] == len(feature_columns)
