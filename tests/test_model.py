"""
tests/test_model.py

Unit tests for src/models/gcn.py, graphsage.py, and gat.py. Uses randomly
generated tensors (not the real dataset) purely to validate that each
model's forward pass produces the expected output shape and that
construction-time validation (e.g. num_layers >= 2) works correctly.
"""

import torch
import pytest

from src import config
from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gat import GAT


# ---------------------------------------------------------------------------
# Shared synthetic graph fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_graph():
    """A small random graph: 15 nodes, 6 input features, 60 directed edges."""
    torch.manual_seed(config.RANDOM_SEED)
    num_nodes = 15
    in_channels = 6
    num_edges = 60

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weight = torch.rand(num_edges)

    return x, edge_index, edge_weight, num_nodes, in_channels


# ---------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------

def test_gcn_forward_output_shape(synthetic_graph):
    x, edge_index, edge_weight, num_nodes, in_channels = synthetic_graph

    model = GCN(in_channels=in_channels, hidden_channels=8, out_channels=config.NUM_CLASSES, num_layers=3)
    logits = model(x, edge_index, edge_weight)

    assert logits.shape == (num_nodes, config.NUM_CLASSES)


def test_gcn_forward_without_edge_weight(synthetic_graph):
    """edge_weight is optional -- the model must also work without it."""
    x, edge_index, _, num_nodes, in_channels = synthetic_graph

    model = GCN(in_channels=in_channels, out_channels=config.NUM_CLASSES)
    logits = model(x, edge_index)

    assert logits.shape == (num_nodes, config.NUM_CLASSES)


def test_gcn_rejects_single_layer():
    with pytest.raises(ValueError):
        GCN(in_channels=6, out_channels=config.NUM_CLASSES, num_layers=1)


# ---------------------------------------------------------------------------
# GraphSAGE
# ---------------------------------------------------------------------------

def test_graphsage_forward_output_shape(synthetic_graph):
    x, edge_index, _, num_nodes, in_channels = synthetic_graph

    model = GraphSAGE(in_channels=in_channels, hidden_channels=8, out_channels=config.NUM_CLASSES, num_layers=3)
    logits = model(x, edge_index)

    assert logits.shape == (num_nodes, config.NUM_CLASSES)


def test_graphsage_rejects_single_layer():
    with pytest.raises(ValueError):
        GraphSAGE(in_channels=6, out_channels=config.NUM_CLASSES, num_layers=1)


# ---------------------------------------------------------------------------
# GAT
# ---------------------------------------------------------------------------

def test_gat_forward_output_shape(synthetic_graph):
    x, edge_index, edge_weight, num_nodes, in_channels = synthetic_graph

    model = GAT(
        in_channels=in_channels,
        hidden_channels=8,
        out_channels=config.NUM_CLASSES,
        num_layers=3,
        heads=2,
    )
    logits = model(x, edge_index, edge_weight)

    assert logits.shape == (num_nodes, config.NUM_CLASSES)


def test_gat_forward_without_edge_weight(synthetic_graph):
    x, edge_index, _, num_nodes, in_channels = synthetic_graph

    model = GAT(in_channels=in_channels, out_channels=config.NUM_CLASSES, heads=2)
    logits = model(x, edge_index)

    assert logits.shape == (num_nodes, config.NUM_CLASSES)


def test_gat_rejects_single_layer():
    with pytest.raises(ValueError):
        GAT(in_channels=6, out_channels=config.NUM_CLASSES, num_layers=1)


# ---------------------------------------------------------------------------
# Cross-model consistency (all three should be usable interchangeably by
# training/evaluation code, since train.py relies on this)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_class", [GCN, GraphSAGE, GAT])
def test_all_models_share_compatible_forward_signature(synthetic_graph, model_class):
    x, edge_index, edge_weight, num_nodes, in_channels = synthetic_graph

    model = model_class(in_channels=in_channels, out_channels=config.NUM_CLASSES)
    logits = model(x, edge_index, edge_weight)

    assert logits.shape == (num_nodes, config.NUM_CLASSES)
    assert not torch.isnan(logits).any()
