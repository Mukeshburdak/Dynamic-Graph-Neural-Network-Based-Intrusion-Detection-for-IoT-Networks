"""
gcn.py

Graph Convolutional Network (GCN) for node-level IoT intrusion
classification. Each node in a graph snapshot is one network flow record;
the model predicts one of config.NUM_CLASSES (Benign + 7 attack categories)
for every node, using feature information aggregated from its k
similarity-based neighbors (see src/graph/graph_builder.py).

Reference architecture: Kipf & Welling, "Semi-Supervised Classification
with Graph Convolutional Networks" (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from src import config


class GCN(nn.Module):
    """
    Stacked GCNConv layers with ReLU activation, dropout, and a final
    linear-equivalent GCNConv layer that outputs raw class logits per node
    (no softmax applied here -- use nn.CrossEntropyLoss during training,
    which applies log-softmax internally).

    Parameters
    ----------
    in_channels : int
        Number of input node features (length of config.FEATURE_COLUMNS,
        or the reduced set from feature_selection.py).
    hidden_channels : int
        Size of the hidden node embeddings between layers.
    out_channels : int
        Number of output classes (config.NUM_CLASSES).
    num_layers : int
        Total number of GCNConv layers (including the input and output
        layers). Must be >= 2.
    dropout : float
        Dropout probability applied after each hidden layer's activation.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = config.HIDDEN_CHANNELS,
        out_channels: int = config.NUM_CLASSES,
        num_layers: int = config.NUM_GNN_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError(
                f"GCN requires at least 2 layers (an input layer and an "
                f"output layer), got num_layers={num_layers}."
            )

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # Input layer: in_channels -> hidden_channels
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # Hidden layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Output layer: hidden_channels -> out_channels (raw logits)
        self.convs.append(GCNConv(hidden_channels, out_channels))

        # Batch normalization stabilizes training across the hidden layers
        # (not applied to the final output layer, since that produces
        # logits directly).
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (num_nodes, in_channels)
            Node feature matrix.
        edge_index : torch.Tensor, shape (2, num_edges)
            Graph connectivity in COO format.
        edge_weight : torch.Tensor, shape (num_edges,), optional
            Optional per-edge similarity weight (from graph_builder's
            edge_attr). GCNConv uses this to weight neighbor contributions.

        Returns
        -------
        torch.Tensor, shape (num_nodes, out_channels)
            Raw class logits for every node.
        """
        # edge_attr from graph_builder.py has shape (num_edges, 1);
        # GCNConv expects a 1-D edge_weight tensor of shape (num_edges,).
        if edge_weight is not None and edge_weight.dim() > 1:
            edge_weight = edge_weight.squeeze(-1)

        # Pass through every layer except the last with activation + dropout.
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_weight=edge_weight)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer: no activation, no dropout -- raw logits for
        # CrossEntropyLoss.
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        return x


# ---------------------------------------------------------------------------
# Manual smoke test: `python -m src.models.gcn` from the project root.
# Builds a tiny random graph and verifies the forward pass produces the
# expected output shape, without needing the real dataset.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    num_nodes = 20
    in_channels = config.NUM_FEATURES
    out_channels = config.NUM_CLASSES

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 80))
    edge_weight = torch.rand(80)

    model = GCN(in_channels=in_channels, out_channels=out_channels)
    logits = model(x, edge_index, edge_weight)

    print(model)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}  (expected: ({num_nodes}, {out_channels}))")
    assert logits.shape == (num_nodes, out_channels), "Output shape mismatch!"
    print("Smoke test passed.")
