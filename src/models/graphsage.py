"""
graphsage.py

GraphSAGE (Sample and Aggregate) for node-level IoT intrusion
classification. Unlike GCN's spectral-style convolution, GraphSAGE learns
an aggregation function (here: mean aggregation) over each node's
neighbors and concatenates it with the node's own features before
projecting to the next layer. This tends to generalize better to graph
structures not seen during training -- relevant here since every graph
snapshot is built from a different window of traffic with a different
neighbor structure.

Reference architecture: Hamilton, Ying & Leskovec, "Inductive
Representation Learning on Large Graphs" (2017).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src import config


class GraphSAGE(nn.Module):
    """
    Stacked SAGEConv layers with ReLU activation, dropout, and a final
    layer producing raw class logits per node.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Size of the hidden node embeddings between layers.
    out_channels : int
        Number of output classes (config.NUM_CLASSES).
    num_layers : int
        Total number of SAGEConv layers (including input and output).
        Must be >= 2.
    dropout : float
        Dropout probability applied after each hidden layer's activation.
    aggr : str
        Neighbor aggregation function used by SAGEConv: "mean", "max", or
        "lstm". "mean" is the standard/default choice and works well for
        similarity graphs like ours.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = config.HIDDEN_CHANNELS,
        out_channels: int = config.NUM_CLASSES,
        num_layers: int = config.NUM_GNN_LAYERS,
        dropout: float = config.DROPOUT,
        aggr: str = "mean",
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError(
                f"GraphSAGE requires at least 2 layers (an input layer and "
                f"an output layer), got num_layers={num_layers}."
            )

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # Input layer: in_channels -> hidden_channels
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))

        # Hidden layers: hidden_channels -> hidden_channels
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))

        # Output layer: hidden_channels -> out_channels (raw logits)
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)]
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (num_nodes, in_channels)
            Node feature matrix.
        edge_index : torch.Tensor, shape (2, num_edges)
            Graph connectivity in COO format.
        edge_weight : torch.Tensor, optional
            Accepted for interface consistency with GCN/GAT, but standard
            SAGEConv does not use edge weights (mean/max aggregation treats
            all neighbors equally). Silently ignored if provided.

        Returns
        -------
        torch.Tensor, shape (num_nodes, out_channels)
            Raw class logits for every node.
        """
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x


# ---------------------------------------------------------------------------
# Manual smoke test: `python -m src.models.graphsage` from the project root.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    num_nodes = 20
    in_channels = config.NUM_FEATURES
    out_channels = config.NUM_CLASSES

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 80))

    model = GraphSAGE(in_channels=in_channels, out_channels=out_channels)
    logits = model(x, edge_index)

    print(model)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}  (expected: ({num_nodes}, {out_channels}))")
    assert logits.shape == (num_nodes, out_channels), "Output shape mismatch!"
    print("Smoke test passed.")
