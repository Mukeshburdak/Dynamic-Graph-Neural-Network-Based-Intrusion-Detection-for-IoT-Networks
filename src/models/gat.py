"""
gat.py

Graph Attention Network (GAT) for node-level IoT intrusion classification.
Unlike GCN (fixed spectral weighting) and GraphSAGE (uniform mean/max
aggregation), GAT learns per-edge ATTENTION weights -- for every node, it
decides how much to trust each of its k similarity-based neighbors when
aggregating information, rather than treating them all equally. This is
well suited to our similarity graphs, where some of a node's k nearest
neighbors may be much more relevant than others (e.g. a node's #1 nearest
neighbor is far more similar than its #8 nearest neighbor).

Multi-head attention is used in the hidden layers (multiple independent
attention mechanisms run in parallel and are concatenated) and averaged in
the final output layer, following the original paper's design.

Reference architecture: Velickovic et al., "Graph Attention Networks"
(2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from src import config


class GAT(nn.Module):
    """
    Stacked GATConv layers with multi-head attention, ELU activation,
    dropout, and a final single-head layer producing raw class logits per
    node.

    Parameters
    ----------
    in_channels : int
        Number of input node features.
    hidden_channels : int
        Size of the hidden node embeddings PER ATTENTION HEAD. The actual
        hidden layer output width is hidden_channels * heads, since head
        outputs are concatenated.
    out_channels : int
        Number of output classes (config.NUM_CLASSES).
    num_layers : int
        Total number of GATConv layers (including input and output).
        Must be >= 2.
    dropout : float
        Dropout probability applied to attention coefficients and after
        each hidden layer's activation.
    heads : int
        Number of parallel attention heads used in hidden layers.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = config.HIDDEN_CHANNELS,
        out_channels: int = config.NUM_CLASSES,
        num_layers: int = config.NUM_GNN_LAYERS,
        dropout: float = config.DROPOUT,
        heads: int = 4,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError(
                f"GAT requires at least 2 layers (an input layer and an "
                f"output layer), got num_layers={num_layers}."
            )

        self.dropout = dropout
        self.convs = nn.ModuleList()

        # Input layer: in_channels -> hidden_channels, with `heads` parallel
        # attention heads concatenated (output width = hidden_channels * heads).
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=1,  # allows GATConv to use our 1-D edge_attr (similarity score)
            )
        )

        # Hidden layers: (hidden_channels * heads) -> hidden_channels, still
        # multi-head concatenated.
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=1,
                )
            )

        # Output layer: (hidden_channels * heads) -> out_channels, single
        # head with concat=False so the output width is exactly
        # out_channels (heads are averaged instead of concatenated here).
        self.convs.append(
            GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
                edge_dim=1,
            )
        )

        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_channels * heads) for _ in range(num_layers - 1)]
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
        edge_weight : torch.Tensor, shape (num_edges,) or (num_edges, 1), optional
            Similarity score per edge (graph_builder.py's edge_attr). Passed
            to GATConv as edge_attr so the model can use it alongside its
            learned attention -- e.g. as a prior that biases attention
            toward geometrically closer neighbors.

        Returns
        -------
        torch.Tensor, shape (num_nodes, out_channels)
            Raw class logits for every node.
        """
        edge_attr = edge_weight
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # GATConv expects (num_edges, edge_dim)

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.batch_norms[i](x)
            x = F.elu(x)  # ELU is standard for GAT (per the original paper)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        return x


# ---------------------------------------------------------------------------
# Manual smoke test: `python -m src.models.gat` from the project root.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    num_nodes = 20
    in_channels = config.NUM_FEATURES
    out_channels = config.NUM_CLASSES

    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, 80))
    edge_weight = torch.rand(80)

    model = GAT(in_channels=in_channels, out_channels=out_channels)
    logits = model(x, edge_index, edge_weight)

    print(model)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}  (expected: ({num_nodes}, {out_channels}))")
    assert logits.shape == (num_nodes, out_channels), "Output shape mismatch!"
    print("Smoke test passed.")
