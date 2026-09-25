"""
GNN Models for Node Classification.

All models are DGL-based (for GETS compatibility).
From: ref/GETS/model/gnns.py

Models:
- GCN: Graph Convolutional Network
- GAT: Graph Attention Network
- GIN: Graph Isomorphism Network
- GraphSAGE: Graph Sample and Aggregate
- MLP: Multi-Layer Perceptron (baseline, no graph structure)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import dgl.nn as dglnn


# =============================================================================
# GCN (Graph Convolutional Network)
# =============================================================================

class GCN(nn.Module):
    """Graph Convolutional Network.

    From: ref/GETS/model/gnns.py

    Args:
        in_size: Input feature dimension
        hid_size: Hidden layer dimension
        out_size: Output dimension (number of classes)
        num_layer: Number of GCN layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        norm: Whether to use BatchNorm (default: False)
    """
    def __init__(self, in_size, hid_size, out_size, num_layer=2, dropout=0.5, norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()

        if num_layer == 1:
            # True 1-layer: single message passing in_size -> out_size
            self.layers.append(dglnn.GraphConv(in_size, out_size))
        else:
            # First layer
            self.layers.append(dglnn.GraphConv(in_size, hid_size))
            if norm:
                self.norms.append(nn.BatchNorm1d(hid_size))

            # Middle layers
            for _ in range(num_layer - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size))
                if norm:
                    self.norms.append(nn.BatchNorm1d(hid_size))

            # Output layer
            self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


# =============================================================================
# GAT (Graph Attention Network)
# =============================================================================

class GAT(nn.Module):
    """Graph Attention Network.

    From: ref/GETS/model/gnns.py

    Args:
        in_size: Input feature dimension
        hid_size: Hidden layer dimension
        out_size: Output dimension (number of classes)
        num_layer: Number of GAT layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        norm: Whether to use BatchNorm (default: False)
        num_heads: Number of attention heads (default: 2)
    """
    def __init__(self, in_size, hid_size, out_size, num_layer=2, dropout=0.5, norm=False, num_heads=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()

        # First GAT layer
        self.layers.append(
            dglnn.GATConv(in_size, hid_size, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
        )
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size * num_heads))

        # Output projection
        self.final_project = nn.Linear(hid_size * num_heads, out_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        # Flatten multi-head output and project
        h = self.final_project(h.view(h.size(0), -1))
        return h


# =============================================================================
# GIN (Graph Isomorphism Network)
# =============================================================================

class GIN(nn.Module):
    """Graph Isomorphism Network.

    From: ref/GETS/model/gnns.py

    Args:
        in_size: Input feature dimension
        hid_size: Hidden layer dimension
        out_size: Output dimension (number of classes)
        num_layer: Number of GIN layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        norm: Whether to use BatchNorm (default: False)
    """
    def __init__(self, in_size, hid_size, out_size, num_layer=2, dropout=0.5, norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()

        # First layer
        self.layers.append(
            dglnn.GINConv(
                nn.Sequential(nn.Linear(in_size, hid_size)),
                'mean'
            )
        )
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size))

        # Middle layers
        for _ in range(num_layer - 2):
            self.layers.append(
                dglnn.GINConv(
                    nn.Sequential(nn.Linear(hid_size, hid_size)),
                    'mean'
                )
            )
            if norm:
                self.norms.append(nn.BatchNorm1d(hid_size))

        # Output layer
        self.layers.append(
            dglnn.GINConv(
                nn.Sequential(nn.Linear(hid_size, out_size)),
                'mean'
            )
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


# =============================================================================
# GraphSAGE (Graph Sample and Aggregate)
# =============================================================================

class GraphSAGE(nn.Module):
    """GraphSAGE: Graph Sample and Aggregate.

    Args:
        in_size: Input feature dimension
        hid_size: Hidden layer dimension
        out_size: Output dimension (number of classes)
        num_layer: Number of SAGE layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        norm: Whether to use BatchNorm (default: False)
        aggregator_type: Aggregator type ('mean', 'gcn', 'pool', 'lstm') (default: 'mean')
    """
    def __init__(self, in_size, hid_size, out_size, num_layer=2, dropout=0.5, norm=False, aggregator_type='mean'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()

        # First layer
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, aggregator_type))
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size))

        # Middle layers
        for _ in range(num_layer - 2):
            self.layers.append(dglnn.SAGEConv(hid_size, hid_size, aggregator_type))
            if norm:
                self.norms.append(nn.BatchNorm1d(hid_size))

        # Output layer
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, aggregator_type))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


# =============================================================================
# MLP (Multi-Layer Perceptron) - No graph structure baseline
# =============================================================================

class MLP(nn.Module):
    """Multi-Layer Perceptron (no graph structure).

    Used as baseline to compare with GNNs.
    Takes same interface as GNNs but ignores graph structure.

    Args:
        in_size: Input feature dimension
        hid_size: Hidden layer dimension
        out_size: Output dimension (number of classes)
        num_layer: Number of layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        norm: Whether to use BatchNorm (default: False)
    """
    def __init__(self, in_size, hid_size, out_size, num_layer=2, dropout=0.5, norm=False):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = norm
        if norm:
            self.norms = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(in_size, hid_size))
        if norm:
            self.norms.append(nn.BatchNorm1d(hid_size))

        # Middle layers
        for _ in range(num_layer - 2):
            self.layers.append(nn.Linear(hid_size, hid_size))
            if norm:
                self.norms.append(nn.BatchNorm1d(hid_size))

        # Output layer
        self.layers.append(nn.Linear(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        """Forward pass. Graph g is ignored (MLP doesn't use graph structure)."""
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                if self.norm:
                    h = self.norms[i](h)
                h = F.relu(h)
                h = self.dropout(h)
        return h


# =============================================================================
# Model Factory
# =============================================================================

# =============================================================================
# Heterophily-designed backbones (for reviewer 1i4a Q3/Q4)
# All share the (in_size, hid_size, out_size, num_layer, dropout, norm) interface
# and forward(g, features) -> logits. HoTS is post-hoc, so it calibrates whatever
# logits these produce. Graphs are expected to carry self-loops (data pipeline
# adds them), consistent with GCN/GAT here.
# =============================================================================

class GCNII(nn.Module):
    """GCNII (Chen et al., 2020): identity mapping + initial residual.

    h^{l+1} = ReLU( ((1-a) P h^l + a h0) ((1-b_l) I + b_l W_l) ),
    P = sym-normalized adjacency (self-loops included), b_l = log(lambda/l + 1).
    """
    def __init__(self, in_size, hid_size, out_size, num_layer=2, dropout=0.5, norm=False,
                 alpha=0.1, lamda=0.5):
        super().__init__()
        self.alpha = alpha
        self.lamda = lamda
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(in_size, hid_size)
        self.fc_out = nn.Linear(hid_size, out_size)
        self.convs = nn.ModuleList(
            [nn.Linear(hid_size, hid_size, bias=False) for _ in range(max(num_layer, 1))])
        self.prop = dglnn.GraphConv(hid_size, hid_size, weight=False, bias=False,
                                    norm='both', allow_zero_in_degree=True)

    def forward(self, g, features):
        h = self.dropout(features)
        h = F.relu(self.fc_in(h))
        h0 = h
        for l, W in enumerate(self.convs):
            h = self.dropout(h)
            hi = self.prop(g, h)
            support = (1 - self.alpha) * hi + self.alpha * h0
            beta = math.log(self.lamda / (l + 1) + 1)
            h = F.relu((1 - beta) * support + beta * W(support))
        h = self.dropout(h)
        return self.fc_out(h)


class GPRGNN(nn.Module):
    """GPR-GNN (Chien et al., 2021): predict then Generalized-PageRank propagate.

    Z = sum_{k=0}^K gamma_k P^k f_theta(X), gamma learnable (PPR-initialized).
    """
    def __init__(self, in_size, hid_size, out_size, num_layer=2, dropout=0.5, norm=False,
                 K=10, alpha=0.1):
        super().__init__()
        self.K = K
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_size, hid_size)
        self.fc2 = nn.Linear(hid_size, out_size)
        self.prop = dglnn.GraphConv(out_size, out_size, weight=False, bias=False,
                                    norm='both', allow_zero_in_degree=True)
        gamma = [alpha * (1 - alpha) ** k for k in range(K + 1)]
        gamma[-1] = (1 - alpha) ** K
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, g, features):
        h = self.dropout(features)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        z = self.gamma[0] * h
        for k in range(1, self.K + 1):
            h = self.prop(g, h)
            z = z + self.gamma[k] * h
        return z


class FAGCN(nn.Module):
    """FAGCN (Bo et al., 2021): frequency-adaptive signed aggregation.

    h^{l+1} = eps * h0 + sum_j (alpha_ij / sqrt(d_i d_j)) h_j,
    alpha_ij = tanh(g^T [h_i || h_j]) in (-1,1) mixes low/high-frequency signals.
    """
    def __init__(self, in_size, hid_size, out_size, num_layer=2, dropout=0.5, norm=False,
                 eps=0.3):
        super().__init__()
        self.eps = eps
        self.num_layer = max(num_layer, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc_in = nn.Linear(in_size, hid_size)
        self.fc_out = nn.Linear(hid_size, out_size)
        self.gates = nn.ModuleList([nn.Linear(2 * hid_size, 1) for _ in range(self.num_layer)])

    def _propagate(self, g, h, gate):
        g = g.local_var()
        deg = g.in_degrees().float().clamp(min=1)
        g.ndata['h'] = h
        g.ndata['d'] = torch.pow(deg, -0.5).unsqueeze(1)

        def edge_fn(edges):
            z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
            a = torch.tanh(gate(z2))  # signed coefficient in (-1, 1)
            coef = a * edges.src['d'] * edges.dst['d']
            return {'m': edges.src['h'] * coef}

        g.apply_edges(edge_fn)
        g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_new'))
        return g.ndata['h_new']

    def forward(self, g, features):
        h = self.dropout(features)
        h = F.relu(self.fc_in(h))
        h0 = h
        for l in range(self.num_layer):
            h = self.dropout(h)
            h = self.eps * h0 + self._propagate(g, h, self.gates[l])
        h = self.dropout(h)
        return self.fc_out(h)


def create_model(model_name: str, in_size: int, hid_size: int, out_size: int,
                 num_layer: int = 2, dropout: float = 0.5, norm: bool = False, **kwargs):
    """Create a GNN model by name.

    Args:
        model_name: One of 'gcn', 'gat', 'gin', 'graphsage'
        in_size: Input feature dimension
        hid_size: Hidden layer dimension
        out_size: Output dimension (number of classes)
        num_layer: Number of layers (default: 2)
        dropout: Dropout rate (default: 0.5)
        norm: Whether to use BatchNorm (default: False)
        **kwargs: Additional model-specific arguments

    Returns:
        GNN model instance
    """
    model_name = model_name.lower()

    if model_name == 'gcn':
        return GCN(in_size, hid_size, out_size, num_layer, dropout, norm)
    elif model_name == 'gat':
        num_heads = kwargs.get('num_heads', 2)
        return GAT(in_size, hid_size, out_size, num_layer, dropout, norm, num_heads)
    elif model_name == 'gin':
        return GIN(in_size, hid_size, out_size, num_layer, dropout, norm)
    elif model_name == 'graphsage' or model_name == 'sage':
        aggregator_type = kwargs.get('aggregator_type', 'mean')
        return GraphSAGE(in_size, hid_size, out_size, num_layer, dropout, norm, aggregator_type)
    elif model_name == 'mlp':
        return MLP(in_size, hid_size, out_size, num_layer, dropout, norm)
    elif model_name == 'gcnii':
        return GCNII(in_size, hid_size, out_size, num_layer, dropout, norm)
    elif model_name == 'gprgnn':
        return GPRGNN(in_size, hid_size, out_size, num_layer, dropout, norm)
    elif model_name == 'fagcn':
        return FAGCN(in_size, hid_size, out_size, num_layer, dropout, norm)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: gcn, gat, gin, graphsage, mlp, gcnii, gprgnn, fagcn")


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    import dgl

    # Create a simple test graph
    num_nodes = 100
    num_edges = 300
    in_size = 16
    hid_size = 32
    out_size = 5

    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    g = dgl.graph((src, dst))
    g = dgl.add_self_loop(g)

    features = torch.randn(num_nodes, in_size)

    print("Testing all models...")
    for model_name in ['gcn', 'gat', 'gin', 'graphsage', 'mlp']:
        model = create_model(model_name, in_size, hid_size, out_size)
        model.eval()
        with torch.no_grad():
            out = model(g, features)
        print(f"  {model_name.upper():10s}: input {features.shape} -> output {out.shape}")

    print("\nAll models work correctly!")
