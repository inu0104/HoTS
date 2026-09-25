"""
Data Loading for GNN Calibration (GETS Style).

Split Strategy:
- ALL datasets: 20% train / 10% val / 70% test (random split by seed)
- This matches GETS paper's approach for fair comparison

LCC Extraction (GETS style):
- CoraFull, Amazon (Computers, Photo), Coauthor (CS, Physics): Extract LCC
- Planetoid (Cora, CiteSeer, PubMed): Use full graph with self-loop

From: run_experiment.py / GETS-main/dataset/dataset.py
"""

import os
import numpy as np
import torch
import dgl
import networkx as nx
import random
from dgl import DGLGraph, AddSelfLoop
from dgl.data import (
    CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset,
    CoraFullDataset, RedditDataset,
    AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset,
    CoauthorCSDataset, CoauthorPhysicsDataset,
    TexasDataset, CornellDataset, WisconsinDataset,
    ChameleonDataset, SquirrelDataset, ActorDataset,
)

from pathlib import Path
DATA_ROOT = os.environ.get('HOTS_DATA_ROOT') or str(
    Path(__file__).resolve().parent.parent.parent / 'data'
)


# =============================================================================
# Seed Setting
# =============================================================================

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    dgl.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# Synthetic SBM Generator
# =============================================================================

def _generate_synthetic_sbm(device, cfg):
    """
    Generate a multi-class SBM graph with class-conditional Gaussian features.

    Supports per-class homophily (mixed-h SBM) and per-node feature noise
    heterogeneity to expose node-level calibration differences that uniform
    SBM hides.

    Args:
        device: torch device
        cfg: dict with keys:
            h:            target graph homophily ∈ [0, 1] (used when h_per_class not set)
            h_per_class:  optional list of length C; per-class intra-class homophily
                          (overrides h when provided). Inter-class probability is
                          derived from mean(h_per_class).
            C:            number of classes
            n:            number of nodes (will be rounded to multiple of C)
            d:            target average degree
            mu:           class feature separation (Gaussian mean magnitude)
            sigma:        feature noise std
            sigma_hetero: optional bool. If True, half the nodes use sigma*0.5
                          and half use sigma*2.0, creating per-node confidence
                          heterogeneity.
            dim:          feature dimension (>= C, default = max(C, 16))
            seed:         RNG seed (for reproducibility)

    Returns:
        g: DGLGraph with self-loops
        features: (n, dim) tensor
        labels: (n,) tensor
        num_classes: C
    """
    C = int(cfg['C'])
    n = int(cfg['n'])
    d_target = float(cfg.get('d', 10))
    mu = float(cfg.get('mu', 2.0))
    sigma = float(cfg.get('sigma', 1.0))
    dim = int(cfg.get('dim', max(C, 16)))
    seed = int(cfg.get('seed', 0))
    sigma_hetero = bool(cfg.get('sigma_hetero', False))
    h_per_class = cfg.get('h_per_class', None)

    rng = np.random.RandomState(seed)

    # Round n to be divisible by C
    n = (n // C) * C
    nc = n // C  # nodes per class

    # Labels: equal-size classes
    labels = np.repeat(np.arange(C, dtype=np.int64), nc)

    # Generate edges (undirected, no self-loop yet)
    rows, cols = np.triu_indices(n, k=1)
    same_class = labels[rows] == labels[cols]

    if h_per_class is not None:
        # Mixed-h SBM: per-class intra-class AND per-class inter-class probabilities.
        # For an inter-class edge (c, c'), we use (q_c + q_{c'})/2 to preserve
        # symmetry while approximately matching each class's target degree and h.
        h_arr = np.asarray(h_per_class, dtype=np.float64)
        assert len(h_arr) == C, (
            f"h_per_class must have length C={C}, got {len(h_arr)}"
        )
        # Per-class intra-class edge probability (to hit h_c * d_target intra-edges)
        p_per_class = h_arr * d_target / max(nc - 1, 1)
        # Per-class inter-class probability (to hit (1 - h_c) * d_target inter-edges)
        q_per_class = ((1.0 - h_arr) * d_target / max(nc * (C - 1), 1)
                       if C > 1 else np.zeros(C, dtype=np.float64))
        p_per_class = np.clip(p_per_class, 0.0, 1.0)
        q_per_class = np.clip(q_per_class, 0.0, 1.0)

        lr = labels[rows]
        lc = labels[cols]
        probs = np.empty(len(rows), dtype=np.float64)
        same_idx = np.where(same_class)[0]
        diff_idx = np.where(~same_class)[0]
        probs[same_idx] = p_per_class[lr[same_idx]]
        probs[diff_idx] = 0.5 * (q_per_class[lr[diff_idx]] + q_per_class[lc[diff_idx]])
    else:
        # Uniform homophily: single p, single q
        h_target = float(cfg['h'])
        a = h_target * d_target
        b = (1 - h_target) * d_target
        p = a / max(nc - 1, 1)
        q = b / max(nc * (C - 1), 1) if C > 1 else 0.0
        p = float(np.clip(p, 0.0, 1.0))
        q = float(np.clip(q, 0.0, 1.0))
        probs = np.where(same_class, p, q)

    mask = rng.random(len(probs)) < probs
    src_arr = np.concatenate([rows[mask], cols[mask]])
    dst_arr = np.concatenate([cols[mask], rows[mask]])

    # Build DGL graph and add self-loops
    g = dgl.graph((src_arr, dst_arr), num_nodes=n)
    g = g.remove_self_loop().add_self_loop()
    g = g.int().to(device)

    # Class-conditional Gaussian features
    # Class means: orthogonal one-hot directions scaled by mu
    means = np.zeros((C, dim), dtype=np.float32)
    for c in range(C):
        means[c, c % dim] = mu

    if sigma_hetero:
        # Per-node sigma: half nodes get sigma*0.5 (clean), half get sigma*2.0 (noisy)
        # Sampled deterministically from the same RNG for reproducibility.
        hetero_mask = rng.random(n) < 0.5
        per_node_sigma = np.where(hetero_mask, sigma * 0.5, sigma * 2.0).astype(np.float32)
        noise = rng.randn(n, dim).astype(np.float32) * per_node_sigma[:, None]
    else:
        noise = rng.randn(n, dim).astype(np.float32) * sigma

    features_np = (noise + means[labels]).astype(np.float32)
    features = torch.from_numpy(features_np).to(device)
    labels_t = torch.from_numpy(labels).to(device)

    return g, features, labels_t, C


# =============================================================================
# Dataset Loading (GETS Style)
# =============================================================================

def load_dataset(name, device, synthetic_cfg=None):
    """Load dataset GETS style.

    Args:
        name: Dataset name (case-insensitive)
        device: torch device
        synthetic_cfg: dict with keys (h, C, n, d, mu, sigma, dim, seed) for synthetic SBM

    Returns:
        g: DGL graph (with self-loops for Planetoid/heterophilic)
        features: Node features tensor
        labels: Node labels tensor
        num_classes: Number of classes
    """
    name_lower = name.lower()

    # Synthetic SBM dataset (on-the-fly, no caching)
    if name_lower == 'synthetic':
        return _generate_synthetic_sbm(device, synthetic_cfg)

    # Planetoid datasets: with self-loop, full graph
    if name_lower == 'cora':
        data = CoraGraphDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    elif name_lower == 'citeseer':
        data = CiteseerGraphDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    elif name_lower == 'pubmed':
        data = PubmedGraphDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    # Reddit: with self-loop, full graph (GETS style)
    elif name_lower == 'reddit':
        data = RedditDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    # CoraFull: AddSelfLoop BEFORE LCC extraction (GETS style)
    elif name_lower == 'corafull':
        data = CoraFullDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']

        # Extract LCC
        nx_g = g.to_networkx().to_undirected()
        largest_cc = max(nx.connected_components(nx_g), key=len)
        largest_cc = list(largest_cc)
        subgraph = nx_g.subgraph(largest_cc).copy()
        g = DGLGraph(subgraph).to(device)

        features = features[largest_cc].to(device)
        labels = labels[largest_cc].to(device)

    # Amazon/Coauthor: NO self-loop initially, LCC extraction, then add self-loop
    elif name_lower in ['computers', 'photo', 'cs', 'physics']:
        if name_lower == 'computers':
            data = AmazonCoBuyComputerDataset(raw_dir=DATA_ROOT)
        elif name_lower == 'photo':
            data = AmazonCoBuyPhotoDataset(raw_dir=DATA_ROOT)
        elif name_lower == 'cs':
            data = CoauthorCSDataset(raw_dir=DATA_ROOT)
        else:
            data = CoauthorPhysicsDataset(raw_dir=DATA_ROOT)

        g = data[0]
        features = g.ndata['feat']
        labels = g.ndata['label']

        # Extract LCC
        nx_g = g.to_networkx().to_undirected()
        largest_cc = max(nx.connected_components(nx_g), key=len)
        largest_cc = list(largest_cc)
        subgraph = nx_g.subgraph(largest_cc).copy()
        g = DGLGraph(subgraph).to(device)

        features = features[largest_cc].to(device)
        labels = labels[largest_cc].to(device)

    # Heterophilic datasets (WebKB) - with self-loop (has 0-in-degree nodes)
    elif name_lower == 'texas':
        data = TexasDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    elif name_lower == 'cornell':
        data = CornellDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    elif name_lower == 'wisconsin':
        data = WisconsinDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    # Heterophilic datasets (Wikipedia) - no self-loop, no LCC
    elif name_lower == 'chameleon':
        data = ChameleonDataset(raw_dir=DATA_ROOT)
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    elif name_lower == 'squirrel':
        data = SquirrelDataset(raw_dir=DATA_ROOT)
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    elif name_lower == 'actor':
        data = ActorDataset(raw_dir=DATA_ROOT, transform=AddSelfLoop())
        g = data[0].int().to(device)
        features = g.ndata['feat'].to(device)
        labels = g.ndata['label'].to(device)

    # HGB datasets (PyG) - no self-loop, LCC already applied
    elif name_lower == 'roman_empire':
        from torch_geometric.datasets import HeterophilousGraphDataset
        pyg_data = HeterophilousGraphDataset(root=DATA_ROOT, name='Roman-empire')[0]

        src = pyg_data.edge_index[0].numpy()
        dst = pyg_data.edge_index[1].numpy()
        g = dgl.graph((src, dst)).to(device)

        features = pyg_data.x.to(device)
        labels = pyg_data.y.to(device)

    elif name_lower == 'tolokers':
        from torch_geometric.datasets import HeterophilousGraphDataset
        pyg_data = HeterophilousGraphDataset(root=DATA_ROOT, name='Tolokers')[0]

        src = pyg_data.edge_index[0].numpy()
        dst = pyg_data.edge_index[1].numpy()
        g = dgl.graph((src, dst)).to(device)

        features = pyg_data.x.to(device)
        labels = pyg_data.y.to(device)

    # OGB datasets - GETS style: bidirectional + self-loop
    elif name_lower == 'ogbn-arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        ogb_root = DATA_ROOT
        data = DglNodePropPredDataset(name="ogbn-arxiv", root=ogb_root)
        g, labels = data[0]
        g = g.int().to(device)

        # Add bidirectional edges (GETS style)
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)

        # Remove then add self-loop (GETS style)
        g = g.remove_self_loop().add_self_loop()
        g.create_formats_()

        features = g.ndata['feat'].to(device)
        labels = labels[:, 0].to(device)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    num_classes = len(torch.unique(labels))
    return g, features, labels, num_classes


# =============================================================================
# Split Creation (GETS Style - 20/10/70)
# =============================================================================

def create_split(n, seed):
    """Create 20/10/70 train/val/test split (GETS style).

    Args:
        n: Number of nodes
        seed: Random seed

    Returns:
        train_idx, val_idx, test_idx: numpy arrays of indices
    """
    set_seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    train_end = int(0.2 * n)
    val_end = int(0.3 * n)

    return idx[:train_end], idx[train_end:val_end], idx[val_end:]


def get_official_split(name):
    """Return the dataset's OFFICIAL split (numpy index arrays).

    For ogbn-arxiv this is the standard OGB time-based (chronological) split:
    train on papers up to 2017, validate on 2018, test on 2019. Node indexing
    matches the graph returned by `load_dataset` (nodes are not permuted there),
    so these indices align with the loaded logits/labels.

    Raises for datasets without a defined official split.
    """
    name_lower = name.lower()
    if name_lower == 'ogbn-arxiv':
        from ogb.nodeproppred import DglNodePropPredDataset
        data = DglNodePropPredDataset(name="ogbn-arxiv", root=DATA_ROOT)
        split = data.get_idx_split()
        train_idx = split['train'].cpu().numpy()
        val_idx = split['valid'].cpu().numpy()
        test_idx = split['test'].cpu().numpy()
        return train_idx, val_idx, test_idx
    raise NotImplementedError(
        f"No official split defined for dataset '{name}'. "
        f"Official split is only supported for ogbn-arxiv."
    )


# =============================================================================
# Utility Functions
# =============================================================================

def compute_gt_homophily(g, labels, device):
    """Compute ground-truth node-level homophily.

    Args:
        g: DGL graph
        labels: Node labels tensor
        device: torch device

    Returns:
        Tensor of shape [num_nodes] with homophily values
    """
    src, dst = g.edges()
    src, dst = src.long(), dst.long()
    num_nodes = g.num_nodes()
    same_label = (labels[src] == labels[dst]).float()
    same_count = torch.zeros(num_nodes, device=device)
    same_count.scatter_add_(0, src, same_label)
    out_degrees = torch.zeros(num_nodes, device=device)
    out_degrees.scatter_add_(0, src, torch.ones(len(src), device=device))
    out_degrees = torch.clamp(out_degrees, min=1.0)
    return same_count / out_degrees


def get_available_datasets():
    """Get list of all available datasets."""
    return {
        'homophilic': ['cora', 'citeseer', 'pubmed', 'computers', 'photo', 'cs', 'physics', 'corafull'],
        'heterophilic': ['texas', 'cornell', 'wisconsin', 'chameleon', 'squirrel', 'actor',
                         'roman_empire', 'tolokers'],
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    print("Testing GETS-style dataset loading...\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Test a few datasets
    test_datasets = ['cora', 'citeseer', 'computers', 'texas']

    for name in test_datasets:
        print(f"{'='*60}")
        print(f"Dataset: {name}")
        try:
            g, features, labels, num_classes = load_dataset(name, device)
            print(f"  Nodes: {g.num_nodes()}")
            print(f"  Edges: {g.num_edges()}")
            print(f"  Features: {features.shape}")
            print(f"  Classes: {num_classes}")

            # Test split
            train_idx, val_idx, test_idx = create_split(g.num_nodes(), seed=0)
            n = g.num_nodes()
            print(f"  Split: {len(train_idx)}/{len(val_idx)}/{len(test_idx)} "
                  f"({len(train_idx)/n*100:.1f}%/{len(val_idx)/n*100:.1f}%/{len(test_idx)/n*100:.1f}%)")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All tests completed!")
