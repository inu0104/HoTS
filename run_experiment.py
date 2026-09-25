"""
Unified experiment script for GNN calibration.

Features:
- GNN model training with model.pt saving/loading
- All calibrators run on THE SAME GNN logits (consistency guaranteed)
- Results saved in structured format

Usage:
    python run_experiment.py --dataset Cora --model GCN --seeds 0-9 --gpu 1
    python run_experiment.py --dataset all --model all --seeds 0-29 --gpu 1
"""

import os
import sys
import argparse

# Parse GPU argument BEFORE importing torch
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=int, default=0)
_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import json
import copy
from pathlib import Path
import yaml

import torch
import torch.nn.functional as F
from torch import optim

# Make `hots` package importable when run from project root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hots.models import create_model
from hots.data import load_dataset, create_split, get_official_split, compute_gt_homophily, set_seed
from hots.calibrators import TS, VS, ETS, CaGCN, GATS, GETS, HTS, WATS, DCGC, HoTS


# =============================================================================
# Constants
# =============================================================================

DATASETS = [
    'Cora', 'CiteSeer', 'PubMed',
    'Computers', 'Photo', 'CS', 'Physics', 'CoraFull',
    'Texas', 'Cornell', 'Wisconsin', 'Chameleon', 'Squirrel', 'Actor',
    'roman_empire', 'tolokers',
    'ogbn-arxiv', 'reddit'
]

MODELS = ['GCN', 'GAT', 'GIN', 'GraphSAGE']

# Calibrators that preserve predictions (pure calibration)
PURE_CALIBRATORS = ['TS', 'ETS', 'HTS', 'CaGCN', 'GATS', 'WATS', 'HoTS']

# Calibrators that may change predictions
PRED_CHANGING_CALIBRATORS = ['VS', 'GETS']

ALL_CALIBRATORS = PURE_CALIBRATORS + PRED_CHANGING_CALIBRATORS

MAIN_CALIBRATORS = ['Uncal', 'TS', 'VS', 'ETS', 'HTS', 'CaGCN', 'GATS', 'GETS', 'WATS', 'HoTS']
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_calibrators(names):
    """Resolve the main benchmark or an explicit list of calibrators."""
    if names is None or names in (['all'], ['main']):
        return MAIN_CALIBRATORS.copy()
    unknown = set(names) - set(ALL_CALIBRATORS) - {'Uncal'}
    if unknown:
        raise ValueError(f'Unknown calibrators: {sorted(unknown)}')
    return list(dict.fromkeys(names))

def load_config(dataset, calibrator=None, model=None):
    """Load dataset-specific config from yaml file.

    Args:
        dataset: Dataset name (e.g., 'Cora', 'CiteSeer')
        calibrator: If 'GETS', load from gets/; if 'HoTS', load from hots/
        model: Model name (e.g., 'GCN', 'GAT') - unused, kept for compatibility
    """
    # Handle dataset name mapping for base config
    name_map = {
        'corafull': 'cora-full',
        'citeseer': 'citeseer',
        'cora': 'cora',
        'pubmed': 'pubmed',
        'computers': 'computers',
        'photo': 'photo',
        'cs': 'cs',
        'physics': 'physics',
    }
    dataset_lower = dataset.lower()
    config_name = name_map.get(dataset_lower, dataset_lower)

    if calibrator == 'GETS':
        config_path = (PROJECT_ROOT / f'configs/gets/{config_name}.yaml')
    elif calibrator == 'HoTS':
        config_path = (PROJECT_ROOT / f'configs/hots/{config_name}.yaml')
        if not config_path.exists():
            # Fallback to hots/default.yaml for datasets without specific config
            config_path = (PROJECT_ROOT / 'configs/hots/default.yaml')
    elif calibrator == 'WATS':
        config_path = (PROJECT_ROOT / f'configs/wats/{config_name}.yaml')
        if not config_path.exists():
            config_path = (PROJECT_ROOT / 'configs/wats/default.yaml')
    else:
        config_path = (PROJECT_ROOT / f'configs/{config_name}.yaml')

    if not config_path.exists():
        # Fallback for datasets without config (heterophilic)
        return {
            'gnn': {'hid_dim': 16, 'dropout': 0.5, 'num_layer': 2},
            'train': {'epochs': 200, 'lr': 0.01, 'weight_decay': 5e-4, 'patience': 50},
            'calibration': {'epochs': 1000, 'patience': 50, 'cal_lr': 0.01,
                           'cal_weight_decay': 0, 'cal_dropout': 0.5}
        }

    with open(config_path) as f:
        return yaml.safe_load(f)


# =============================================================================
# Utility Functions
# =============================================================================

def set_seed_deterministic(seed):
    """Set seed with extra deterministic settings for training."""
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seeds(seed_str):
    """Parse seed string like '0-9' or '0,1,2' into list."""
    if '-' in seed_str:
        start, end = seed_str.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(s) for s in seed_str.split(',')]


# =============================================================================
# Metrics
# =============================================================================

def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error (equal-width bins)."""
    confidences = probs.max(dim=1).values
    predictions = probs.argmax(dim=1)
    accuracies = (predictions == labels).float()

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1, device=probs.device)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += torch.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece.item()


def compute_nll(probs, labels):
    """Compute Negative Log Likelihood."""
    eps = 1e-10
    probs_clamped = torch.clamp(probs, min=eps, max=1 - eps)
    log_probs = torch.log(probs_clamped)
    nll = F.nll_loss(log_probs, labels)
    return nll.item()


def compute_degree_ece(probs, labels, degrees, n_bins=10):
    """Compute degree-based ECE."""
    confidences = probs.max(dim=1).values
    predictions = probs.argmax(dim=1)
    correct = (predictions == labels).float()

    sorted_idx = torch.argsort(degrees)
    sorted_conf = confidences[sorted_idx]
    sorted_correct = correct[sorted_idx]

    n = len(degrees)
    bin_size = (n + n_bins - 1) // n_bins

    deg_ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, n)
        if start >= n:
            break

        bin_conf = sorted_conf[start:end].mean().item()
        bin_acc = sorted_correct[start:end].mean().item()
        bin_weight = (end - start) / n

        deg_ece += abs(bin_conf - bin_acc) * bin_weight

    return deg_ece


def compute_all_metrics(probs, labels, degrees):
    """Compute all metrics at once."""
    predictions = probs.argmax(dim=1)
    acc = (predictions == labels).float().mean().item()

    return {
        'acc': acc,
        'ece': compute_ece(probs, labels, n_bins=15),
        'deg_ece': compute_degree_ece(probs, labels, degrees, n_bins=10),
        'nll': compute_nll(probs, labels)
    }


# =============================================================================
# GNN Training
# =============================================================================

def train_gnn(g, features, labels, num_classes, train_idx, val_idx, device,
              model_type='gcn', hid_dim=16, num_layer=2, epochs=200, lr=0.01,
              weight_decay=5e-4, dropout=0.5, norm=False, patience=50, early_stop='acc'):
    """Train GNN model.

    Args:
        early_stop: 'acc' (GETS style, validation accuracy) or 'loss' (validation loss)
    """
    model = create_model(
        model_name=model_type,
        in_size=features.shape[1],
        hid_size=hid_dim,
        out_size=num_classes,
        num_layer=num_layer,
        dropout=dropout,
        norm=norm
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_idx_t = torch.tensor(train_idx, device=device)
    val_idx_t = torch.tensor(val_idx, device=device)

    # Initialize based on early_stop criterion
    if early_stop == 'acc':
        best_val_metric = -1.0  # Higher is better
    else:  # loss
        best_val_metric = float('inf')  # Lower is better

    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        logits = model(g, features)
        loss = F.cross_entropy(logits[train_idx_t], labels[train_idx_t])

        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = logits[val_idx_t]
            if early_stop == 'acc':
                val_preds = val_logits.argmax(dim=1)
                val_metric = (val_preds == labels[val_idx_t]).float().mean().item()
                is_better = val_metric > best_val_metric
            else:  # loss
                val_metric = F.cross_entropy(val_logits, labels[val_idx_t]).item()
                is_better = val_metric < best_val_metric

        if is_better:
            best_val_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience and patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    return model


# =============================================================================
# Main Experiment Logic
# =============================================================================
# NOTE: Calibrator classes removed. Import from hots/calibrators/ when needed.


# =============================================================================
# Main Experiment Logic
# =============================================================================

def save_result(save_dir, name, probs, logits, labels, degrees, homophily, metrics, extra_info=None):
    """Save calibration result."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics to JSON
    result = {'metrics': metrics}
    if extra_info:
        result['calibrator_info'] = extra_info

    with open(save_dir / f'{name}.json', 'w') as f:
        json.dump(result, f, indent=2)

    # Save per-node data
    per_node = {
        'probs': probs,
        'logits': logits,
        'labels': labels,
        'preds': probs.argmax(dim=1),
        'confidences': probs.max(dim=1).values,
        'correct': (probs.argmax(dim=1) == labels),
        'degrees': degrees,
        'homophily': homophily
    }
    torch.save(per_node, save_dir / f'{name}_per_node.pt')


def run_single_experiment(dataset, model_type, seed, save_dir, device, calibrators=None, early_stop='acc', synthetic_cfg=None, base_training='erm', split='random'):
    """Run single experiment (one dataset, one model, one seed).

    Args:
        calibrators: List of calibrators to run. If None or 'all', run all.
        early_stop: 'acc' (GETS style) or 'loss' for early stopping criterion.
        base_training: 'erm' (default) or 'dcgc'. If 'dcgc', run DCGC once after
            ERM training to produce corrected base logits, then all downstream
            calibrators fit on those corrected logits.
    """
    calibrators = resolve_calibrators(calibrators)

    # ERM save path (model.pt, logits_hash.pt, and per-calibrator results).
    if dataset.lower() == 'synthetic' and synthetic_cfg is not None:
        if synthetic_cfg.get('h_per_class') is not None:
            h_tag = "hmix" + "-".join(f"{h:.2f}" for h in synthetic_cfg['h_per_class'])
        else:
            h_tag = f"h{synthetic_cfg['h']:.2f}"
        shet_tag = "_shet" if synthetic_cfg.get('sigma_hetero', False) else ""
        tag = (f"{h_tag}_C{int(synthetic_cfg['C'])}"
               f"_d{synthetic_cfg.get('d', 10)}"
               f"_mu{synthetic_cfg.get('mu', 2.0)}"
               f"_n{int(synthetic_cfg['n'])}{shet_tag}")
        erm_save_path = Path(save_dir) / 'synthetic' / tag / model_type / f'seed_{seed}'
    else:
        erm_save_path = Path(save_dir) / dataset / model_type / f'seed_{seed}'
    erm_save_path.mkdir(parents=True, exist_ok=True)

    # For DCGC base correction, redirect to a sibling dir so ERM results stay intact.
    if base_training == 'dcgc':
        base_dir = Path(save_dir).name
        parent = Path(save_dir).parent
        dcgc_save_dir = parent / (base_dir + '_dcgc_base')
        if dataset.lower() == 'synthetic' and synthetic_cfg is not None:
            save_path = dcgc_save_dir / 'synthetic' / tag / model_type / f'seed_{seed}'
        else:
            save_path = dcgc_save_dir / dataset / model_type / f'seed_{seed}'
    else:
        save_path = erm_save_path
    save_path.mkdir(parents=True, exist_ok=True)

    # Set seed
    set_seed(seed)

    # Load dataset (synthetic uses cfg with seed for reproducibility)
    if dataset.lower() == 'synthetic' and synthetic_cfg is not None:
        cfg_with_seed = {**synthetic_cfg, 'seed': seed}
        g, features, labels, num_classes = load_dataset('synthetic', device, synthetic_cfg=cfg_with_seed)
    else:
        g, features, labels, num_classes = load_dataset(dataset, device)
    n = len(labels)

    # Create split
    if split == 'official':
        train_idx, val_idx, test_idx = get_official_split(dataset)
        print(f"  Using OFFICIAL split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        train_idx, val_idx, test_idx = create_split(n, seed)

    # Get config from yaml
    cfg = load_config(dataset)
    gnn_cfg = cfg['gnn']
    train_cfg = cfg['train']
    cal_cfg = cfg.get('calibration', {})

    # Ensure numeric types (yaml may parse 1e-2 as string, ~ as None)
    train_cfg['lr'] = float(train_cfg['lr'])
    train_cfg['weight_decay'] = float(train_cfg['weight_decay'])
    # patience: ~ means no early stopping (GETS style)
    # Don't set default - let None pass through

    # Calibration HP from yaml (with defaults)
    cal_epochs = cal_cfg.get('epochs', 1000)
    cal_patience = cal_cfg.get('patience', 50)
    cal_lr = float(cal_cfg.get('cal_lr', 0.01))
    cal_wd = float(cal_cfg.get('cal_weight_decay', 0))
    cal_dropout = cal_cfg.get('cal_dropout', 0.5)

    # Check if model exists (always loaded/trained at ERM save path,
    # at the ERM save path).
    model_path = erm_save_path / 'model.pt'
    gnn_norm = gnn_cfg.get('norm', False) or False  # Handle None as False
    gnn_num_layer = gnn_cfg.get('num_layer', 2)

    if model_path.exists():
        print(f"  Loading existing model from {model_path}")
        model = create_model(
            model_name=model_type.lower(),
            in_size=features.shape[1],
            hid_size=gnn_cfg['hid_dim'],
            out_size=num_classes,
            num_layer=gnn_num_layer,
            dropout=gnn_cfg['dropout'],
            norm=gnn_norm
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print(f"  Training new model...")
        model = train_gnn(
            g, features, labels, num_classes, train_idx, val_idx, device,
            model_type=model_type.lower(),
            hid_dim=gnn_cfg['hid_dim'],
            num_layer=gnn_num_layer,
            epochs=train_cfg['epochs'],
            lr=train_cfg['lr'],
            weight_decay=train_cfg['weight_decay'],
            dropout=gnn_cfg['dropout'],
            norm=gnn_norm,
            patience=train_cfg.get('patience'),  # None = no early stopping
            early_stop=early_stop
        )
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved to {model_path}")

    # Get logits (SAME for all calibrators)
    model.eval()
    with torch.no_grad():
        logits = model(g, features)

    # Verify model consistency via logits hash (ERM backbone hash).
    logits_hash_path = erm_save_path / 'logits_hash.pt'
    current_hash = logits.sum().item()
    if logits_hash_path.exists():
        saved_hash = torch.load(logits_hash_path, map_location='cpu', weights_only=True)
        # Use relative tolerance for large values (1e-5 relative error)
        rel_diff = abs(saved_hash - current_hash) / (abs(saved_hash) + 1e-8)
        if rel_diff > 1e-5:
            raise RuntimeError(f"Model inconsistency! Logits hash mismatch: saved={saved_hash:.6f}, current={current_hash:.6f}, rel_diff={rel_diff:.2e}")
        print(f"  Model consistency verified (hash={current_hash:.6f})")
    else:
        torch.save(current_hash, logits_hash_path)
        print(f"  Logits hash saved: {current_hash:.6f}")

    # Prepare data
    test_idx_t = torch.tensor(test_idx, device=device)
    val_idx_t = torch.tensor(val_idx, device=device)
    train_idx_t = torch.tensor(train_idx, device=device)

    # Optional DCGC base correction. Replaces `logits` with DCGC-corrected ones,
    # so downstream calibrators (TS, HoTS, ...) fit on top of DCGC.
    if base_training == 'dcgc':
        print(f"  Applying DCGC base correction...")
        set_seed(seed)
        _large_graph = g.num_edges() > 50_000_000
        if _large_graph:
            print(f"    (enabling grad-ckpt + fp16 for large graph: E={g.num_edges():,})")
        dcgc_base = DCGC(num_classes, device, dropout=cal_dropout, alpha=0.5, beta=10.0,
                         checkpoint_forward=_large_graph,
                         amp_forward=_large_graph)
        dcgc_base.fit(logits, labels, g, features, val_idx_t, train_idx_t,
                      model=model, epochs=cal_epochs, patience=cal_patience,
                      lr=cal_lr, weight_decay=cal_wd)
        logits = dcgc_base._calibrated_logits.detach()
        dcgc_base_acc = (F.softmax(logits[test_idx_t], dim=1).argmax(dim=1)
                         == labels[test_idx_t]).float().mean().item()
        print(f"  DCGC base acc: {dcgc_base_acc:.4f}")
        del dcgc_base

    test_logits = logits[test_idx_t]
    test_labels = labels[test_idx_t]
    val_logits = logits[val_idx_t].cpu().numpy()
    val_labels = labels[val_idx_t].cpu().numpy()

    degrees = g.in_degrees()[test_idx_t].float().cpu()
    gt_homophily = compute_gt_homophily(g, labels, device)
    homophily_test = gt_homophily[test_idx_t].cpu()

    # Uncalibrated accuracy (reference)
    probs_uncal = F.softmax(test_logits, dim=1).cpu()
    uncal_acc = (probs_uncal.argmax(dim=1) == test_labels.cpu()).float().mean().item()

    results = {}

    def should_run(cal_name):
        return cal_name in calibrators

    # 1. Uncalibrated
    if should_run('Uncal'):
        metrics = compute_all_metrics(probs_uncal, test_labels.cpu(), degrees)
        save_result(save_path, 'Uncal', probs_uncal, test_logits.cpu(), test_labels.cpu(),
                    degrees, homophily_test, metrics)
        results['Uncal'] = metrics

    # 2. TS (GETS style)
    if should_run('TS'):
        ts = TS(num_classes, device)
        ts.fit(logits, labels, val_idx_t, train_idx_t, epochs=cal_epochs, patience=cal_patience, lr=cal_lr, weight_decay=cal_wd)
        cal_logits_ts = ts.calibrate(logits)[test_idx_t].cpu()
        probs_ts = F.softmax(cal_logits_ts, dim=1)

        # Check accuracy unchanged (warning only)
        ts_acc = (probs_ts.argmax(dim=1) == test_labels.cpu()).float().mean().item()
        if abs(ts_acc - uncal_acc) > 1e-6:
            print(f"  WARNING: TS changed accuracy! {uncal_acc:.6f} -> {ts_acc:.6f}")

        metrics = compute_all_metrics(probs_ts, test_labels.cpu(), degrees)
        save_result(save_path, 'TS', probs_ts, cal_logits_ts, test_labels.cpu(),
                    degrees, homophily_test, metrics, {'temperature': ts.temperature.item()})
        results['TS'] = metrics

    # 3. VS (GETS style - may change predictions)
    if should_run('VS'):
        vs = VS(num_classes, device)
        vs.fit(logits, labels, val_idx_t, train_idx_t, epochs=cal_epochs, patience=cal_patience, lr=cal_lr, weight_decay=cal_wd)
        cal_logits_vs = vs.calibrate(logits)[test_idx_t].cpu()
        probs_vs = F.softmax(cal_logits_vs, dim=1)
        metrics = compute_all_metrics(probs_vs, test_labels.cpu(), degrees)
        save_result(save_path, 'VS', probs_vs, cal_logits_vs, test_labels.cpu(),
                    degrees, homophily_test, metrics, {'temp': vs.temperature.tolist(), 'bias': vs.bias.tolist()})
        results['VS'] = metrics

    # 4. ETS (GETS style)
    if should_run('ETS'):
        ets = ETS(num_classes, device)
        ets.fit(logits, labels, val_idx_t, train_idx_t, epochs=cal_epochs, patience=cal_patience, lr=cal_lr, weight_decay=cal_wd)
        cal_logits_ets = ets.calibrate(logits[test_idx_t]).cpu()
        probs_ets = F.softmax(cal_logits_ets, dim=1)

        # Check accuracy (warning only - ETS can change predictions due to probability mixing)
        ets_acc = (probs_ets.argmax(dim=1) == test_labels.cpu()).float().mean().item()
        if abs(ets_acc - uncal_acc) > 1e-6:
            print(f"  WARNING: ETS changed accuracy! {uncal_acc:.6f} -> {ets_acc:.6f}")

        metrics = compute_all_metrics(probs_ets, test_labels.cpu(), degrees)
        save_result(save_path, 'ETS', probs_ets, cal_logits_ets, test_labels.cpu(),
                    degrees, homophily_test, metrics, {'weights': [ets.w1, ets.w2, ets.w3]})
        results['ETS'] = metrics

    # 5. HTS
    if should_run('HTS'):
        hts = HTS(num_classes, device)
        hts.fit(logits, labels, val_idx_t, train_idx_t, epochs=cal_epochs, patience=cal_patience, lr=cal_lr, weight_decay=cal_wd)
        cal_logits_hts = hts.calibrate(logits)[test_idx_t].cpu()
        probs_hts = F.softmax(cal_logits_hts, dim=1)

        hts_acc = (probs_hts.argmax(dim=1) == test_labels.cpu()).float().mean().item()
        if abs(hts_acc - uncal_acc) > 1e-6:
            print(f"  WARNING: HTS changed accuracy! {uncal_acc:.6f} -> {hts_acc:.6f}")

        metrics = compute_all_metrics(probs_hts, test_labels.cpu(), degrees)
        save_result(save_path, 'HTS', probs_hts, cal_logits_hts, test_labels.cpu(),
                    degrees, homophily_test, metrics)
        results['HTS'] = metrics

    # 7. CaGCN
    if should_run('CaGCN'):
        set_seed(seed)  # Reset seed for reproducibility
        cagcn = CaGCN(num_classes, device, dropout=cal_dropout)
        cagcn.fit(logits, labels, g, val_idx_t, train_idx_t, epochs=cal_epochs, patience=cal_patience, lr=cal_lr, weight_decay=cal_wd)
        cal_logits_cagcn = cagcn.calibrate(logits)[test_idx_t].cpu()
        probs_cagcn = F.softmax(cal_logits_cagcn, dim=1)

        cagcn_acc = (probs_cagcn.argmax(dim=1) == test_labels.cpu()).float().mean().item()
        if abs(cagcn_acc - uncal_acc) > 1e-4:
            print(f"  WARNING: CaGCN changed accuracy! {uncal_acc:.6f} -> {cagcn_acc:.6f}")

        metrics = compute_all_metrics(probs_cagcn, test_labels.cpu(), degrees)
        save_result(save_path, 'CaGCN', probs_cagcn, cal_logits_cagcn, test_labels.cpu(),
                    degrees, homophily_test, metrics)
        results['CaGCN'] = metrics

    # 8. GATS (needs edge_index in PyG format)
    if should_run('GATS'):
        try:
            set_seed(seed)  # Reset seed for reproducibility
            src, dst = g.edges()
            edge_index = torch.stack([src, dst], dim=0).to(device)

            gats_heads = cal_cfg.get('heads', 8)
            gats_bias = cal_cfg.get('bias', 1)
            gats = GATS(num_classes, edge_index, g.num_nodes(), train_idx, device, heads=gats_heads, bias=gats_bias)
            gats.fit(logits, labels, edge_index, val_idx_t, train_idx_t, epochs=cal_epochs, patience=cal_patience, lr=cal_lr, weight_decay=cal_wd)
            cal_logits_gats = gats.calibrate(logits)[test_idx_t].cpu()
            probs_gats = F.softmax(cal_logits_gats, dim=1)

            gats_acc = (probs_gats.argmax(dim=1) == test_labels.cpu()).float().mean().item()
            if abs(gats_acc - uncal_acc) > 1e-6:
                print(f"  WARNING: GATS changed accuracy! {uncal_acc:.6f} -> {gats_acc:.6f}")

            metrics = compute_all_metrics(probs_gats, test_labels.cpu(), degrees)
            save_result(save_path, 'GATS', probs_gats, cal_logits_gats, test_labels.cpu(),
                        degrees, homophily_test, metrics)
            results['GATS'] = metrics
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
                raise
            print(f"  OOM in GATS — skipping this seed. ({type(e).__name__}: {e})")
            torch.cuda.empty_cache()
            results['GATS'] = {'error': 'OOM'}

    # 9. GETS (may change predictions like VS)
    if should_run('GETS'):
        try:
            set_seed(seed)  # Reset seed for reproducibility
            # Load GETS-specific config
            gets_config = load_config(dataset, calibrator='GETS')
            gets_cal_cfg = gets_config.get('calibration', {})
            gets_hidden_dim = gets_cal_cfg.get('hidden_dim', 16)
            gets_dropout = gets_cal_cfg.get('cal_dropout', 0.5)
            gets_num_layers = gets_cal_cfg.get('cal_num_layer', 2)
            gets_expert_select = gets_cal_cfg.get('expert_select', 2)
            gets_feature_hidden_dim = gets_cal_cfg.get('feature_hidden_dim', 16)
            gets_degree_hidden_dim = gets_cal_cfg.get('degree_hidden_dim', 16)
            gets_noisy_gating = gets_cal_cfg.get('noisy_gating', True)
            gets_coef = gets_cal_cfg.get('coef', 1.0)
            gets_lr = float(gets_cal_cfg.get('cal_lr', 0.01))
            gets_wd = float(gets_cal_cfg.get('cal_weight_decay', 0))
            gets_epochs = gets_cal_cfg.get('epochs', 1000)
            gets_patience = gets_cal_cfg.get('patience', 50)

            gets = GETS(num_classes, features.shape[1], device,
                        hidden_dim=gets_hidden_dim, dropout=gets_dropout,
                        num_layers=gets_num_layers, expert_select=gets_expert_select,
                        feature_hidden_dim=gets_feature_hidden_dim,
                        degree_hidden_dim=gets_degree_hidden_dim,
                        noisy_gating=gets_noisy_gating, loss_coef=gets_coef)
            gets.fit(logits, labels, g, features, val_idx_t, train_idx_t,
                     epochs=gets_epochs, patience=gets_patience, lr=gets_lr, weight_decay=gets_wd)
            cal_logits_gets = gets.calibrate(logits)[test_idx_t].cpu()
            probs_gets = F.softmax(cal_logits_gets, dim=1)
            # Note: GETS may change predictions (like VS), so no accuracy assertion
            metrics = compute_all_metrics(probs_gets, test_labels.cpu(), degrees)
            save_result(save_path, 'GETS', probs_gets, cal_logits_gets, test_labels.cpu(),
                        degrees, homophily_test, metrics)
            results['GETS'] = metrics
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
                raise
            print(f"  OOM in GETS — skipping this seed. ({type(e).__name__}: {e})")
            torch.cuda.empty_cache()
            results['GETS'] = {'error': 'OOM'}

    # 10. WATS (pure calibrator — wavelet-based temperature)
    if should_run('WATS'):
        set_seed(seed)  # Reset seed for reproducibility
        # Load WATS-specific config
        wats_config = load_config(dataset, calibrator='WATS')
        wats_cal_cfg = wats_config.get('calibration', {})
        wats_hidden_dim = wats_cal_cfg.get('cal_hidden_dim', 16)
        wats_dropout = wats_cal_cfg.get('cal_dropout', 0.4)
        wats_cheb = wats_cal_cfg.get('chebyshev_order', 2)
        wats_scale = wats_cal_cfg.get('wavelet_scale', 1.2)
        wats_lr = float(wats_cal_cfg.get('cal_lr', cal_lr))
        wats_wd = float(wats_cal_cfg.get('cal_weight_decay', cal_wd))
        wats_epochs = wats_cal_cfg.get('epochs', cal_epochs)
        wats_patience = wats_cal_cfg.get('patience', cal_patience)

        wats = WATS(num_classes, features.shape[1], device,
                    hidden_dim=wats_hidden_dim,
                    chebyshev_order=wats_cheb,
                    wavelet_scale=wats_scale,
                    cal_dropout=wats_dropout)
        wats.fit(logits, labels, g, features, val_idx_t, train_idx_t,
                 epochs=wats_epochs, patience=wats_patience,
                 lr=wats_lr, weight_decay=wats_wd)
        cal_logits_wats = wats.calibrate(logits)[test_idx_t].cpu()
        probs_wats = F.softmax(cal_logits_wats, dim=1)

        wats_acc = (probs_wats.argmax(dim=1) == test_labels.cpu()).float().mean().item()
        if abs(wats_acc - uncal_acc) > 1e-6:
            print(f"  WARNING: WATS changed accuracy! {uncal_acc:.6f} -> {wats_acc:.6f}")

        metrics = compute_all_metrics(probs_wats, test_labels.cpu(), degrees)
        wats_info = {
            'hidden_dim': wats_hidden_dim,
            'chebyshev_order': wats_cheb,
            'wavelet_scale': wats_scale,
        }
        save_result(save_path, 'WATS', probs_wats, cal_logits_wats, test_labels.cpu(),
                    degrees, homophily_test, metrics, wats_info)
        results['WATS'] = metrics

    # 11. HoTS
    if should_run('HoTS'):
        set_seed(seed)
        hots_config = load_config(dataset, calibrator='HoTS')
        hidden_dim = hots_config.get('calibration', {}).get('hidden_dim', 32)
        hots = HoTS(num_classes, features.shape[1], device, hidden_dim=hidden_dim)
        hots.fit(logits, labels, g, features, val_idx_t, train_idx_t,
                 epochs=cal_epochs, patience=cal_patience, lr=cal_lr, weight_decay=cal_wd)
        torch.save({
            'estimated_homophily': hots.homophily.detach().cpu(),
            'h_conv1': {k: v.detach().cpu() for k, v in hots.h_conv1.state_dict().items()},
            'h_conv2': {k: v.detach().cpu() for k, v in hots.h_conv2.state_dict().items()},
            'T_base': hots.T_base.detach().cpu(),
            'beta_raw': hots.alpha.detach().cpu(),
            'alpha_raw': hots.log_alpha_exp.detach().cpu(),
            'hidden_dim': hidden_dim,
            'alpha_floor': hots.FLOOR,
        }, save_path / 'HoTS_predictor_and_estimates.pt')
        cal_logits_hots = hots.calibrate(logits)[test_idx_t].cpu()
        probs_hots = F.softmax(cal_logits_hots, dim=1)
        hots_acc = (probs_hots.argmax(dim=1) == test_labels.cpu()).float().mean().item()
        if abs(hots_acc - uncal_acc) > 1e-6:
            print(f"  WARNING: HoTS changed accuracy! {uncal_acc:.6f} -> {hots_acc:.6f}")
        metrics = compute_all_metrics(probs_hots, test_labels.cpu(), degrees)
        info = hots.get_params()
        info['hidden_dim'] = hidden_dim
        save_result(save_path, 'HoTS', probs_hots, cal_logits_hots, test_labels.cpu(),
                    degrees, homophily_test, metrics, info)
        results['HoTS'] = metrics

    # Print summary
    if 'Uncal' in results:
        print(f"  Uncal ECE: {results['Uncal']['ece']*100:.2f}%, Acc: {results['Uncal']['acc']*100:.2f}%")
    if 'TS' in results and 'HoTS' in results:
        print(f"  TS ECE: {results['TS']['ece']*100:.2f}%, HoTS ECE: {results['HoTS']['ece']*100:.2f}%")
    if 'CaGCN' in results and 'ece' in results.get('GATS', {}):
        print(f"  CaGCN ECE: {results['CaGCN']['ece']*100:.2f}%, GATS ECE: {results['GATS']['ece']*100:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Unified GNN Calibration Experiment')
    parser.add_argument('--dataset', type=str, default='Cora',
                        help='Dataset name or "all" for all datasets')
    parser.add_argument('--model', type=str, default='GCN',
                        help='Model type or "all" for all models')
    parser.add_argument('--seeds', type=str, default='0-9',
                        help='Seeds (e.g., "0-9" or "0,1,2")')
    parser.add_argument('--save_dir', type=str, default='results/main')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--worker', type=int, default=None,
                        help='Worker ID for parallel execution (0-7)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Total number of workers')
    parser.add_argument('--calibrator', type=str, default='all',
                        help='"all"/"main": baselines + HoTS; or comma-separated names (e.g., Uncal,TS,HoTS)')
    parser.add_argument('--early_stop', type=str, default='loss', choices=['acc', 'loss'],
                        help='Early stopping criterion: "loss" (paper default) or "acc" (GETS style)')
    parser.add_argument('--base_training', type=str, default='erm', choices=['erm', 'dcgc'],
                        help='"erm" (default) or "dcgc": apply DCGC base correction before downstream calibrators')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable torch deterministic algorithms so repeated runs of the same '
                             'seed reproduce exactly. Off by default: scatter_add_ in the homophily '
                             'target is nondeterministic on CUDA, which makes single-run ECE vary '
                             'run-to-run at fixed seed.')
    parser.add_argument('--split', type=str, default='random', choices=['random', 'official'],
                        help='"random" (default 20/10/70 per seed) or "official" (dataset official split; ogbn-arxiv chronological)')
    # Synthetic SBM args (used when --dataset synthetic)
    parser.add_argument('--synthetic_h', type=float, default=0.7,
                        help='Target graph homophily for synthetic SBM')
    parser.add_argument('--synthetic_C', type=int, default=5,
                        help='Number of classes for synthetic SBM')
    parser.add_argument('--synthetic_n', type=int, default=2000,
                        help='Number of nodes for synthetic SBM')
    parser.add_argument('--synthetic_d', type=float, default=10.0,
                        help='Target average degree for synthetic SBM')
    parser.add_argument('--synthetic_mu', type=float, default=2.0,
                        help='Class feature separation (mu)')
    parser.add_argument('--synthetic_sigma', type=float, default=1.0,
                        help='Feature noise std (sigma)')
    parser.add_argument('--synthetic_dim', type=int, default=0,
                        help='Feature dimension (0 = auto = max(C, 16))')
    parser.add_argument('--synthetic_h_per_class', type=str, default='',
                        help='Comma-separated per-class homophily (e.g., "0.1,0.3,0.5,0.7,0.9"). '
                             'Overrides --synthetic_h when set. Length must equal --synthetic_C.')
    parser.add_argument('--synthetic_sigma_hetero', action='store_true',
                        help='Per-node feature noise heterogeneity: half nodes sigma*0.5, half sigma*2.0')
    args = parser.parse_args()

    # GPU already set at top of file before torch import
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} (GPU {args.gpu})")

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("Deterministic algorithms ENABLED")

    # Parse arguments
    if args.dataset.lower() == 'all':
        datasets = DATASETS
    elif ',' in args.dataset:
        datasets = [d.strip() for d in args.dataset.split(',')]
    else:
        datasets = [args.dataset]

    if args.model.lower() == 'all':
        models = MODELS
    elif ',' in args.model:
        models = [m.strip() for m in args.model.split(',')]
    else:
        models = [args.model]

    seeds = parse_seeds(args.seeds)

    # Create all tasks
    all_tasks = []
    for dataset in datasets:
        for model in models:
            for seed in seeds:
                all_tasks.append((dataset, model, seed))

    # Filter tasks for this worker
    if args.worker is not None:
        my_tasks = [t for i, t in enumerate(all_tasks) if i % args.num_workers == args.worker]
        print(f"Worker {args.worker}/{args.num_workers}: {len(my_tasks)}/{len(all_tasks)} tasks")
    else:
        my_tasks = all_tasks

    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Save dir: {args.save_dir}")
    print(f"Total tasks: {len(my_tasks)}")

    # Parse calibrators
    calibrators = resolve_calibrators([c.strip() for c in args.calibrator.split(',')])

    print(f"Calibrators: {calibrators}")

    # Build synthetic config (used only when dataset == 'synthetic')
    synthetic_cfg = {
        'h': args.synthetic_h,
        'C': args.synthetic_C,
        'n': args.synthetic_n,
        'd': args.synthetic_d,
        'mu': args.synthetic_mu,
        'sigma': args.synthetic_sigma,
        'dim': args.synthetic_dim if args.synthetic_dim > 0 else max(args.synthetic_C, 16),
        'sigma_hetero': args.synthetic_sigma_hetero,
    }
    if args.synthetic_h_per_class.strip():
        h_list = [float(x) for x in args.synthetic_h_per_class.split(',')]
        if len(h_list) != args.synthetic_C:
            raise ValueError(
                f"--synthetic_h_per_class has {len(h_list)} entries, expected C={args.synthetic_C}"
            )
        synthetic_cfg['h_per_class'] = h_list
    if any(d.lower() == 'synthetic' for d in datasets):
        print(f"Synthetic config: {synthetic_cfg}")

    # Run experiments
    failures = []
    for i, (dataset, model, seed) in enumerate(my_tasks):
        print(f"\n[{i+1}/{len(my_tasks)}] Dataset: {dataset}, Model: {model}, Seed: {seed}")
        try:
            run_single_experiment(dataset, model, seed, args.save_dir, device,
                                  calibrators, args.early_stop,
                                  synthetic_cfg=synthetic_cfg if dataset.lower() == 'synthetic' else None,
                                  base_training=args.base_training, split=args.split)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            failures.append((dataset, model, seed))
    if failures:
        print(f'Failed runs: {failures}')
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
