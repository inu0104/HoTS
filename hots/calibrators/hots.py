"""
HoTS: Homophily-aware Temperature Scaling.

Per-node temperature
    T_i = T_base + beta * sqrt(2 K log K (1 - e_i)) / (|h_tilde_i| + eps)^alpha

Three learnable scalar parameters: T_base > 0, beta > 0, alpha > 0,
all enforced through softplus parameterizations.

The final main protocol uses only observed train/validation neighbors for
predictor targets, excludes self-loops from these targets, and uses exponent
floor 0.01. Temperature fitting and the original state-dict keys are preserved.

This module is self-contained: it bundles the homophily predictor (a 2-layer
GCN), the entropy/homophily helpers, and the calibration fit loop.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn as dglnn


class HoTS(nn.Module):
    EPS = 0.02
    FLOOR = 0.01

    def __init__(self, num_classes, feature_dim, device, hidden_dim=32,
                 init_T_base=0.5, init_alpha=0.5, init_gamma=0.5,
                 init_log_alpha_exp=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.hidden_dim = hidden_dim

        # Learnable scalar parameters
        self.T_base = nn.Parameter(torch.tensor([init_T_base], device=device))
        self.alpha = nn.Parameter(torch.tensor([init_alpha], device=device))
        self.gamma = nn.Parameter(torch.tensor([init_gamma], device=device))
        self.log_alpha_exp = nn.Parameter(
            torch.tensor([init_log_alpha_exp], device=device))

        # Homophily predictor (2-layer GCN)
        self.h_conv1 = dglnn.GraphConv(feature_dim, hidden_dim)
        self.h_conv2 = dglnn.GraphConv(hidden_dim, 1)

        self.g = None
        self.features = None
        self.register_buffer('gt_homophily', None)
        self.register_buffer('homophily', None)
        self.register_buffer('entropy_norm', None)

    # -------- Helpers --------

    def _predict_homophily(self, g, features):
        h = F.relu(self.h_conv1(g, features))
        h = torch.sigmoid(self.h_conv2(g, h)).squeeze()
        return h

    def _compute_gt_homophily_masked(self, g, labels, observed_mask):
        """Observed-label-only homophily target:
        N_L(i) = { j in N(i) | j != i (self-loop removed), j in I_known };
        h^gt_i = mean_{j in N_L(i)} 1{y_j == y_i} defined only when |N_L(i)| > 0.
        Returns (gt, valid) where valid[i] = (|N_L(i)| > 0).
        """
        num_nodes = g.num_nodes()
        src, dst = g.edges()
        src, dst = src.to(self.device).long(), dst.to(self.device).long()
        labels = labels.to(self.device)
        obs = observed_mask.to(self.device).float()

        # Remove self-loops (j != i).
        non_self = src != dst
        src, dst = src[non_self], dst[non_self]

        # Keep an edge (i<-j) only when the neighbor j has an observed label.
        dst_obs = obs[dst]
        same_label = (labels[src] == labels[dst]).float() * dst_obs

        same_count = torch.zeros(num_nodes, device=self.device)
        same_count.scatter_add_(0, src, same_label)

        obs_degree = torch.zeros(num_nodes, device=self.device)
        obs_degree.scatter_add_(0, src, dst_obs)

        valid = obs_degree > 0
        gt = torch.zeros(num_nodes, device=self.device)
        gt[valid] = same_count[valid] / obs_degree[valid]
        return gt, valid

    def _train_homophily_predictor(self, g, features, labels, train_idx, val_idx,
                                   epochs=200, lr=0.01):
        optimizer = optim.Adam(
            list(self.h_conv1.parameters()) + list(self.h_conv2.parameters()),
            lr=lr, weight_decay=1e-4,
        )

        num_nodes = g.num_nodes()
        known_idx = torch.cat([train_idx, val_idx]).to(self.device)
        observed_mask = torch.zeros(num_nodes, dtype=torch.bool, device=self.device)
        observed_mask[known_idx] = True

        # Predictor targets use observed non-self neighbors only.
        gt_homophily, valid = self._compute_gt_homophily_masked(g, labels, observed_mask)
        # M = supervised (known) nodes that have >= 1 observed non-self neighbor.
        known_valid = known_idx[valid[known_idx]]
        if known_valid.numel() == 0:
            raise ValueError("No observed nodes have an observed non-self neighbor")

        best_loss = float('inf')
        patience = 30
        patience_counter = 0
        best_state = None

        for _ in range(epochs):
            self.h_conv1.train()
            self.h_conv2.train()
            optimizer.zero_grad()

            pred_h = self._predict_homophily(g, features)
            loss = F.mse_loss(pred_h[known_valid], gt_homophily[known_valid])
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
                best_state = {
                    'h_conv1': {k: v.clone() for k, v in self.h_conv1.state_dict().items()},
                    'h_conv2': {k: v.clone() for k, v in self.h_conv2.state_dict().items()},
                }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            self.h_conv1.load_state_dict(best_state['h_conv1'])
            self.h_conv2.load_state_dict(best_state['h_conv2'])

        return gt_homophily

    def _compute_entropy(self, logits):
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        max_entropy = np.log(self.num_classes)
        return entropy / max_entropy

    # -------- Temperature / forward --------

    def compute_temperature(self):
        """T = T_base + β√(2K logK (1-e)) / |h̃|^α"""
        T_base = F.softplus(self.T_base) + 0.1
        beta = F.softplus(self.alpha) + 0.01
        alpha_exp = F.softplus(self.log_alpha_exp) + self.FLOOR

        baseline = 1.0 / self.num_classes
        h_tilde = (self.homophily - baseline) / (1.0 - baseline)
        h_abs = torch.abs(h_tilde) + self.EPS
        h_pow = torch.exp(alpha_exp * torch.log(h_abs))

        coeff = 2 * self.num_classes * math.log(self.num_classes)
        one_minus_e = (1.0 - self.entropy_norm).clamp(min=0.001)
        ratio_term = beta * torch.sqrt(coeff * one_minus_e) / h_pow

        temperature = T_base + ratio_term
        return torch.clamp(temperature, min=0.05, max=50.0).unsqueeze(1)

    def forward(self, logits):
        temperature = self.compute_temperature()
        return logits / temperature

    def calibrate(self, logits):
        return self.forward(logits)

    # -------- Fit --------

    def fit(self, logits, labels, g, features, val_idx, train_idx,
            epochs=1000, patience=50, lr=0.01, weight_decay=0,
            hp_epochs=200, hp_lr=0.01):
        self.g = g
        self.features = features
        self.to(self.device)

        gt_homophily = self._train_homophily_predictor(
            g, features, labels, train_idx, val_idx, hp_epochs, hp_lr)
        self.gt_homophily = gt_homophily

        with torch.no_grad():
            self.h_conv1.eval()
            self.h_conv2.eval()
            self.homophily = self._predict_homophily(g, features)
            self.entropy_norm = self._compute_entropy(logits)

        optimizer = optim.Adam(
            [self.T_base, self.alpha, self.log_alpha_exp],
            lr=lr, weight_decay=weight_decay)

        vlss_mn = float('Inf')
        curr_step = 0
        best_state = None
        for _ in range(epochs):
            self.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits)[val_idx], labels[val_idx])
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                self.eval()
                ml = F.cross_entropy(self.forward(logits)[train_idx], labels[train_idx])
                if ml <= vlss_mn:
                    vlss_mn = ml.item()
                    curr_step = 0
                    best_state = {
                        'T_base': self.T_base.clone(),
                        'alpha': self.alpha.clone(),
                        'log_alpha_exp': self.log_alpha_exp.clone(),
                    }
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break
        if best_state:
            self.T_base.data = best_state['T_base']
            self.alpha.data = best_state['alpha']
            self.log_alpha_exp.data = best_state['log_alpha_exp']
        return self

    def get_params(self):
        return {
            "T_base": (F.softplus(self.T_base) + 0.1).item(),
            "beta": (F.softplus(self.alpha) + 0.01).item(),
            "alpha_exp": (F.softplus(self.log_alpha_exp) + self.FLOOR).item(),
            "n_params": 3,
            "alpha_floor": self.FLOOR,
            "homophily_source": "observed-label-only predictor",
            "form": "T_base + β√(2KlogK(1-e)) / |h̃|^α",
        }
