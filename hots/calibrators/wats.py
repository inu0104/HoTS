"""
WATS (Wavelet-based Adaptive Temperature Scaling).

Based on: https://github.com/lxy1134/WATS

Uses graph wavelet features (Chebyshev-approximated heat kernel) to predict
node-wise temperatures via a 2-layer MLP. Prediction preserving.

API (matches HoTS codebase style):
    wats = WATS(num_classes, feature_dim, device, hidden_dim=16)
    wats.fit(logits, labels, g, features, val_idx, train_idx, epochs, patience, lr)
    calibrated_logits = wats.calibrate(logits)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csgraph
from scipy import integrate


class WATS(nn.Module):
    """
    WATS: Wavelet-based Adaptive Temperature Scaling.

    Extracts graph wavelet features via Chebyshev polynomial approximation
    of the heat kernel, then predicts per-node temperature with a 2-layer MLP.

    T_i = softplus(MLP(wavelet_features_i))
    calibrated_logits_i = logits_i / T_i
    """

    def __init__(self, num_classes, feature_dim, device, hidden_dim=16,
                 chebyshev_order=2, wavelet_scale=1.2, cal_dropout=0.4):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.chebyshev_order = chebyshev_order
        self.wavelet_scale = wavelet_scale

        # Wavelet features dimension: K+1
        wavelet_dim = chebyshev_order + 1

        # SimpleTempMLP: 2-layer MLP -> positive temperature
        self.fc1 = nn.Linear(wavelet_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(cal_dropout)
        self.softplus = nn.Softplus()

        # Buffers
        self.register_buffer('wavelet_features', None)

    def _compute_wavelet_features(self, g):
        """Compute graph wavelet features using Chebyshev approximation of heat kernel."""
        k = self.chebyshev_order
        s = self.wavelet_scale

        # Symmetric normalized Laplacian
        A = g.adj_external(scipy_fmt='csr').astype(float)
        L_scipy = csgraph.laplacian(A, normed=True)
        lam_max = 2.0  # spectral upper bound for normalized Laplacian

        N = g.num_nodes()
        L = torch.sparse_coo_tensor(
            torch.tensor([L_scipy.row, L_scipy.col]),
            torch.tensor(L_scipy.data, dtype=torch.float32),
            size=L_scipy.shape,
            device=self.device
        )

        # Rescale to [-1, 1]: L_hat = (2/lam_max)*L - I
        index = torch.arange(N, device=self.device)
        I_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([index, index]),
            values=torch.ones(N, device=self.device),
            size=(N, N)
        )
        L_hat = L.mul(2.0 / lam_max) - I_sparse

        # Initial signal: log(1 + degree)
        in_deg = g.in_degrees().float().to(self.device)
        out_deg = g.out_degrees().float().to(self.device)
        deg = torch.log1p((in_deg + out_deg).clamp(min=1e-6)).unsqueeze(1)  # [N, 1]

        # Chebyshev recurrence: T_0, T_1, ..., T_K
        x0 = deg
        T_prev2 = x0
        if k == 0:
            feats = T_prev2
        else:
            T_prev1 = torch.sparse.mm(L_hat, x0)
            feats = torch.cat([T_prev2, T_prev1], dim=1)
            for _ in range(2, k + 1):
                T_cur = 2 * torch.sparse.mm(L_hat, T_prev1) - T_prev2
                feats = torch.cat([feats, T_cur], dim=1)
                T_prev2, T_prev1 = T_prev1, T_cur

        # Chebyshev coefficients via numerical integration of heat kernel
        coeffs = self._compute_chebyshev_coeffs(k, s, lam_max)
        final_feats = feats * coeffs
        final_feats = F.normalize(final_feats, p=1, dim=1)

        return final_feats

    def _compute_chebyshev_coeffs(self, k, s, lam_max):
        """Compute Chebyshev expansion coefficients for heat kernel e^{-sx}."""
        a = lam_max / 2.0
        target_func = lambda theta, s, a: np.exp(-s * a * (np.cos(theta) + 1))

        coeffs = []
        for ki in range(k + 1):
            integrand = lambda theta, ki=ki: np.cos(ki * theta) * target_func(theta, s, a)
            integral_result, _ = integrate.quad(integrand, 0, np.pi)
            c_k = (2 / np.pi) * integral_result
            coeffs.append(c_k)

        return torch.tensor(coeffs, device=self.device, dtype=torch.float32)

    def _predict_temperature(self):
        """Predict per-node temperature from wavelet features."""
        h = F.relu(self.fc1(self.wavelet_features))
        h = self.dropout(h)
        h = self.fc2(h)
        T = self.softplus(h)  # [N, 1], positive
        return T

    def forward(self, logits):
        """Calibrate logits."""
        T = self._predict_temperature()
        return logits / (T + 1e-6)

    def calibrate(self, logits):
        """Calibrate pre-computed logits."""
        return self.forward(logits)

    def fit(self, logits, labels, g, features, val_idx, train_idx,
            epochs=1000, patience=50, lr=0.01, weight_decay=0,
            **kwargs):
        """Fit WATS on calibration set."""
        self.to(self.device)

        # Step 1: Extract wavelet features (frozen, not learned)
        with torch.no_grad():
            self.wavelet_features = self._compute_wavelet_features(g)

        # Step 2: Train MLP temperature predictor
        optimizer = optim.Adam(
            list(self.fc1.parameters()) + list(self.fc2.parameters()),
            lr=lr, weight_decay=weight_decay
        )

        vlss_mn = float('Inf')
        curr_step = 0
        best_state = None

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            calibrated = self.forward(logits)
            loss = F.cross_entropy(calibrated[val_idx], labels[val_idx])
            loss.backward()
            optimizer.step()

            # Validation
            with torch.no_grad():
                self.eval()
                calibrated = self.forward(logits)
                val_loss = F.cross_entropy(calibrated[train_idx], labels[train_idx])

                if val_loss <= vlss_mn:
                    vlss_mn = val_loss.item()
                    curr_step = 0
                    best_state = {
                        'fc1': copy.deepcopy(self.fc1.state_dict()),
                        'fc2': copy.deepcopy(self.fc2.state_dict()),
                    }
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break

        # Restore best state
        if best_state:
            self.fc1.load_state_dict(best_state['fc1'])
            self.fc2.load_state_dict(best_state['fc2'])

        return self

    def get_params(self):
        """Get summary of learned parameters."""
        with torch.no_grad():
            T = self._predict_temperature()
        return {
            "T_mean": T.mean().item(),
            "T_std": T.std().item(),
            "T_min": T.min().item(),
            "T_max": T.max().item(),
            "chebyshev_order": self.chebyshev_order,
            "wavelet_scale": self.wavelet_scale,
        }
