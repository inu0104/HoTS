"""
Temperature Scaling (TS) - GETS-main style implementation.

From: GETS-main/GETS-main/model/calibrator.py

API:
    ts = TS(num_classes, device)
    ts.fit(logits, labels, val_idx, train_idx, epochs, patience, lr)
    calibrated_logits = ts.calibrate(logits)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def fit_calibration(calibrator, eval_fn, logits, labels, val_idx, train_idx,
                    epochs=1000, patience=50, lr=0.01, weight_decay=0):
    """
    Fit calibration model with early stopping.

    GETS-main convention:
    - val_idx: calibration training set (original validation set)
    - train_idx: early stopping validation set (original training set)

    Args:
        calibrator: Calibration model with parameters to optimize
        eval_fn: Function that takes logits and returns calibrated logits
        logits: Pre-computed logits from frozen GNN [N, C]
        labels: Node labels [N]
        val_idx: Indices for calibration training
        train_idx: Indices for early stopping validation
        epochs: Maximum epochs
        patience: Early stopping patience
        lr: Learning rate
        weight_decay: Weight decay
    """
    optimizer = optim.Adam(calibrator.parameters(), lr=lr, weight_decay=weight_decay)

    vlss_mn = float('Inf')
    curr_step = 0
    best_state = None

    for epoch in range(epochs):
        calibrator.train()
        optimizer.zero_grad()

        # Forward pass
        ret = eval_fn(logits)
        if isinstance(ret, tuple):
            calibrated, aux_loss, _ = ret
        else:
            calibrated = ret
            aux_loss = 0

        # Loss on calibration training set (val_idx)
        loss = F.cross_entropy(calibrated[val_idx], labels[val_idx])
        if isinstance(aux_loss, torch.Tensor):
            loss = loss + aux_loss

        loss.backward()
        optimizer.step()

        # Validation (early stopping on train_idx)
        with torch.no_grad():
            calibrator.eval()
            ret = eval_fn(logits)
            if isinstance(ret, tuple):
                calibrated, _, _ = ret
            else:
                calibrated = ret

            val_loss = F.cross_entropy(calibrated[train_idx], labels[train_idx])

            if val_loss <= vlss_mn:
                vlss_mn = val_loss.item()
                curr_step = 0
                best_state = copy.deepcopy(calibrator.state_dict())
            else:
                curr_step += 1
                if curr_step >= patience:
                    break

    # Restore best state
    if best_state is not None:
        calibrator.load_state_dict(best_state)

    return calibrator


class TS(nn.Module):
    """
    Temperature Scaling: learns a single scalar temperature.

    calibrated_logits = logits / temperature

    From: GETS-main/GETS-main/model/calibrator.py
    """

    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.ones(1, device=device))
        self.device = device

    def temperature_scale(self, logits):
        """Expand temperature to match logits shape."""
        # [1] -> [N, C]
        return self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))

    def forward(self, logits):
        """Calibrate logits."""
        temperature = self.temperature_scale(logits)
        return logits / temperature

    def calibrate(self, logits):
        """Calibrate pre-computed logits."""
        return self.forward(logits)

    def fit(self, logits, labels, val_idx, train_idx, epochs=1000, patience=50, lr=0.01, weight_decay=0):
        """Fit temperature on calibration set."""
        self.to(self.device)

        def eval_fn(logits):
            temperature = self.temperature_scale(logits)
            return logits / temperature

        fit_calibration(self, eval_fn, logits, labels, val_idx, train_idx,
                       epochs=epochs, patience=patience, lr=lr, weight_decay=weight_decay)
        return self
