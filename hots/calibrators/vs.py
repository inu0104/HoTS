"""
Vector Scaling (VS) - GETS-main style implementation.

From: GETS-main/GETS-main/model/calibrator.py

API:
    vs = VS(num_classes, device)
    vs.fit(logits, labels, val_idx, train_idx, epochs, patience, lr)
    calibrated_logits = vs.calibrate(logits)

Note: VS may change predictions (not pure calibration).
"""

import torch
import torch.nn as nn

from .ts import fit_calibration


class VS(nn.Module):
    """
    Vector Scaling: learns per-class temperature and bias.

    calibrated_logits = logits * temperature + bias

    From: GETS-main/GETS-main/model/calibrator.py
    """

    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.ones(num_classes, device=device))
        self.bias = nn.Parameter(torch.ones(num_classes, device=device))
        self.device = device

    def vector_scale(self, logits):
        """Expand temperature to match logits shape."""
        # [C] -> [N, C]
        return self.temperature.unsqueeze(0).expand(logits.size(0), logits.size(1))

    def forward(self, logits):
        """Calibrate logits."""
        temperature = self.vector_scale(logits)
        return logits * temperature + self.bias

    def calibrate(self, logits):
        """Calibrate pre-computed logits."""
        return self.forward(logits)

    def fit(self, logits, labels, val_idx, train_idx, epochs=1000, patience=50, lr=0.01, weight_decay=0):
        """Fit vector scaling on calibration set."""
        self.to(self.device)

        def eval_fn(logits):
            temperature = self.vector_scale(logits)
            return logits * temperature + self.bias

        fit_calibration(self, eval_fn, logits, labels, val_idx, train_idx,
                       epochs=epochs, patience=patience, lr=lr, weight_decay=weight_decay)
        return self
