"""
Ensemble Temperature Scaling (ETS) - GETS-main style implementation.

From: GETS-main/GETS-main/model/calibrator.py

API:
    ets = ETS(num_classes, device)
    ets.fit(logits, labels, val_idx, train_idx, epochs, patience, lr)
    calibrated_logits = ets.calibrate(logits)

Based on: Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration
"""

import numpy as np
import scipy.optimize
from scipy.special import softmax
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ts import TS


class ETS(nn.Module):
    """
    Ensemble Temperature Scaling: weighted combination of TS, uncalibrated, and uniform.

    p = w1 * softmax(logits/T) + w2 * softmax(logits) + w3 * (1/num_classes)

    From: GETS-main/GETS-main/model/calibrator.py
    """

    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.temp_model = TS(num_classes, device)
        self.w1 = 1.0
        self.w2 = 0.0
        self.w3 = 0.0

    def forward(self, logits):
        """Return calibrated log-probabilities."""
        temp = self.temp_model.temperature
        p = (self.w1 * F.softmax(logits / temp, dim=1) +
             self.w2 * F.softmax(logits, dim=1) +
             self.w3 * (1.0 / self.num_classes))
        return torch.log(p + 1e-10)

    def calibrate(self, logits):
        """Calibrate pre-computed logits."""
        return self.forward(logits)

    def fit(self, logits, labels, val_idx, train_idx, epochs=1000, patience=50, lr=0.01, weight_decay=0):
        """Fit ETS on calibration set."""
        self.to(self.device)

        # Step 1: Fit temperature scaling first
        self.temp_model.fit(logits, labels, val_idx, train_idx, epochs, patience, lr, weight_decay)

        # Step 2: Find optimal weights using scipy optimization
        # Use val_idx (calibration training set) for weight optimization
        cal_logits = logits[val_idx].detach().cpu().numpy()
        cal_labels = labels[val_idx].detach().cpu().numpy()

        # Create one-hot labels
        one_hot = np.zeros((len(cal_labels), self.num_classes))
        one_hot[np.arange(len(cal_labels)), cal_labels] = 1

        temp = self.temp_model.temperature.detach().cpu().numpy()
        w = self._ensemble_scaling(cal_logits, one_hot, temp)
        self.w1, self.w2, self.w3 = w[0], w[1], w[2]

        return self

    def _ensemble_scaling(self, logit, label, t):
        """
        Official ETS implementation from Mix-n-Match paper.
        Code taken from (https://github.com/zhang64-llnl/Mix-n-Match-Calibration)
        """
        # p1: uncalibrated softmax
        p1 = softmax(logit, axis=1)

        # p0: temperature scaled softmax
        logit_scaled = logit / t
        p0 = softmax(logit_scaled, axis=1)

        # p2: uniform distribution
        p2 = np.ones_like(p0) / self.num_classes

        # Optimize weights
        bnds_w = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))

        def constraint_fun(x):
            return np.sum(x) - 1

        constraints = {"type": "eq", "fun": constraint_fun}

        w = scipy.optimize.minimize(
            self._ll_w, (1.0, 0.0, 0.0),
            args=(p0, p1, p2, label),
            method='SLSQP',
            constraints=constraints,
            bounds=bnds_w,
            tol=1e-12,
            options={'disp': False}
        )
        return w.x

    @staticmethod
    def _ll_w(w, *args):
        """Cross-entropy loss for weight optimization."""
        p0, p1, p2, label = args
        p = w[0] * p0 + w[1] * p1 + w[2] * p2
        N = p.shape[0]
        ce = -np.sum(label * np.log(p + 1e-10)) / N
        return ce
