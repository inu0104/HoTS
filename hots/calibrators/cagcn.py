"""
CaGCN (Calibration GCN) - GETS-main style implementation.

From: GETS-main/GETS-main/model/calibrator.py

API:
    cagcn = CaGCN(num_classes, device, dropout=0.5)
    cagcn.fit(logits, labels, g, val_idx, train_idx, epochs, patience, lr)
    calibrated_logits = cagcn.calibrate(logits)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn as dglnn


class GCN_Temp(nn.Module):
    """GCN for temperature prediction."""

    def __init__(self, in_channels, num_classes, num_hidden, drop_rate, num_layers):
        super().__init__()
        self.drop_rate = drop_rate
        self.feature_list = [in_channels, num_hidden, num_classes]
        for _ in range(num_layers - 2):
            self.feature_list.insert(-1, num_hidden)

        layer_list = []
        for i in range(len(self.feature_list) - 1):
            layer_list.append(
                ["conv" + str(i + 1),
                 dglnn.GraphConv(self.feature_list[i], self.feature_list[i + 1])]
            )
        self.layer_list = nn.ModuleDict(layer_list)

    def forward(self, features, g):
        x = features
        for i in range(len(self.feature_list) - 1):
            x = self.layer_list["conv" + str(i + 1)](g, x)
            if i < len(self.feature_list) - 2:
                x = F.relu(x)
                x = F.dropout(x, self.drop_rate, self.training)
        return x


class CaGCN(nn.Module):
    """
    CaGCN: GCN-based node-wise temperature prediction.

    calibrated_logits = logits * softplus(temperature)

    From: GETS-main/GETS-main/model/calibrator.py
    """

    def __init__(self, num_classes, device, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        # GCN: input=num_classes (logits), output=1 (temperature)
        self.cagcn = GCN_Temp(num_classes, 1, 16, drop_rate=dropout, num_layers=2)
        self.g = None  # Will be set during fit

    def graph_temperature_scale(self, logits, g):
        """Compute node-wise temperature using GCN."""
        temperature = self.cagcn(logits, g)
        return temperature

    def forward(self, logits):
        """Calibrate logits."""
        temperature = self.graph_temperature_scale(logits, self.g)
        return logits * F.softplus(temperature)

    def calibrate(self, logits):
        """Calibrate pre-computed logits."""
        return self.forward(logits)

    def fit(self, logits, labels, g, val_idx, train_idx, epochs=1000, patience=50, lr=0.01, weight_decay=0):
        """Fit CaGCN on calibration set."""
        self.g = g
        self.to(self.device)

        optimizer = optim.Adam(self.cagcn.parameters(), lr=lr, weight_decay=weight_decay)

        vlss_mn = float('Inf')
        curr_step = 0
        best_state = None

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            # Calibrate
            temperature = self.graph_temperature_scale(logits, g)
            calibrated = logits * F.softplus(temperature)

            # Loss on calibration training set (val_idx)
            loss = F.cross_entropy(calibrated[val_idx], labels[val_idx])
            loss.backward()
            optimizer.step()

            # Validation (early stopping on train_idx)
            with torch.no_grad():
                self.eval()
                temperature = self.graph_temperature_scale(logits, g)
                calibrated = logits * F.softplus(temperature)
                val_loss = F.cross_entropy(calibrated[train_idx], labels[train_idx])

                if val_loss <= vlss_mn:
                    vlss_mn = val_loss.item()
                    curr_step = 0
                    best_state = copy.deepcopy(self.cagcn.state_dict())
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break

        # Restore best state
        if best_state is not None:
            self.cagcn.load_state_dict(best_state)

        return self
