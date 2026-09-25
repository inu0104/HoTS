"""
HTS: Entropy-based Temperature Scaling.

Based on: "Adaptive Temperature Scaling for Robust Calibration"

Formula: T = a + b × entropy_norm

Standalone calibrator that uses only entropy information from the logits.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class HTS(nn.Module):

    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

        # Parameters: T = a + b * entropy
        self.a = nn.Parameter(torch.tensor(1.0, device=device))
        self.b = nn.Parameter(torch.tensor(0.0, device=device))

        self.entropy_norm = None

    def _compute_entropy(self, logits):
        """Compute normalized entropy from logits."""
        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        max_entropy = np.log(self.num_classes)
        return entropy / max_entropy

    def compute_temperature(self):
        T_base = F.softplus(self.a) + 0.1
        temperature = T_base + self.b * self.entropy_norm
        temperature = torch.clamp(temperature, min=0.1)
        return temperature.unsqueeze(1)

    def forward(self, logits):
        temperature = self.compute_temperature()
        return logits / temperature

    def calibrate(self, logits):
        return self.forward(logits)

    def fit(self, logits, labels, val_idx, train_idx,
            epochs=1000, patience=50, lr=0.01, weight_decay=0):
        self.to(self.device)

        with torch.no_grad():
            self.entropy_norm = self._compute_entropy(logits)

        optimizer = optim.Adam([self.a, self.b], lr=lr, weight_decay=weight_decay)

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

            with torch.no_grad():
                self.eval()
                calibrated = self.forward(logits)
                val_loss = F.cross_entropy(calibrated[train_idx], labels[train_idx])

                if val_loss <= vlss_mn:
                    vlss_mn = val_loss.item()
                    curr_step = 0
                    best_state = {'a': self.a.clone(), 'b': self.b.clone()}
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break

        if best_state:
            self.a.data = best_state['a']
            self.b.data = best_state['b']

        return self

    def get_params(self):
        return {
            "a": (F.softplus(self.a) + 0.1).item(),
            "b": self.b.item(),
        }
