"""
GETS (Graph Expert Temperature Scaling) - Mixture of Experts calibrator.

From: GETS-main/GETS-main/model/GETS.py

API:
    gets = GETS(num_classes, feature_dim, device, hidden_dim=16, dropout=0.5,
                num_layers=2, expert_select=2, expert_configs=None)
    gets.fit(logits, labels, g, features, val_idx, train_idx, epochs, patience, lr)
    calibrated_logits = gets.calibrate(logits)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn as dglnn
from torch.distributions.normal import Normal


# =============================================================================
# Expert Networks (GCN, GAT, GIN backbones)
# =============================================================================

class GCN_GETS(nn.Module):
    """GCN-based expert for GETS."""

    def __init__(self, num_classes, hidden_dim, dropout_rate, num_layers,
                 device, expert_config, feature_dim, feature_hidden_dim, degree_hidden_dim):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device

        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim

        self.feature_list = [in_channels, hidden_dim, num_classes]
        for _ in range(num_layers - 2):
            self.feature_list.insert(-1, hidden_dim)

        layer_list = []
        for i in range(len(self.feature_list) - 1):
            layer_list.append(
                ["conv" + str(i + 1),
                 dglnn.GraphConv(self.feature_list[i], self.feature_list[i + 1])]
            )
        self.layer_list = nn.ModuleDict(layer_list)
        self.degree_dim = degree_hidden_dim

    def forward(self, g, logits, features):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features = self.proj_feature(features)
            inputs.append(features)
        if "degrees" in self.expert_config:
            if not hasattr(self, "degrees"):
                degrees = g.in_degrees() + g.out_degrees()
                max_degree = degrees.max() + 1
                self.degree_embedder = nn.Embedding(
                    num_embeddings=max_degree, embedding_dim=self.degree_dim
                ).to(self.device)
                self.degrees = degrees.unsqueeze(-1)
            degree_embeds = self.degree_embedder(self.degrees.squeeze(-1))
            inputs.append(degree_embeds)

        x = torch.cat(inputs, dim=-1)
        for i in range(len(self.feature_list) - 1):
            x = self.layer_list["conv" + str(i + 1)](g, x)
            if i < len(self.feature_list) - 2:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, self.training)
        return x


class GAT_GETS(nn.Module):
    """GAT-based expert for GETS."""

    def __init__(self, num_classes, hidden_dim, dropout_rate, num_layers,
                 device, expert_config, feature_dim, feature_hidden_dim, degree_hidden_dim,
                 num_heads=2):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device
        self.num_heads = num_heads

        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim

        self.feature_list = [in_channels] + [hidden_dim] * (num_layers - 1)
        layer_list = []
        for i in range(len(self.feature_list) - 1):
            layer_list.append(
                ("conv" + str(i + 1),
                 dglnn.GATConv(self.feature_list[i], self.feature_list[i + 1] // num_heads,
                               num_heads=num_heads))
            )
        self.layer_list = nn.ModuleDict(layer_list)
        self.degree_dim = degree_hidden_dim
        self.final_proj = nn.Linear(hidden_dim, num_classes)

    def forward(self, g, logits, features):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features = self.proj_feature(features)
            inputs.append(features)
        if "degrees" in self.expert_config:
            if not hasattr(self, "degrees"):
                degrees = g.in_degrees() + g.out_degrees()
                max_degree = degrees.max().item() + 1
                self.degree_embedder = nn.Embedding(
                    num_embeddings=max_degree, embedding_dim=self.degree_dim
                ).to(self.device)
                self.degrees = degrees.unsqueeze(-1)
            degree_embeds = self.degree_embedder(self.degrees.squeeze(-1))
            inputs.append(degree_embeds)

        x = torch.cat(inputs, dim=-1)
        for i in range(len(self.feature_list) - 1):
            x = self.layer_list["conv" + str(i + 1)](g, x)
            x = x.flatten(start_dim=2)
            if i < len(self.feature_list) - 2:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.final_proj(x.view(x.size(0), -1))
        return x


class GIN_GETS(nn.Module):
    """GIN-based expert for GETS."""

    def __init__(self, num_classes, hidden_dim, dropout_rate, num_layers,
                 device, expert_config, feature_dim, feature_hidden_dim, degree_hidden_dim):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.expert_config = expert_config
        self.device = device

        in_channels = 0
        if "logits" in expert_config:
            in_channels += num_classes
        if "features" in expert_config:
            self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)
            in_channels += feature_hidden_dim
        if "degrees" in expert_config:
            in_channels += degree_hidden_dim

        self.feature_list = [in_channels, hidden_dim, num_classes]
        for _ in range(num_layers - 2):
            self.feature_list.insert(-1, hidden_dim)

        layer_list = []
        for i in range(len(self.feature_list) - 1):
            layer_list.append(
                ["conv" + str(i + 1),
                 dglnn.GINConv(
                     nn.Sequential(
                         nn.Linear(self.feature_list[i], self.feature_list[i + 1]),
                         nn.ReLU(),
                         nn.Linear(self.feature_list[i + 1], self.feature_list[i + 1])
                     )
                 )]
            )
        self.layer_list = nn.ModuleDict(layer_list)
        self.degree_dim = degree_hidden_dim

    def forward(self, g, logits, features):
        inputs = []
        if "logits" in self.expert_config:
            inputs.append(logits)
        if "features" in self.expert_config:
            features = self.proj_feature(features)
            inputs.append(features)
        if "degrees" in self.expert_config:
            if not hasattr(self, "degrees"):
                degrees = g.in_degrees() + g.out_degrees()
                max_degree = degrees.max() + 1
                self.degree_embedder = nn.Embedding(
                    num_embeddings=max_degree, embedding_dim=self.degree_dim
                ).to(self.device)
                self.degrees = degrees.unsqueeze(-1)
            degree_embeds = self.degree_embedder(self.degrees.squeeze(-1))
            inputs.append(degree_embeds)

        x = torch.cat(inputs, dim=-1)
        for i in range(len(self.feature_list) - 1):
            x = self.layer_list["conv" + str(i + 1)](g, x)
            if i < len(self.feature_list) - 2:
                x = F.relu(x)
                x = F.dropout(x, self.dropout_rate, training=self.training)
        return x


# =============================================================================
# GETS Main Class
# =============================================================================

# Default expert configurations (7 experts with different input combinations)
DEFAULT_EXPERT_CONFIGS = [
    ["logits"],
    ["features"],
    ["degrees"],
    ["logits", "features"],
    ["features", "degrees"],
    ["logits", "degrees"],
    ["logits", "features", "degrees"]
]


class GETS(nn.Module):
    """
    GETS: Graph Expert Temperature Scaling.

    Sparsely gated mixture of experts for node-wise temperature prediction.
    calibrated_logits = logits * softplus(temperature)

    From: GETS-main/GETS-main/model/GETS.py
    """

    def __init__(self, num_classes, feature_dim, device,
                 hidden_dim=16, dropout=0.5, num_layers=2,
                 expert_select=2, expert_configs=None,
                 feature_hidden_dim=16, degree_hidden_dim=16,
                 noisy_gating=True, loss_coef=1.0, backbone='gcn'):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.noisy_gating = noisy_gating
        self.loss_coef = loss_coef
        self.backbone = backbone

        # Expert configs
        if expert_configs is None:
            expert_configs = DEFAULT_EXPERT_CONFIGS
        self.num_experts = len(expert_configs)
        self.k = expert_select  # How many experts to use per node

        # Feature projection for gating
        self.proj_feature = nn.Linear(feature_dim, feature_hidden_dim)

        # Build experts based on backbone
        if backbone == 'gcn':
            ExpertClass = GCN_GETS
        elif backbone == 'gat':
            ExpertClass = GAT_GETS
        elif backbone == 'gin':
            ExpertClass = GIN_GETS
        else:
            raise NotImplementedError(f"Backbone {backbone} not supported")

        self.experts = nn.ModuleList([
            ExpertClass(
                num_classes=num_classes,
                hidden_dim=hidden_dim,
                dropout_rate=dropout,
                num_layers=num_layers,
                device=device,
                expert_config=expert_configs[i],
                feature_dim=feature_dim,
                feature_hidden_dim=feature_hidden_dim,
                degree_hidden_dim=degree_hidden_dim,
            ) for i in range(self.num_experts)
        ])

        # Gating network parameters
        gating_input_dim = feature_hidden_dim + num_classes
        self.w_gate = nn.Parameter(torch.zeros(gating_input_dim, self.num_experts))
        self.w_noise = nn.Parameter(torch.zeros(gating_input_dim, self.num_experts))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        # Store graph for calibrate()
        self.g = None
        self.features = None

        assert self.k <= self.num_experts

    def cv_squared(self, x):
        """Squared coefficient of variation (for load balancing loss)."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute load per expert (number of examples where gate > 0)."""
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Probability that value is in top k (for load balancing)."""
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating mechanism."""
        clean_logits = x @ self.w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        # Top-k selection
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def forward(self, logits, g=None, features=None):
        """Forward pass with mixture of experts."""
        if g is None:
            g = self.g
        if features is None:
            features = self.features

        # Gating input: projected features + logits
        features_trans = self.proj_feature(features)
        gating_input = torch.cat([features_trans, logits], dim=1)

        # Compute gates
        node_gates, load = self.noisy_top_k_gating(gating_input, self.training)

        # Load balancing loss
        importance = node_gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        # Expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            expert_i_output = self.experts[i](g, logits, features)
            expert_outputs.append(expert_i_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        # Weighted combination of expert temperatures
        temperature = (expert_outputs * node_gates.unsqueeze(-1)).sum(dim=1)
        calibrated = logits * F.softplus(temperature)

        return calibrated, loss, node_gates

    def calibrate(self, logits):
        """Calibrate pre-computed logits (inference mode)."""
        self.eval()
        with torch.no_grad():
            calibrated, _, _ = self.forward(logits, self.g, self.features)
        return calibrated

    def fit(self, logits, labels, g, features, val_idx, train_idx,
            epochs=1000, patience=50, lr=0.001, weight_decay=0):
        """Fit GETS on calibration set."""
        self.g = g
        self.features = features
        self.to(self.device)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        vlss_mn = float('Inf')
        curr_step = 0
        best_state = None

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            # Forward with load balancing loss
            calibrated, loss_load, _ = self.forward(logits, g, features)

            # Cross-entropy loss on calibration training set (val_idx)
            loss = F.cross_entropy(calibrated[val_idx], labels[val_idx])
            loss = loss + loss_load
            loss.backward()
            optimizer.step()

            # Validation (early stopping on train_idx)
            with torch.no_grad():
                self.eval()
                calibrated, loss_load, _ = self.forward(logits, g, features)
                val_loss = F.cross_entropy(calibrated[train_idx], labels[train_idx])

                if val_loss <= vlss_mn:
                    vlss_mn = val_loss.item()
                    curr_step = 0
                    best_state = copy.deepcopy(self.state_dict())
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break

        # Restore best state
        if best_state is not None:
            self.load_state_dict(best_state)

        return self
