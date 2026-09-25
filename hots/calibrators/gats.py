"""
GATS (Graph Attention Temperature Scaling) - GETS-main style implementation.

From: GETS-main/GETS-main/model/calibrator.py

API:
    gats = GATS(num_classes, edge_index, num_nodes, train_idx, device, heads=8, bias=1)
    gats.fit(logits, labels, edge_index, val_idx, train_idx, epochs, patience, lr)
    calibrated_logits = gats.calibrate(logits)
"""

import copy
from typing import Union, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree
from torch_geometric.typing import OptPairTensor, Adj, OptTensor


def shortest_path_length(edge_index, mask, max_hop, device):
    """
    Return the shortest path length to the mask for every node.
    From: GETS-main/GETS-main/model/calibrator.py
    """
    dist_to_train = torch.ones_like(mask, dtype=torch.long, device=device) * torch.iinfo(torch.long).max
    seen_mask = torch.clone(mask).to(device)
    for hop in range(max_hop):
        current_hop = torch.nonzero(mask).to(device)
        dist_to_train[mask] = hop
        next_hop = torch.zeros_like(mask, dtype=torch.bool, device=device)
        for node in current_hop:
            node_mask = edge_index[0, :] == node
            nbrs = edge_index[1, node_mask]
            next_hop[nbrs] = True
        hop += 1
        mask = torch.logical_and(next_hop, ~seen_mask)
        seen_mask[next_hop] = True
    return dist_to_train


class CalibAttentionLayer(MessagePassing):
    """
    Calibration Attention Layer from GATS paper.
    From: GETS-main/GETS-main/model/calibrator.py
    """
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            edge_index: Adj,
            num_nodes: int,
            train_mask,
            dist_to_train: Tensor = None,
            heads: int = 8,
            negative_slope: float = 0.2,
            bias: float = 1,
            self_loops: bool = True,
            fill_value: Union[float, Tensor, str] = 'mean',
            bfs_depth=2,
            device='cpu',
            **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.fill_value = fill_value
        self.edge_index = edge_index
        self.num_nodes = num_nodes

        self.temp_lin = Linear(in_channels, heads,
                               bias=False, weight_initializer='glorot')

        self.conf_coef = Parameter(torch.zeros([]))
        self.bias = Parameter(torch.ones(1) * bias)
        self.train_a = Parameter(torch.ones(1))
        self.dist1_a = Parameter(torch.ones(1))

        # Compute distances to nearest training node
        if isinstance(train_mask, np.ndarray):
            train_mask_indices_tensor = torch.from_numpy(train_mask).to(device)
        else:
            train_mask_indices_tensor = train_mask.to(device)
        train_mask_tensor = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask_tensor.scatter_(0, train_mask_indices_tensor, True)
        dist_to_train = dist_to_train if dist_to_train is not None else shortest_path_length(
            edge_index, train_mask_tensor, bfs_depth, device)
        self.register_buffer('dist_to_train', dist_to_train)

        self.reset_parameters()
        if self_loops:
            self.edge_index, _ = remove_self_loops(self.edge_index, None)
            self.edge_index, _ = add_self_loops(
                self.edge_index, None, fill_value=self.fill_value, num_nodes=num_nodes)

    def reset_parameters(self):
        self.temp_lin.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor]):
        N, H = self.num_nodes, self.heads

        # Normalize logits
        normalized_x = x - torch.min(x, 1, keepdim=True)[0]
        denom = torch.max(x, 1, keepdim=True)[0] - torch.min(x, 1, keepdim=True)[0]
        denom = torch.clamp(denom, min=1e-8)
        normalized_x = normalized_x / denom

        x_sorted = torch.sort(normalized_x, -1)[0]
        temp = self.temp_lin(x_sorted)

        # Spatial coefficient
        a_cluster = torch.ones(N, dtype=torch.float32, device=x.device)
        a_cluster[self.dist_to_train == 0] = self.train_a
        a_cluster[self.dist_to_train == 1] = self.dist1_a

        # Confidence smoothing
        conf = F.softmax(x, dim=1).amax(-1)
        deg = degree(self.edge_index[0, :], self.num_nodes)
        deg_inverse = 1 / deg
        deg_inverse[deg_inverse == float('inf')] = 0

        out = self.propagate(self.edge_index,
                             temp=temp.view(N, H) * a_cluster.unsqueeze(-1),
                             alpha=x / a_cluster.unsqueeze(-1),
                             conf=conf)
        sim, dconf = out[:, :-1], out[:, -1:]
        out = F.softplus(sim + self.conf_coef * dconf * deg_inverse.unsqueeze(-1))
        out = out.mean(dim=1) + self.bias
        return out.unsqueeze(1)

    def message(
            self,
            temp_j: Tensor,
            alpha_j: Tensor,
            alpha_i: OptTensor,
            conf_i: Tensor,
            conf_j: Tensor,
            index: Tensor,
            ptr: OptTensor,
            size_i: Optional[int]) -> Tensor:
        if alpha_i is None:
            print("alpha_i is none")
        alpha = (alpha_j * alpha_i).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        return torch.cat([
            (temp_j * alpha.unsqueeze(-1).expand_as(temp_j)),
            (conf_i - conf_j).unsqueeze(-1)], -1)


class GATS(nn.Module):
    """
    GATS: Graph Attention Temperature Scaling.

    calibrated_logits = logits / temperature

    From: GETS-main/GETS-main/model/calibrator.py
    """

    def __init__(self, num_classes, edge_index, num_nodes, train_idx, device,
                 heads=8, bias=1, dist_to_train=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.device = device

        self.cagat = CalibAttentionLayer(
            in_channels=num_classes,
            out_channels=1,
            edge_index=edge_index,
            num_nodes=num_nodes,
            train_mask=train_idx,
            dist_to_train=dist_to_train,
            heads=heads,
            bias=bias,
            device=device
        )

    def graph_temperature_scale(self, logits):
        """Compute node-wise temperature using attention."""
        temperature = self.cagat(logits).view(self.num_nodes, -1)
        return temperature.expand(self.num_nodes, logits.size(1))

    def forward(self, logits):
        """Calibrate logits."""
        temperature = self.graph_temperature_scale(logits)
        return logits / temperature

    def calibrate(self, logits):
        """Calibrate pre-computed logits."""
        return self.forward(logits)

    def fit(self, logits, labels, edge_index, val_idx, train_idx,
            epochs=1000, patience=50, lr=0.01, weight_decay=0):
        """Fit GATS on calibration set."""
        self.to(self.device)

        optimizer = optim.Adam(self.cagat.parameters(), lr=lr, weight_decay=weight_decay)

        vlss_mn = float('Inf')
        curr_step = 0
        best_state = None

        for epoch in range(epochs):
            self.train()
            optimizer.zero_grad()

            # Calibrate
            temperature = self.graph_temperature_scale(logits)
            calibrated = logits / temperature

            # Loss on calibration training set (val_idx)
            loss = F.cross_entropy(calibrated[val_idx], labels[val_idx])
            loss.backward()
            optimizer.step()

            # Validation (early stopping on train_idx)
            with torch.no_grad():
                self.eval()
                temperature = self.graph_temperature_scale(logits)
                calibrated = logits / temperature
                val_loss = F.cross_entropy(calibrated[train_idx], labels[train_idx])

                if val_loss <= vlss_mn:
                    vlss_mn = val_loss.item()
                    curr_step = 0
                    best_state = copy.deepcopy(self.cagat.state_dict())
                else:
                    curr_step += 1
                    if curr_step >= patience:
                        break

        # Restore best state
        if best_state is not None:
            self.cagat.load_state_dict(best_state)

        return self
