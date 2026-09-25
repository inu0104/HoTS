"""
DCGC (Data-Centric Graph Calibration).

Based on: "Calibrating Graph Neural Networks from a Data-Centric Perspective"

DCGC learns per-edge weights via an MLP on base model embeddings,
then re-runs the model with the learned edge weights. A coefficient
correction step reweights edges based on prediction disagreement.

Note: Unlike other calibrators that only transform logits, DCGC re-runs
the base model with learned edge weights and MAY change predictions.

API:
    dcgc = DCGC(num_classes, device, dropout=0.5)
    dcgc.fit(logits, labels, g, features, val_idx, train_idx, model=model)
    calibrated_logits = dcgc.calibrate(logits)
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn as dglnn


def _forward_with_edge_weight(model, g, features, edge_weight):
    """Run a DGL GNN model with per-edge weights.

    Handles GCN, GAT, GIN, GraphSAGE from src/models.py.
    All DGL conv layers support edge_weight in their forward().
    """
    h = features
    for i, layer in enumerate(model.layers):
        h = layer(g, h, edge_weight=edge_weight)
        if i < len(model.layers) - 1:
            if model.norm:
                h = model.norms[i](h)
            h = F.relu(h)
            h = model.dropout(h)

    # GAT has a final projection layer
    if hasattr(model, 'final_project'):
        h = model.final_project(h.view(h.size(0), -1))

    return h


class DCGC(nn.Module):
    """
    DCGC: Data-Centric Graph Calibration.

    Learns per-edge weights from base model logits:
        edge_weight = ReLU(MLP(concat(logit_src, logit_dst)))

    Then applies coefficient correction based on prediction disagreement
    and re-runs the model with weighted edges.

    This calibrator MAY change predictions.
    """

    def __init__(self, num_classes, device, dropout=0.5, alpha=0.5, beta=10.0,
                 edge_chunk_size=2_000_000, checkpoint_forward=False,
                 amp_forward=False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.alpha_coef = alpha   # coefficient correction parameter
        self.beta_coef = beta     # prediction sharpening parameter
        # Chunk size for per-edge MLP / coefficient correction. Needed for
        # large graphs (e.g. Reddit ~114M edges) where concatenating the full
        # [E, 2C] tensor would OOM.
        self.edge_chunk_size = int(edge_chunk_size)
        # Gradient checkpointing for the base-model forward during fit().
        # Trades compute for backward-activation memory.
        self.checkpoint_forward = bool(checkpoint_forward)
        # FP16 autocast for the base-model forward during fit(). Halves the
        # per-edge message tensors (~7.3 GB → ~3.7 GB at Reddit scale) at the
        # cost of slightly lower numerical precision. DGL's edge-weight path
        # in GraphConv materializes per-edge tensors that cannot be avoided
        # without rewriting the message-passing kernel.
        self.amp_forward = bool(amp_forward)

        # Edge weight MLP: concat(logit_src, logit_dst) -> scalar weight
        self.extractor = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes * 4),
            nn.BatchNorm1d(num_classes * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes * 4, 1),
        )

        self._calibrated_logits = None

    def _get_edge_weight(self, emb, g):
        """Compute edge weights from base model output.

        For small graphs, processes all edges at once. For large graphs
        (E > edge_chunk_size), processes edges in chunks to bound memory.
        Chunking is gradient-safe because each chunk's output is written
        into its slice of the preallocated result tensor.
        """
        src, dst = g.edges()
        src, dst = src.long(), dst.long()
        E = src.numel()

        if E <= self.edge_chunk_size:
            f12 = torch.cat([emb[src], emb[dst]], dim=-1)
            return self.extractor(f12).relu()

        # Chunked path: concatenate outputs from each chunk.
        out_chunks = []
        cs = self.edge_chunk_size
        for start in range(0, E, cs):
            end = min(start + cs, E)
            f12 = torch.cat([emb[src[start:end]], emb[dst[start:end]]], dim=-1)
            out_chunks.append(self.extractor(f12).relu())
        return torch.cat(out_chunks, dim=0)

    def _apply_coefficient_correction(self, model, g, features, edge_weight):
        """Apply DCGC coefficient correction.

        Upweights edges between nodes with similar predictions,
        downweights edges between nodes with different predictions.

        For large graphs, computes per-edge disagreement in chunks.
        """
        import contextlib
        amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                   if self.amp_forward else contextlib.nullcontext())
        with torch.no_grad(), amp_ctx:
            output = _forward_with_edge_weight(model, g, features, edge_weight.squeeze())
            pred = F.softmax(output.float(), dim=1)

            # Sharpen predictions
            pred = torch.exp(self.beta_coef * pred)
            pred = pred / pred.sum(dim=1, keepdim=True)

            # Prediction disagreement per edge (chunked for large graphs).
            src, dst = g.edges()
            src, dst = src.long(), dst.long()
            E = src.numel()

            if E <= self.edge_chunk_size:
                coefficient = torch.norm(pred[src] - pred[dst], dim=1)
            else:
                coefficient = torch.empty(E, device=pred.device, dtype=pred.dtype)
                cs = self.edge_chunk_size
                for start in range(0, E, cs):
                    end = min(start + cs, E)
                    coefficient[start:end] = torch.norm(
                        pred[src[start:end]] - pred[dst[start:end]], dim=1
                    )
            coefficient = 1.0 / (coefficient + self.alpha_coef)

        return edge_weight.squeeze() * coefficient

    def forward(self, logits):
        """Return cached calibrated logits."""
        return self._calibrated_logits

    def calibrate(self, logits):
        """Return cached calibrated logits."""
        return self._calibrated_logits

    def fit(self, logits, labels, g, features, val_idx, train_idx,
            model=None, epochs=1000, patience=100, lr=0.01, weight_decay=5e-3):
        """Fit DCGC on calibration set.

        Args:
            model: Pre-trained base GNN model (required, will be frozen during training)
        """
        if model is None:
            raise ValueError("DCGC requires the base model. Pass model=model to fit().")

        self.to(self.device)

        # Freeze base model
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # Cache base model output (constant since model is frozen)
        with torch.no_grad():
            emb = model(g, features).detach()

        optimizer = optim.Adam(self.extractor.parameters(),
                               lr=lr, weight_decay=weight_decay)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        # AMP context: wraps forward passes in fp16 autocast for large graphs.
        # Use a no-op context for the default path to keep fp32 behavior exact.
        import contextlib
        if self.amp_forward:
            amp_ctx = lambda: torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        else:
            amp_ctx = contextlib.nullcontext

        for epoch in range(epochs):
            self.extractor.train()
            optimizer.zero_grad()

            edge_weight = self._get_edge_weight(emb, g)
            if self.checkpoint_forward:
                # Recompute GCN activations in backward rather than storing them.
                def _ckpt_fn(ew):
                    with amp_ctx():
                        return _forward_with_edge_weight(model, g, features, ew)
                out = torch.utils.checkpoint.checkpoint(
                    _ckpt_fn, edge_weight.squeeze(), use_reentrant=False
                )
            else:
                with amp_ctx():
                    out = _forward_with_edge_weight(model, g, features, edge_weight.squeeze())

            loss = F.cross_entropy(out[val_idx], labels[val_idx])
            loss.backward()
            optimizer.step()

            # Validation (early stopping)
            with torch.no_grad():
                self.extractor.eval()
                edge_weight = self._get_edge_weight(emb, g)
                with amp_ctx():
                    out = _forward_with_edge_weight(model, g, features, edge_weight.squeeze())
                val_loss = F.cross_entropy(out[train_idx], labels[train_idx])

                if val_loss < best_loss:
                    best_loss = val_loss.item()
                    best_state = copy.deepcopy(self.extractor.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        # Restore best state
        if best_state:
            self.extractor.load_state_dict(best_state)

        # Compute final calibrated logits with coefficient correction
        with torch.no_grad():
            self.extractor.eval()
            edge_weight = self._get_edge_weight(emb, g)
            corrected_ew = self._apply_coefficient_correction(
                model, g, features, edge_weight
            )
            with amp_ctx():
                out_final = _forward_with_edge_weight(
                    model, g, features, corrected_ew
                )
            # Cast back to fp32 so downstream calibrators (TS, HoTS, ...) see
            # the same precision as the ERM logits.
            self._calibrated_logits = out_final.float()

        # Re-enable base model gradients
        for param in model.parameters():
            param.requires_grad = True

        return self

    def get_params(self):
        return {
            "alpha": self.alpha_coef,
            "beta": self.beta_coef,
        }
