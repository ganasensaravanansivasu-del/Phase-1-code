"""
=============================================================================
network_updated.py — PINN with Standard MLP + Multi-Scale Fourier Features
                     + Hard IC Enforcement + Soft Attention Weights
=============================================================================
Methods Implemented:
1. Multi-Scale Fourier Feature Embedding (σ=1,5,10)
2. Standard MLP (5 hidden layers, 256 neurons each)
3. Hard Initial Condition Enforcement (output = t* × network_output)
4. Soft Attention Adaptive Loss Weights
=============================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from config import (DEVICE, HIDDEN_LAYERS, HIDDEN_NEURONS,
                    FF_SIGMA, FF_FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  MULTI-SCALE FOURIER FEATURE EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

class MultiscaleFourierEmbedding(nn.Module):
    """
    Maps input (x*, y*, t*) to multi-scale Fourier features.
    Output size = 2 * FF_FEATURES * len(FF_SIGMA) = 768
    """

    def __init__(self, input_dim=3, n_features=FF_FEATURES, sigmas=FF_SIGMA):
        super().__init__()
        self.sigmas     = sigmas
        self.n_features = n_features

        # Register fixed (non-trainable) frequency matrices for each scale
        for i, sigma in enumerate(sigmas):
            B = torch.randn(input_dim, n_features) * sigma
            self.register_buffer(f'B_{i}', B)

        self.output_dim = 2 * n_features * len(sigmas)  # 2×128×3 = 768

    def forward(self, x_star, y_star, t_star):
        v = torch.cat([x_star, y_star, t_star], dim=1)
        features = []
        for i in range(len(self.sigmas)):
            B = getattr(self, f'B_{i}')
            proj = v @ B
            features.append(torch.cos(proj))
            features.append(torch.sin(proj))
        return torch.cat(features, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  STANDARD MLP (SIMPLIFIED - NO GATING)
# ─────────────────────────────────────────────────────────────────────────────

class StandardMLP(nn.Module):
    """Standard fully-connected MLP with tanh activation."""

    def __init__(self, embed_dim=768, hidden_dim=256, n_layers=5, output_dim=3):
        super().__init__()
        self.input_layer = nn.Linear(embed_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, embedding):
        H = torch.tanh(self.input_layer(embedding))
        for layer in self.hidden_layers:
            H = torch.tanh(layer(H))
        return self.output_layer(H)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COMPLETE PINN NETWORK WITH HARD IC ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

class PINNNetwork(nn.Module):
    """Complete PINN with Fourier features + Standard MLP + Hard IC"""

    def __init__(self):
        super().__init__()
        self.embedding = MultiscaleFourierEmbedding(
            input_dim=3, n_features=FF_FEATURES, sigmas=FF_SIGMA)
        embed_dim = self.embedding.output_dim
        self.mlp = StandardMLP(
            embed_dim=embed_dim, hidden_dim=HIDDEN_NEURONS,
            n_layers=HIDDEN_LAYERS, output_dim=3)

    def forward(self, x_star, y_star, t_star):
        emb = self.embedding(x_star, y_star, t_star)
        raw_output = self.mlp(emb)
        T_raw = raw_output[:, 0:1]
        u_raw = raw_output[:, 1:2]
        v_raw = raw_output[:, 2:3]
        # Hard IC enforcement
        T_star = t_star * T_raw
        u_star = t_star * u_raw
        v_star = t_star * v_raw
        return T_star, u_star, v_star


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SOFT ATTENTION ADAPTIVE LOSS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

class SoftAttentionWeights(nn.Module):
    """Learnable soft attention weights for each loss term."""

    def __init__(self, phase='A'):
        super().__init__()
        self.phase = phase
        if phase == 'A':
            self.loss_names = ['ic', 'thermal_bc', 'thermal_pde']
        else:
            self.loss_names = ['elastic_bc', 'elastic_pde', 'interface']
        n = len(self.loss_names)
        self.log_weights = nn.Parameter(torch.zeros(n))

    def forward(self):
        w = torch.softmax(self.log_weights, dim=0)
        return {name: w[i] for i, name in enumerate(self.loss_names)}

    def get_weights_numpy(self):
        with torch.no_grad():
            return torch.softmax(self.log_weights, dim=0).cpu().numpy()

    def reset_weights(self):
        with torch.no_grad():
            self.log_weights.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMBINED MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class PINNModel(nn.Module):
    """Wraps PINNNetwork + SoftAttentionWeights"""

    def __init__(self, phase='A'):
        super().__init__()
        self.phase = phase
        self.net = PINNNetwork()
        self.weights = SoftAttentionWeights(phase=phase)

    def forward(self, x_star, y_star, t_star):
        return self.net(x_star, y_star, t_star)

    def get_loss_weights(self):
        return self.weights()

    def reset_attention_weights(self):
        self.weights.reset_weights()

    def count_parameters(self):
        n_net = sum(p.numel() for p in self.net.parameters())
        n_wts = sum(p.numel() for p in self.weights.parameters())
        return n_net, n_wts, n_net + n_wts

    def freeze_for_phase_B(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()