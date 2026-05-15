"""
=============================================================================
network_updated.py — Single PINN Network with Dropout
=============================================================================
CORRECTED VERSION:
- Single network outputs [T*, u*, v*] together (no decoupling)
- 4 layers × 128 neurons (reduced for overfitting)
- 10% dropout after each layer
- Multi-scale Fourier features
- Hard IC enforcement
=============================================================================
"""

import torch
import torch.nn as nn
from config import (DEVICE, HIDDEN_LAYERS, HIDDEN_NEURONS, DROPOUT_RATE,
                    FF_SIGMA, FF_FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MULTI-SCALE FOURIER FEATURE EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

class MultiscaleFourierEmbedding(nn.Module):
    """
    Multi-scale Fourier feature embedding.
    Output: 2 * FF_FEATURES * len(FF_SIGMA) = 768-dim
    """
    def __init__(self, input_dim=3, n_features=FF_FEATURES, sigmas=FF_SIGMA):
        super().__init__()
        self.sigmas = sigmas
        self.n_features = n_features
        
        for i, sigma in enumerate(sigmas):
            B = torch.randn(input_dim, n_features) * sigma
            self.register_buffer(f'B_{i}', B)
        
        self.output_dim = 2 * n_features * len(sigmas)

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
# 2. STANDARD MLP WITH DROPOUT
# ─────────────────────────────────────────────────────────────────────────────

class StandardMLP(nn.Module):
    """
    Standard MLP with dropout for regularization.
    4 layers × 128 neurons with 10% dropout.
    """
    def __init__(self, embed_dim=768, hidden_dim=128, n_layers=4, 
                 output_dim=3, dropout_rate=0.1):
        super().__init__()
        
        self.input_layer = nn.Linear(embed_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, embedding):
        H = torch.tanh(self.input_layer(embedding))
        H = self.dropout(H)
        
        for layer in self.hidden_layers:
            H = torch.tanh(layer(H))
            H = self.dropout(H)
        
        return self.output_layer(H)


# ─────────────────────────────────────────────────────────────────────────────
# 3. COMPLETE PINN NETWORK (SINGLE - OUTPUTS T, u, v TOGETHER)
# ─────────────────────────────────────────────────────────────────────────────

class PINNNetwork(nn.Module):
    """
    Single PINN network outputs [T*, u*, v*] together.
    No thermal-mechanical decoupling.
    """
    def __init__(self):
        super().__init__()
        
        self.embedding = MultiscaleFourierEmbedding(
            input_dim=3, n_features=FF_FEATURES, sigmas=FF_SIGMA)
        
        embed_dim = self.embedding.output_dim
        
        self.mlp = StandardMLP(
            embed_dim=embed_dim,
            hidden_dim=HIDDEN_NEURONS,
            n_layers=HIDDEN_LAYERS,
            output_dim=3,
            dropout_rate=DROPOUT_RATE
        )

    def forward(self, x_star, y_star, t_star):
        """
        Returns:
            T_star: (N,1) dimensionless temperature
            u_star: (N,1) dimensionless x-displacement
            v_star: (N,1) dimensionless y-displacement
        """
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
# 4. SOFT ATTENTION WEIGHTS (SINGLE NETWORK - ALL LOSSES)
# ─────────────────────────────────────────────────────────────────────────────

class SoftAttentionWeights(nn.Module):
    """
    Learnable attention weights for all loss terms (single network).
    
    Loss terms:
        0: ic
        1: thermal_bc
        2: elastic_bc
        3: thermal_pde
        4: elastic_pde
        5: interface
    """
    def __init__(self):
        super().__init__()
        self.loss_names = ['ic', 'thermal_bc', 'elastic_bc', 
                          'thermal_pde', 'elastic_pde', 'interface']
        n = len(self.loss_names)
        self.log_weights = nn.Parameter(torch.zeros(n))

    def forward(self):
        w = torch.softmax(self.log_weights, dim=0)
        return {name: w[i] for i, name in enumerate(self.loss_names)}

    def reset_weights(self):
        with torch.no_grad():
            self.log_weights.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMBINED MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PINNModel(nn.Module):
    """Complete PINN model with network + attention weights"""
    def __init__(self):
        super().__init__()
        self.net = PINNNetwork()
        self.weights = SoftAttentionWeights()

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
