"""
=============================================================================
network.py — PINN with Modified MLP Gating + Multi-Scale Fourier Features
             + Hard IC Enforcement + Soft Attention Weights
=============================================================================
Methods Implemented:
1. Multi-Scale Fourier Feature Embedding (σ=1,5,10)
2. Modified MLP with Gating (Wang et al. 2022)
3. Hard Initial Condition Enforcement (output = t* × network_output)
4. Soft Attention Adaptive Loss Weights

Architecture:
    Input (x*, y*, t*)
    ↓
    Multi-Scale Fourier Embedding (768-dim)
    ↓
    U, V Gates (768→256) — computed from embedding
    ↓
    5 × [Linear(256) + Tanh + Gating]
    ↓
    Output (T̂*, û*, v̂*)
    ↓
    Hard IC: T* = t* × T̂*, u* = t* × û*, v* = t* × v̂*
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

    For each scale σ_k:
        B_k ~ N(0, σ_k²)  — fixed random frequency matrix (not trained)
        γ_k(v) = [cos(B_k v), sin(B_k v)]

    Concatenation of all scales:
        γ(v) = [γ_1(v), γ_2(v), ..., γ_K(v)]

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
        """
        x_star, y_star, t_star : (N,1) tensors
        Returns                 : (N, 768) tensor
        """
        v = torch.cat([x_star, y_star, t_star], dim=1)   # (N,3)

        features = []
        for i in range(len(self.sigmas)):
            B = getattr(self, f'B_{i}')                   # (3, n_features)
            proj = v @ B                                   # (N, n_features)
            features.append(torch.cos(proj))
            features.append(torch.sin(proj))

        return torch.cat(features, dim=1)                 # (N, 768)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODIFIED MLP WITH GATING (Wang et al. 2022)
# ─────────────────────────────────────────────────────────────────────────────

class ModifiedMLPWithGating(nn.Module):
    """
    Modified MLP architecture with U, V gating mechanism.

    Key idea: Gates U and V modulate each hidden layer activation,
    creating skip-connection-like paths that improve gradient flow.

    Architecture:
        embedding (768-dim) → U, V gates (256-dim)
        For each hidden layer k:
            H_k' = tanh(W_k · H_{k-1} + b_k)
            H_k = H_k' ⊙ U + (1 - H_k') ⊙ V

    Reference:
        Wang, S., Wang, H. and Perdikaris, P., 2022.
        "Improved Architectures and Training Algorithms for Deep Operator Networks"
        Journal of Scientific Computing, 92, p.35.
    """

    def __init__(self, embed_dim=768, hidden_dim=256, n_layers=5, output_dim=3):
        super().__init__()

        # U and V encoders — compute gates from embedding
        self.encoder_U = nn.Linear(embed_dim, hidden_dim)
        self.encoder_V = nn.Linear(embed_dim, hidden_dim)

        # First hidden layer — maps embedding to hidden_dim
        self.input_layer = nn.Linear(embed_dim, hidden_dim)

        # Hidden layers with gating
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers - 1)
        ])

        # Output layer (no activation)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Xavier initialization
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, embedding):
        """
        Parameters
        ----------
        embedding : (N, 768) tensor — Fourier feature embedding

        Returns
        -------
        output : (N, 3) tensor — [T̂*, û*, v̂*] before hard IC enforcement
        """
        # Compute U and V gates from embedding
        U = torch.tanh(self.encoder_U(embedding))    # (N, 256)
        V = torch.tanh(self.encoder_V(embedding))    # (N, 256)

        # First hidden layer
        H = torch.tanh(self.input_layer(embedding))  # (N, 256)

        # Gated hidden layers
        for layer in self.hidden_layers:
            H_prime = torch.tanh(layer(H))           # (N, 256)
            H = H_prime * U + (1.0 - H_prime) * V    # Element-wise gating
            # Note: (1 - H_prime) is not the same as (1 - U), this is correct

        # Output layer
        output = self.output_layer(H)                # (N, 3)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# 3.  COMPLETE PINN NETWORK WITH HARD IC ENFORCEMENT
# ─────────────────────────────────────────────────────────────────────────────

class PINNNetwork(nn.Module):
    """
    Complete Physics-Informed Neural Network with:
    - Multi-scale Fourier feature embedding
    - Modified MLP with gating
    - Hard initial condition enforcement

    Input  : (x*, y*, t*)  — dimensionless coordinates
    Output : (T*, u*, v*)  — dimensionless temperature and displacements

    Hard IC Enforcement:
        T*(x*,y*,0) = 0 by construction:
        T* = t* × T̂*(x*, y*, t*)

    This guarantees IC is satisfied exactly without any loss term.
    """

    def __init__(self):
        super().__init__()

        # Multi-scale Fourier embedding
        self.embedding = MultiscaleFourierEmbedding(
            input_dim  = 3,
            n_features = FF_FEATURES,
            sigmas     = FF_SIGMA,
        )
        embed_dim = self.embedding.output_dim   # 768

        # Modified MLP with gating
        self.mlp = ModifiedMLPWithGating(
            embed_dim  = embed_dim,       # 768
            hidden_dim = HIDDEN_NEURONS,  # 256
            n_layers   = HIDDEN_LAYERS,   # 5
            output_dim = 3,
        )

    def forward(self, x_star, y_star, t_star):
        """
        Parameters
        ----------
        x_star, y_star, t_star : (N,1) tensors — dimensionless inputs

        Returns
        -------
        T_star : (N,1) — dimensionless temperature (T* = (T-T_ref)/ΔT_ref)
        u_star : (N,1) — dimensionless x-displacement
        v_star : (N,1) — dimensionless y-displacement

        Hard IC: All outputs are multiplied by t*, ensuring they vanish at t*=0
        """
        # Fourier embedding
        emb = self.embedding(x_star, y_star, t_star)   # (N, 768)

        # Modified MLP
        raw_output = self.mlp(emb)                      # (N, 3)

        # Extract raw predictions
        T_raw = raw_output[:, 0:1]    # (N, 1)
        u_raw = raw_output[:, 1:2]    # (N, 1)
        v_raw = raw_output[:, 2:3]    # (N, 1)

        # Hard IC enforcement: multiply by t*
        # This ensures T*(t*=0)=0, u*(t*=0)=0, v*(t*=0)=0 automatically
        T_star = t_star * T_raw
        u_star = t_star * u_raw
        v_star = t_star * v_raw

        return T_star, u_star, v_star


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SOFT ATTENTION ADAPTIVE LOSS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

class SoftAttentionWeights(nn.Module):
    """
    Learnable soft attention weights for each loss term.

    Each weight:  w_k = exp(λ_k) / Σ_j exp(λ_j)   (softmax)

    λ_k are learnable parameters trained alongside network weights.

    Phase A loss terms (thermal):
        0: ic          — initial condition
        1: thermal_bc  — thermal boundary conditions
        2: thermal_pde — thermal PDE residual

    Phase B loss terms (elastic with frozen thermal):
        0: elastic_bc  — elastic boundary conditions
        1: elastic_pde — elastic PDE residual
        2: interface   — interface conditions (L1, L2, L3, L4)
    """

    def __init__(self, phase='A'):
        super().__init__()
        self.phase = phase

        if phase == 'A':
            self.loss_names = ['ic', 'thermal_bc', 'thermal_pde']
        else:  # phase == 'B'
            self.loss_names = ['elastic_bc', 'elastic_pde', 'interface']

        n = len(self.loss_names)
        # Initialize all λ_k = 0 → all weights equal (1/n) at start
        self.log_weights = nn.Parameter(torch.zeros(n))

    def forward(self):
        """Returns dict of {loss_name: scalar_weight}"""
        w = torch.softmax(self.log_weights, dim=0)
        return {name: w[i] for i, name in enumerate(self.loss_names)}

    def get_weights_numpy(self):
        """For logging purposes — returns numpy array."""
        with torch.no_grad():
            return torch.softmax(self.log_weights, dim=0).cpu().numpy()

    def reset_weights(self):
        """Reset attention weights to equal — call at curriculum transitions."""
        with torch.no_grad():
            self.log_weights.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMBINED MODEL WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class PINNModel(nn.Module):
    """
    Wraps PINNNetwork + SoftAttentionWeights into a single nn.Module
    so both sets of parameters are captured by one optimizer.
    """

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
        """Reset attention weights — call at curriculum transitions."""
        self.weights.reset_weights()

    def count_parameters(self):
        n_net = sum(p.numel() for p in self.net.parameters())
        n_wts = sum(p.numel() for p in self.weights.parameters())
        return n_net, n_wts, n_net + n_wts

    def freeze_for_phase_B(self):
        """
        Freeze the network for Phase B.
        In Phase B, we use this model's thermal predictions but don't update weights.
        Create a new PINNModel for elastic training.
        """
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
