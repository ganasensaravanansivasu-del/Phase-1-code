"""
=============================================================================
sampling_updated.py — Collocation Point Sampling with Non-Uniform Time
=============================================================================
CORRECTED VERSION:
- Y domain: [-14, +14] mm → y* ∈ [-1, +1] (CORRECTED from [-2, +2])
- Non-uniform time sampling (50% in 0-2s, 30% in 2-4s, 20% in 4-10s)
- Interface-biased spatial sampling (40% near interfaces)
- Single data preparation (no Phase A/B split)
=============================================================================
"""

import torch
import numpy as np
from config import (DEVICE, L_REF, T_MAX, t_REF,
                    X_MAX_STAR, Y_MIN_STAR, Y_MAX_STAR, T_STAR_MAX,
                    R_INNER_STAR, R_CuCrZr_STAR, R_Cu_STAR,
                    R_FGM1_STAR, R_FGM2_STAR, R_FGM3_STAR,
                    N_INTERIOR, N_IC, N_BC_TOP, N_BC_BOTTOM,
                    N_BC_INNER, N_BC_LEFT, N_BC_RIGHT, N_INTERFACE,
                    N_VALIDATION, INTERFACE_BIAS_FRACTION, INTERFACE_ZONE_WIDTH)

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN CHECK
# ─────────────────────────────────────────────────────────────────────────────

def in_domain_star(x_s, y_s):
    """Check if points (x*, y*) are inside the valid domain (excluding hole)"""
    r_s = np.sqrt(x_s**2 + y_s**2)
    inside_outer = (x_s >= 0) & (x_s <= X_MAX_STAR) & (y_s >= Y_MIN_STAR) & (y_s <= Y_MAX_STAR)
    outside_hole = (r_s >= R_INNER_STAR)
    return inside_outer & outside_hole


# ─────────────────────────────────────────────────────────────────────────────
# NON-UNIFORM TIME SAMPLING (YOUR BRILLIANT IDEA!)
# ─────────────────────────────────────────────────────────────────────────────

def sample_nonuniform_time(N_total, t_max_star=None):
    """
    Non-uniform time sampling based on physics:
    - 0-2s (extreme transient): 50% of points
    - 2-4s (moderate transient): 30% of points
    - 4-10s (quasi-steady): 20% of points
    
    This is YOUR brilliant contribution!
    """
    if t_max_star is None:
        t_max_star = T_STAR_MAX
    
    # Time breakpoints in dimensionless form
    # t_ref ≈ 8.82s, so:
    # 2s → 2/8.82 ≈ 0.227
    # 4s → 4/8.82 ≈ 0.454
    t_break1 = 2.0 / t_REF
    t_break2 = 4.0 / t_REF
    
    # Distribution
    n_early = int(0.50 * N_total)  # 50% in early transient
    n_mid = int(0.30 * N_total)    # 30% in mid transient
    n_late = N_total - n_early - n_mid  # 20% in quasi-steady
    
    # Generate samples
    t_early = np.random.uniform(0.0, t_break1, n_early)
    t_mid = np.random.uniform(t_break1, t_break2, n_mid)
    t_late = np.random.uniform(t_break2, t_max_star, n_late)
    
    # Concatenate and shuffle
    t_all = np.concatenate([t_early, t_mid, t_late])
    np.random.shuffle(t_all)
    
    return t_all[:, None]


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE-BIASED INTERIOR SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_interior_biased(N, bias_fraction=INTERFACE_BIAS_FRACTION):
    """
    Sample interior points with bias toward interfaces and FGM layers.
    40% of points concentrated in critical regions.
    """
    N_biased = int(N * bias_fraction)
    N_uniform = N - N_biased
    
    # Interface radii
    interfaces = [R_CuCrZr_STAR, R_Cu_STAR, R_FGM1_STAR, R_FGM2_STAR, R_FGM3_STAR]
    
    # Biased sampling
    x_bias_list, y_bias_list, t_bias_list = [], [], []
    n_per_interface = N_biased // len(interfaces)
    
    for r_if in interfaces:
        count = 0
        while count < n_per_interface:
            x_c = np.random.uniform(0, X_MAX_STAR, n_per_interface * 3)
            y_c = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, n_per_interface * 3)
            r_c = np.sqrt(x_c**2 + y_c**2)
            
            # Select points near this interface
            mask = np.abs(r_c - r_if) < INTERFACE_ZONE_WIDTH
            mask &= in_domain_star(x_c, y_c)
            
            valid_x = x_c[mask]
            valid_y = y_c[mask]
            
            needed = n_per_interface - count
            if len(valid_x) > 0:
                take = min(len(valid_x), needed)
                x_bias_list.append(valid_x[:take])
                y_bias_list.append(valid_y[:take])
                count += take
    
    x_bias = np.concatenate(x_bias_list) if x_bias_list else np.array([])
    y_bias = np.concatenate(y_bias_list) if y_bias_list else np.array([])
    
    # Uniform sampling for remaining points
    count_uniform = 0
    x_uni_list, y_uni_list = [], []
    while count_uniform < N_uniform:
        x_c = np.random.uniform(0, X_MAX_STAR, N_uniform * 2)
        y_c = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, N_uniform * 2)
        mask = in_domain_star(x_c, y_c)
        valid_x = x_c[mask]
        valid_y = y_c[mask]
        needed = N_uniform - count_uniform
        if len(valid_x) > 0:
            take = min(len(valid_x), needed)
            x_uni_list.append(valid_x[:take])
            y_uni_list.append(valid_y[:take])
            count_uniform += take
    
    x_uni = np.concatenate(x_uni_list)
    y_uni = np.concatenate(y_uni_list)
    
    # Combine
    x_s = np.concatenate([x_bias, x_uni])[:N]
    y_s = np.concatenate([y_bias, y_uni])[:N]
    
    # Non-uniform time sampling
    t_s = sample_nonuniform_time(N)
    
    return (torch.tensor(x_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_s, dtype=torch.float32, device=DEVICE))


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL CONDITION SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_ic(N):
    """Sample initial condition points at t*=0"""
    count = 0
    x_list, y_list = [], []
    while count < N:
        x_c = np.random.uniform(0, X_MAX_STAR, N * 2)
        y_c = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, N * 2)
        mask = in_domain_star(x_c, y_c)
        valid_x = x_c[mask]
        valid_y = y_c[mask]
        needed = N - count
        if len(valid_x) > 0:
            take = min(len(valid_x), needed)
            x_list.append(valid_x[:take])
            y_list.append(valid_y[:take])
            count += take
    
    x_s = np.concatenate(x_list)[:N]
    y_s = np.concatenate(y_list)[:N]
    t_s = np.zeros((N, 1))
    
    return (torch.tensor(x_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_s, dtype=torch.float32, device=DEVICE))


# ─────────────────────────────────────────────────────────────────────────────
# BOUNDARY CONDITION SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_bc_top(N):
    """Top surface (y* = Y_MAX_STAR = +1.0)"""
    x_s = np.random.uniform(0, X_MAX_STAR, N)
    y_s = np.full(N, Y_MAX_STAR)
    t_s = sample_nonuniform_time(N)
    return (torch.tensor(x_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_s, dtype=torch.float32, device=DEVICE))


def sample_bc_bottom(N):
    """Bottom surface (y* = Y_MIN_STAR = -1.0)"""
    x_s = np.random.uniform(0, X_MAX_STAR, N)
    y_s = np.full(N, Y_MIN_STAR)
    t_s = sample_nonuniform_time(N)
    return (torch.tensor(x_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_s, dtype=torch.float32, device=DEVICE))


def sample_bc_left(N):
    """Left symmetry surface (x* = 0)"""
    x_s = np.zeros(N)
    y_s = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, N)
    t_s = sample_nonuniform_time(N)
    return (torch.tensor(x_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_s, dtype=torch.float32, device=DEVICE))


def sample_bc_right(N):
    """Right edge (x* = X_MAX_STAR = 1.0)"""
    x_s = np.full(N, X_MAX_STAR)
    y_s = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, N)
    t_s = sample_nonuniform_time(N)
    return (torch.tensor(x_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_s, dtype=torch.float32, device=DEVICE))


def sample_bc_inner(N):
    """Inner curved surface (coolant channel)"""
    theta = np.random.uniform(0, np.pi/2, N)
    x_s = R_INNER_STAR * np.cos(theta)
    y_s = R_INNER_STAR * np.sin(theta)
    t_s = sample_nonuniform_time(N)
    
    nx = np.cos(theta)[:, None]
    ny = np.sin(theta)[:, None]
    
    return (torch.tensor(x_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_s, dtype=torch.float32, device=DEVICE),
            torch.tensor(nx, dtype=torch.float32, device=DEVICE),
            torch.tensor(ny, dtype=torch.float32, device=DEVICE))


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_single_interface(r_if_star, N):
    """Sample points on a single circular interface"""
    theta = np.random.uniform(0, np.pi/2, N)
    x_m = (r_if_star - 1e-5) * np.cos(theta)
    y_m = (r_if_star - 1e-5) * np.sin(theta)
    x_p = (r_if_star + 1e-5) * np.cos(theta)
    y_p = (r_if_star + 1e-5) * np.sin(theta)
    t_if = sample_nonuniform_time(N)
    
    nx = np.cos(theta)[:, None]
    ny = np.sin(theta)[:, None]
    
    return (torch.tensor(x_m[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_m[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(x_p[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_p[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_if, dtype=torch.float32, device=DEVICE),
            torch.tensor(nx, dtype=torch.float32, device=DEVICE),
            torch.tensor(ny, dtype=torch.float32, device=DEVICE))


def sample_all_interfaces(N_per_interface):
    """Sample all 5 material interfaces"""
    interfaces = []
    radii = [R_CuCrZr_STAR, R_Cu_STAR, R_FGM1_STAR, R_FGM2_STAR, R_FGM3_STAR]
    for r_if in radii:
        interfaces.append(sample_single_interface(r_if, N_per_interface))
    return interfaces


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_validation(N):
    """Sample validation points with different random seed"""
    np.random.seed(999)  # Different seed from training
    count = 0
    x_list, y_list = [], []
    while count < N:
        x_c = np.random.uniform(0, X_MAX_STAR, N * 2)
        y_c = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, N * 2)
        mask = in_domain_star(x_c, y_c)
        valid_x = x_c[mask]
        valid_y = y_c[mask]
        needed = N - count
        if len(valid_x) > 0:
            take = min(len(valid_x), needed)
            x_list.append(valid_x[:take])
            y_list.append(valid_y[:take])
            count += take
    
    x_s = np.concatenate(x_list)[:N]
    y_s = np.concatenate(y_list)[:N]
    t_s = sample_nonuniform_time(N)
    
    from config import RANDOM_SEED
    np.random.seed(RANDOM_SEED)  # Reset seed
    
    return (torch.tensor(x_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(y_s[:, None], dtype=torch.float32, device=DEVICE),
            torch.tensor(t_s, dtype=torch.float32, device=DEVICE))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data():
    """
    Prepare all collocation points for training.
    Single data preparation (no Phase A/B split).
    """
    print("\nPreparing training data...")
    print(f"  Interior points (interface-biased): {N_INTERIOR:,}")
    print(f"  Initial condition points: {N_IC:,}")
    print(f"  Boundary points: {N_BC_TOP + N_BC_BOTTOM + N_BC_LEFT + N_BC_RIGHT + N_BC_INNER:,}")
    print(f"  Interface points: {5 * N_INTERFACE:,}")
    print(f"  Validation points: {N_VALIDATION:,}")
    print(f"  Time sampling: Non-uniform (50%/30%/20%)")
    
    data = {
        'pde': sample_interior_biased(N_INTERIOR),
        'ic': sample_ic(N_IC),
        'bc_top': sample_bc_top(N_BC_TOP),
        'bc_bot': sample_bc_bottom(N_BC_BOTTOM),
        'bc_left': sample_bc_left(N_BC_LEFT),
        'bc_right': sample_bc_right(N_BC_RIGHT),
        'bc_inner': sample_bc_inner(N_BC_INNER),
        'interfaces': sample_all_interfaces(N_INTERFACE),
        'validation': sample_validation(N_VALIDATION),
    }
    
    print("✓ Data preparation complete\n")
    return data
