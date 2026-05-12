"""
=============================================================================
sampling.py — Collocation Point Sampling with Interface Biasing
=============================================================================
Methods Implemented:
1. Interface-Biased Sampling — 40% of interior points concentrated in:
   - FGM layers (3 thin 0.75mm layers)
   - ±0.2mm around each of 5 material interfaces

Domain (dimensionless):
    x* ∈ [0, 1]       (half width,  symmetry at x*=0)
    y* ∈ [-2, +2]     (full height)
    t* ∈ [0, 1]       (time normalized by t_MAX = 10s)
    Hole: r* < R_INNER* = 0.4286  excluded

All returned tensors are on DEVICE, dtype=float32.
=============================================================================
"""

import torch
import numpy as np
from config import (DEVICE, L_REF, T_MAX, t_REF,
                    X_MAX_STAR, Y_MIN_STAR, Y_MAX_STAR,
                    R_INNER_STAR, R_CuCrZr_STAR, R_Cu_STAR,
                    R_FGM1_STAR, R_FGM2_STAR, R_FGM3_STAR,
                    N_INTERIOR, N_IC, N_BC_TOP, N_BC_BOTTOM,
                    N_BC_INNER, N_BC_LEFT, N_BC_RIGHT, N_INTERFACE,
                    N_VALIDATION)

# Dimensionless time range
T_STAR_MAX = T_MAX / t_REF     # = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _tn(arr):
    """Convert numpy (N,1) float32 array to torch tensor on DEVICE."""
    return torch.tensor(arr, dtype=torch.float32, device=DEVICE)


def in_domain_star(x_s, y_s):
    """
    Boolean mask: True if (x*, y*) is inside the valid domain.
    Excludes the semicircular hole (r* < R_INNER_STAR).
    """
    r_s      = np.sqrt(x_s**2 + y_s**2)
    in_rect  = ((x_s >= 0.0)        & (x_s <= X_MAX_STAR) &
                (y_s >= Y_MIN_STAR) & (y_s <= Y_MAX_STAR))
    out_hole = r_s >= R_INNER_STAR
    return in_rect & out_hole


# ─────────────────────────────────────────────────────────────────────────────
# 1.  INTERIOR PDE POINTS WITH INTERFACE BIASING
# ─────────────────────────────────────────────────────────────────────────────

def sample_interior_biased(N):
    """
    Sample interior PDE collocation points with interface biasing.

    Strategy:
    - 60% uniformly distributed (0.6N points)
    - 40% concentrated in FGM layers and near interfaces (0.4N points)

    FGM layers (thin 0.75mm each):
        - FGM-1 (25%W): r* ∈ [R_Cu_STAR, R_FGM1_STAR]
        - FGM-2 (50%W): r* ∈ [R_FGM1_STAR, R_FGM2_STAR]
        - FGM-3 (75%W): r* ∈ [R_FGM2_STAR, R_FGM3_STAR]

    Interface regions (±0.2mm ≈ ±0.014 dimensionless):
        - CuCrZr|Cu:   r* ≈ R_CuCrZr_STAR ± 0.014
        - Cu|FGM1:     r* ≈ R_Cu_STAR ± 0.014
        - FGM1|FGM2:   r* ≈ R_FGM1_STAR ± 0.014
        - FGM2|FGM3:   r* ≈ R_FGM2_STAR ± 0.014
        - FGM3|W:      r* ≈ R_FGM3_STAR ± 0.014

    Returns (x*, y*, t*) each (N,1) numpy float32.
    """
    interface_width = 0.2e-3 / L_REF  # ±0.2mm in dimensionless units ≈ 0.014

    N_uniform = int(0.6 * N)
    N_biased  = N - N_uniform

    # ── Part 1: Uniform sampling (60%) ────────────────────────────────────────
    pts_uniform = []
    batch = N_uniform * 5
    while len(pts_uniform) < N_uniform:
        xs = np.random.uniform(0.0, X_MAX_STAR, batch)
        ys = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, batch)
        ts = np.random.uniform(0.0, T_STAR_MAX, batch)
        mask = in_domain_star(xs, ys)
        for xi, yi, ti in zip(xs[mask], ys[mask], ts[mask]):
            pts_uniform.append((xi, yi, ti))
            if len(pts_uniform) == N_uniform:
                break

    # ── Part 2: Interface-biased sampling (40%) ───────────────────────────────
    # Define critical radii (FGM layers and interface neighborhoods)
    critical_radii = [
        (R_CuCrZr_STAR - interface_width, R_CuCrZr_STAR + interface_width),  # CuCrZr|Cu
        (R_Cu_STAR - interface_width, R_Cu_STAR + interface_width),            # Cu|FGM1
        (R_Cu_STAR, R_FGM1_STAR),                                              # FGM1 layer
        (R_FGM1_STAR - interface_width, R_FGM1_STAR + interface_width),        # FGM1|FGM2
        (R_FGM1_STAR, R_FGM2_STAR),                                            # FGM2 layer
        (R_FGM2_STAR - interface_width, R_FGM2_STAR + interface_width),        # FGM2|FGM3
        (R_FGM2_STAR, R_FGM3_STAR),                                            # FGM3 layer
        (R_FGM3_STAR - interface_width, R_FGM3_STAR + interface_width),        # FGM3|W
    ]

    pts_biased = []
    batch = N_biased * 10  # larger batch for rejection sampling
    while len(pts_biased) < N_biased:
        xs = np.random.uniform(0.0, X_MAX_STAR, batch)
        ys = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, batch)
        ts = np.random.uniform(0.0, T_STAR_MAX, batch)

        # Compute radii
        rs = np.sqrt(xs**2 + ys**2)

        # Accept points in critical radii
        in_critical = np.zeros(len(xs), dtype=bool)
        for r_min, r_max in critical_radii:
            in_critical |= ((rs >= r_min) & (rs <= r_max))

        mask = in_domain_star(xs, ys) & in_critical
        for xi, yi, ti in zip(xs[mask], ys[mask], ts[mask]):
            pts_biased.append((xi, yi, ti))
            if len(pts_biased) == N_biased:
                break

    # Combine uniform + biased
    pts_all = pts_uniform + pts_biased
    np.random.shuffle(pts_all)  # mix them

    arr = np.array(pts_all, dtype=np.float32)
    return arr[:, 0:1], arr[:, 1:2], arr[:, 2:3]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INITIAL CONDITION POINTS  (t* = 0)
# ─────────────────────────────────────────────────────────────────────────────

def sample_ic(N):
    """
    IC: t* = 0, random (x*, y*) inside domain.
    Returns (x*, y*, t*=0) each (N,1) numpy float32.
    """
    pts = []
    batch = N * 5
    while len(pts) < N:
        xs = np.random.uniform(0.0, X_MAX_STAR, batch)
        ys = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, batch)
        mask = in_domain_star(xs, ys)
        for xi, yi in zip(xs[mask], ys[mask]):
            pts.append((xi, yi, 0.0))
            if len(pts) == N:
                break
    arr = np.array(pts, dtype=np.float32)
    return arr[:, 0:1], arr[:, 1:2], arr[:, 2:3]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  BOUNDARY CONDITION POINTS
# ─────────────────────────────────────────────────────────────────────────────

def sample_bc_top(N):
    """
    Top surface: y* = Y_MAX_STAR = +2.0
    Thermal BC:  -K* dT*/dy* = 1  (Neumann)
    Elastic BC:  σ*·n̂ = 0  (traction-free)
    """
    xs = np.random.uniform(0.0, X_MAX_STAR, (N, 1)).astype(np.float32)
    ys = np.full((N, 1), Y_MAX_STAR, dtype=np.float32)
    ts = np.random.uniform(0.0, T_STAR_MAX, (N, 1)).astype(np.float32)
    return xs, ys, ts


def sample_bc_bottom(N):
    """
    Bottom surface: y* = Y_MIN_STAR = -2.0
    Thermal BC:  +K* dT*/dy* = 1  (Neumann, outward normal is -ŷ)
    Elastic BC:  σ*·n̂ = 0  (traction-free)
    """
    xs = np.random.uniform(0.0, X_MAX_STAR, (N, 1)).astype(np.float32)
    ys = np.full((N, 1), Y_MIN_STAR, dtype=np.float32)
    ts = np.random.uniform(0.0, T_STAR_MAX, (N, 1)).astype(np.float32)
    return xs, ys, ts


def sample_bc_left(N):
    """
    Left symmetry plane: x* = 0
    Two flat regions (above and below curved inner wall):
        - Upper: y* ∈ [R_INNER_STAR, Y_MAX_STAR]
        - Lower: y* ∈ [Y_MIN_STAR, -R_INNER_STAR]

    Thermal BC: dT*/dx* = 0  (insulated — right half domain only)
    Elastic BC: u* = 0  (symmetry)
    """
    N_half = N // 2
    # Upper flat region
    yu = np.random.uniform(R_INNER_STAR, Y_MAX_STAR, (N_half, 1)).astype(np.float32)
    # Lower flat region
    yl = np.random.uniform(Y_MIN_STAR, -R_INNER_STAR, (N - N_half, 1)).astype(np.float32)

    ys = np.concatenate([yu, yl], axis=0)
    xs = np.zeros((N, 1), dtype=np.float32)
    ts = np.random.uniform(0.0, T_STAR_MAX, (N, 1)).astype(np.float32)
    return xs, ys, ts


def sample_bc_right(N):
    """
    Right edge: x* = X_MAX_STAR = 1.0
    Thermal BC: Radiation  -K* dT*/dx* = R_rad * [(C0+T*)^4 - C_env^4]
    Elastic BC: σ*·n̂ = 0  (traction-free)
    """
    xs = np.full((N, 1), X_MAX_STAR, dtype=np.float32)
    ys = np.random.uniform(Y_MIN_STAR, Y_MAX_STAR, (N, 1)).astype(np.float32)
    ts = np.random.uniform(0.0, T_STAR_MAX, (N, 1)).astype(np.float32)
    return xs, ys, ts


def sample_bc_inner_wall(N):
    """
    Inner semicircular wall: r* = R_INNER_STAR, θ ∈ [-π/2, +π/2]
    Thermal BC: Robin  -K* dT*/dn* = Bi(T*) * (T* - T*_cool)
    Elastic BC: σ*·n̂ = 0  (traction-free)
    Returns also (nx, ny) = outward unit normal = (cos θ, sin θ)
    """
    theta = np.random.uniform(-np.pi/2, np.pi/2, (N, 1)).astype(np.float32)
    xs = (R_INNER_STAR * np.cos(theta)).astype(np.float32)
    ys = (R_INNER_STAR * np.sin(theta)).astype(np.float32)
    ts = np.random.uniform(0.0, T_STAR_MAX, (N, 1)).astype(np.float32)
    nx = np.cos(theta).astype(np.float32)
    ny = np.sin(theta).astype(np.float32)
    return xs, ys, ts, nx, ny


# ─────────────────────────────────────────────────────────────────────────────
# 4.  INTERFACE POINTS  (5 circular interfaces)
# ─────────────────────────────────────────────────────────────────────────────

def sample_interface(N, r_star):
    """
    Points just inside (x⁻) and just outside (x⁺) a circular interface
    at dimensionless radius r_star. θ ∈ [-π/2, +π/2] (full semicircle).

    Returns
    -------
    xm, ym : (N,1) — just inside  (minus side)
    xp, yp : (N,1) — just outside (plus side)
    t      : (N,1) — random time
    nx, ny : (N,1) — outward unit normal = (cos θ, sin θ)
    """
    eps = 1e-6
    theta = np.random.uniform(-np.pi/2, np.pi/2, (N, 1)).astype(np.float32)
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)

    xm = ((r_star - eps) * cos_t).astype(np.float32)
    ym = ((r_star - eps) * sin_t).astype(np.float32)
    xp = ((r_star + eps) * cos_t).astype(np.float32)
    yp = ((r_star + eps) * sin_t).astype(np.float32)
    t = np.random.uniform(0.0, T_STAR_MAX, (N, 1)).astype(np.float32)

    return xm, ym, xp, yp, t, cos_t, sin_t


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VALIDATION SET
# ─────────────────────────────────────────────────────────────────────────────

def sample_validation(N):
    """
    Validation points — never used for training, only for checking generalization.
    Same distribution as interior points but completely separate.
    """
    return sample_interior_biased(N)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(phase='A'):
    """
    Sample all collocation points and pack into a data dict.
    All tensors are on DEVICE, dtype=float32.

    Phase A: Thermal training — includes IC, thermal BC, thermal PDE
    Phase B: Elastic training — includes elastic BC, elastic PDE, interfaces
    """
    print(f"\n{'='*70}")
    print(f"  Sampling Collocation Points — Phase {phase}")
    print(f"{'='*70}")

    data = {}

    if phase == 'A':
        # Initial condition
        xi, yi, ti = sample_ic(N_IC)
        data['ic'] = (_tn(xi), _tn(yi), _tn(ti))
        print(f"  Initial condition       : {N_IC:>6,} points")

        # Thermal boundary conditions
        data['bc_top'] = tuple(_tn(a) for a in sample_bc_top(N_BC_TOP))
        data['bc_bot'] = tuple(_tn(a) for a in sample_bc_bottom(N_BC_BOTTOM))
        data['bc_left'] = tuple(_tn(a) for a in sample_bc_left(N_BC_LEFT))
        data['bc_right'] = tuple(_tn(a) for a in sample_bc_right(N_BC_RIGHT))

        xs_iw, ys_iw, ts_iw, nx_iw, ny_iw = sample_bc_inner_wall(N_BC_INNER)
        data['bc_inner'] = (_tn(xs_iw), _tn(ys_iw), _tn(ts_iw),
                            _tn(nx_iw), _tn(ny_iw))

        n_bc_total = N_BC_TOP + N_BC_BOTTOM + N_BC_LEFT + N_BC_RIGHT + N_BC_INNER
        print(f"  Thermal BC (all)        : {n_bc_total:>6,} points")

        # Interior PDE points (interface-biased)
        xp, yp, tp = sample_interior_biased(N_INTERIOR)
        data['pde'] = (_tn(xp), _tn(yp), _tn(tp))
        print(f"  Interior PDE (biased)   : {N_INTERIOR:>6,} points")
        print(f"    → 60% uniform, 40% in FGM + interfaces")

        total = N_IC + n_bc_total + N_INTERIOR

    else:  # phase == 'B'
        # Elastic boundary conditions (reuse same points but for elastic BCs)
        data['bc_top'] = tuple(_tn(a) for a in sample_bc_top(N_BC_TOP))
        data['bc_bot'] = tuple(_tn(a) for a in sample_bc_bottom(N_BC_BOTTOM))
        data['bc_left'] = tuple(_tn(a) for a in sample_bc_left(N_BC_LEFT))
        data['bc_right'] = tuple(_tn(a) for a in sample_bc_right(N_BC_RIGHT))

        xs_iw, ys_iw, ts_iw, nx_iw, ny_iw = sample_bc_inner_wall(N_BC_INNER)
        data['bc_inner'] = (_tn(xs_iw), _tn(ys_iw), _tn(ts_iw),
                            _tn(nx_iw), _tn(ny_iw))

        n_bc_total = N_BC_TOP + N_BC_BOTTOM + N_BC_LEFT + N_BC_RIGHT + N_BC_INNER
        print(f"  Elastic BC (all)        : {n_bc_total:>6,} points")

        # Interior PDE points for elastic
        xp, yp, tp = sample_interior_biased(N_INTERIOR)
        data['pde'] = (_tn(xp), _tn(yp), _tn(tp))
        print(f"  Interior PDE (biased)   : {N_INTERIOR:>6,} points")

        # Five material interfaces
        r_intfs = [R_CuCrZr_STAR, R_Cu_STAR,
                   R_FGM1_STAR, R_FGM2_STAR, R_FGM3_STAR]
        data['interfaces'] = []
        for r_s in r_intfs:
            xm, ym, xp2, yp2, t_if, nx, ny = sample_interface(N_INTERFACE, r_s)
            data['interfaces'].append((
                _tn(xm), _tn(ym), _tn(xp2), _tn(yp2),
                _tn(t_if), _tn(nx), _tn(ny)
            ))
        print(f"  Interface points (×5)   : {N_INTERFACE*5:>6,} points")

        total = n_bc_total + N_INTERIOR + N_INTERFACE * 5

    # Validation set (same for both phases)
    xv, yv, tv = sample_validation(N_VALIDATION)
    data['validation'] = (_tn(xv), _tn(yv), _tn(tv))
    print(f"  Validation set          : {N_VALIDATION:>6,} points")
    print(f"  {'─'*70}")
    print(f"  Total collocation       : {total:>6,} points")
    print(f"  t* range                : [0, {T_STAR_MAX:.3f}]")
    print(f"{'='*70}\n")

    return data