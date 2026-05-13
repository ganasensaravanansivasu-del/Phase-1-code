"""
=============================================================================
config.py — All constants, reference scales, and hyperparameters
=============================================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU-specific settings
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True   # faster matmul on Ampere+
    torch.backends.cudnn.allow_tf32      = True

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
Q_FLUX    = 10.0e6          # W/m²  — heat flux on top and bottom surfaces
T_INIT    = 293.15          # K     — initial temperature (20°C, stress-free)
T_COOL    = 373.15          # K     — coolant temperature (100°C)
T_ENV     = 473.15          # K     — radiation environment temperature (200°C)
EMISSIVITY = 0.3            # —     — W surface emissivity
SIGMA_SB  = 5.67e-8         # W/(m²·K⁴) — Stefan-Boltzmann constant

# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY  (all in metres)
# ─────────────────────────────────────────────────────────────────────────────
mm = 1e-3

X_MIN, X_MAX =  0.0,        14.0 * mm   # half width
Y_MIN, Y_MAX = -28.0 * mm,  28.0 * mm   # full height
T_MIN, T_MAX =  0.0,        10.0        # seconds

R_INNER  = 6.00  * mm   # cooling channel wall (inner radius)
R_CuCrZr = 7.50  * mm   # CuCrZr outer radius
R_Cu     = 8.25  * mm   # OFHC-Cu outer radius
R_FGM1   = 9.00  * mm   # FGM-1 (25%W) outer radius
R_FGM2   = 9.75  * mm   # FGM-2 (50%W) outer radius
R_FGM3   = 10.50 * mm   # FGM-3 (75%W) outer radius
# r > R_FGM3 : W armor

# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE SCALES FOR NON-DIMENSIONALIZATION
# ─────────────────────────────────────────────────────────────────────────────
L_REF   = 14.0  * mm          # m        — half width (x* ∈ [0,1])
K_REF   = 173.0               # W/(m·K)  — W thermal conductivity at 293K
E_REF   = 3.98e11             # Pa       — W Young's modulus at 293K
ALPHA_REF = 17.0e-6           # 1/K      — Cu CTE at 293K (drives mismatch)
RHO_REF = 19250.0             # kg/m³    — W density at 293K
CP_REF  = 132.0               # J/(kg·K) — W specific heat at 293K
T_REF   = T_INIT              # K        — reference temperature
t_REF   = T_MAX               # s        — total simulation time (10s) → t* ∈ [0,1]

# Derived reference scales
DT_REF  = Q_FLUX * L_REF / K_REF               # K       — temperature rise scale
U_REF   = ALPHA_REF * DT_REF * L_REF           # m       — displacement scale
SIG_REF = E_REF * ALPHA_REF * DT_REF           # Pa      — stress scale

# Dimensionless numbers
FO      = K_REF * t_REF / (RHO_REF * CP_REF * L_REF**2)   # Fourier number
FO_INV  = 1.0 / FO                                          # 1/Fo coefficient in thermal PDE

# Radiation coefficient (dimensionless)
R_RAD   = EMISSIVITY * SIGMA_SB * DT_REF**3 / (K_REF / L_REF)

# Dimensionless temperatures
T_COOL_STAR = (T_COOL - T_REF) / DT_REF        # dimensionless coolant temp
T_ENV_STAR  = (T_ENV  - T_REF) / DT_REF        # dimensionless env temp
C0          = T_REF / DT_REF                    # T_ref / DT_ref
C_ENV       = T_ENV / DT_REF                    # T_env / DT_ref (for radiation)

# Dimensionless geometry
R_INNER_STAR  = R_INNER  / L_REF
R_CuCrZr_STAR = R_CuCrZr / L_REF
R_Cu_STAR     = R_Cu     / L_REF
R_FGM1_STAR   = R_FGM1   / L_REF
R_FGM2_STAR   = R_FGM2   / L_REF
R_FGM3_STAR   = R_FGM3   / L_REF

X_MAX_STAR    = X_MAX / L_REF      # = 1.0
Y_MIN_STAR    = Y_MIN / L_REF      # = -2.0
Y_MAX_STAR    = Y_MAX / L_REF      # = +2.0

# ─────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_LAYERS  = 5
HIDDEN_NEURONS = 64
DROPOUT_RATE   = 0.0          # set to 0 — removed as decided

# Multi-scale Fourier Feature parameters
FF_SIGMA       = [1.0, 5.0, 10.0]   # frequency scales
FF_FEATURES    = 128                  # number of features per scale
# Input size = 2 * FF_FEATURES * len(FF_SIGMA) = 768

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
N_EPOCHS_TOTAL  = 20000
N_EPOCHS_ADAM   = 15000       # Adam phase
N_EPOCHS_LBFGS  = 5000        # L-BFGS phase

LR_ADAM         = 1e-3
LR_ADAM_MIN     = 1e-6
WEIGHT_DECAY    = 0.0         # L2 removed as decided

# Cosine annealing (Adam phase only)
T_COSINE        = N_EPOCHS_ADAM   # full cosine period over Adam phase

# ─────────────────────────────────────────────────────────────────────────────
# CURRICULUM LEARNING STAGES — PHASE A (THERMAL) and PHASE B (ELASTIC)
# ─────────────────────────────────────────────────────────────────────────────
# Phase A: Thermal network training only
CURRICULUM_STAGES_A = {
    1: {'start': 0,     'end': 3000,  'losses': ['ic', 'thermal_bc']},
    2: {'start': 3000,  'end': 8000,  'losses': ['ic', 'thermal_bc', 'thermal_pde']},
    3: {'start': 8000,  'end': 15000, 'losses': ['ic', 'thermal_bc', 'thermal_pde']},  # Early stopping active
    4: {'start': 15000, 'end': 20000, 'losses': ['ic', 'thermal_bc', 'thermal_pde']},  # L-BFGS phase
}

# Phase B: Elastic network training using frozen thermal predictions
CURRICULUM_STAGES_B = {
    1: {'start': 0,     'end': 3000,  'losses': ['elastic_bc']},
    2: {'start': 3000,  'end': 8000,  'losses': ['elastic_bc', 'elastic_pde']},
    3: {'start': 8000,  'end': 15000, 'losses': ['elastic_bc', 'elastic_pde', 'interface']},  # Early stopping active
    4: {'start': 15000, 'end': 20000, 'losses': ['elastic_bc', 'elastic_pde', 'interface']},  # L-BFGS phase
}

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION AND EARLY STOPPING
# ─────────────────────────────────────────────────────────────────────────────
N_VALIDATION      = 2000       # number of validation points
VALIDATION_EVERY  = 50         # evaluate validation every N epochs
EARLY_STOP_PATIENCE = 500      # stop if no improvement for N epochs (only in Stage 3)
EARLY_STOP_MIN_DELTA = 1e-6    # minimum improvement to count as improvement
EARLY_STOP_LOSS_THRESHOLD = 1e-2  # stop if both training and validation loss < this value

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE-BIASED SAMPLING (Fixed Dataset)
# ─────────────────────────────────────────────────────────────────────────────
# No additional parameters needed - interface biasing is done during initial sampling

# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE LOSS NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
INTERFACE_NORM_WARMUP = 500          # warmup epochs before normalization activates
INTERFACE_NORM_EPSILON = 1e-6        # epsilon to prevent division by zero

# ─────────────────────────────────────────────────────────────────────────────
# STOCHASTIC WEIGHT AVERAGING (SWA)
# ─────────────────────────────────────────────────────────────────────────────
SWA_START       = 12000       # start SWA at this epoch
SWA_FREQ        = 100         # update SWA weights every N epochs
SWA_LR          = 5e-4        # SWA learning rate
N_ADAM_AFTER_SWA = 100        # 100 Adam steps after SWA before L-BFGS

# ─────────────────────────────────────────────────────────────────────────────
# COLLOCATION POINT COUNTS
# ─────────────────────────────────────────────────────────────────────────────
N_INTERIOR      = 8000        # interior PDE points
N_IC            = 2000        # initial condition points
N_BC_TOP        = 400         # top surface (heat flux)
N_BC_BOTTOM     = 400         # bottom surface (heat flux)
N_BC_INNER      = 600         # inner semicircular wall (convection)
N_BC_LEFT       = 400         # left symmetry plane (x=0)
N_BC_RIGHT      = 400         # right edge (radiation)
N_INTERFACE     = 500         # points per interface × 5 interfaces

# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────
ADAPTIVE_SAMPLING_EVERY = 500   # recompute importance weights every N epochs
ADAPTIVE_ALPHA          = 0.7   # blending: weight = α·residual + (1-α)·uniform

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 42              # fixed seed for reproducibility

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING & SAVING
# ─────────────────────────────────────────────────────────────────────────────
PRINT_EVERY     = 500
SAVE_EVERY      = 50
RESULTS_DIR     = './pinn_results'
CKPT_DIR        = './pinn_checkpoints'

# ─────────────────────────────────────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_config():
    print("=" * 65)
    print("  PINN Configuration Summary")
    print("=" * 65)
    print(f"  Device              : {DEVICE}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU                 : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM                : {props.total_memory / 1e9:.1f} GB")
    else:
        print(f"  WARNING             : No GPU detected — training will be slow!")
    print(f"  Domain              : x* ∈ [0,1], y* ∈ [-2,+2], t* ∈ [0,1]")
    print(f"\n  --- Reference Scales ---")
    print(f"  L_ref               : {L_REF*1e3:.1f} mm")
    print(f"  ΔT_ref              : {DT_REF:.2f} K")
    print(f"  t_ref               : {t_REF:.1f} s  → t* ∈ [0, 1]")
    print(f"  u_ref               : {U_REF*1e6:.4f} µm")
    print(f"  σ_ref               : {SIG_REF:.4e} Pa")
    print(f"\n  --- Dimensionless Numbers ---")
    print(f"  Fourier number (Fo) : {FO:.5f}")
    print(f"  1/Fo                : {FO_INV:.3f}")
    print(f"  R_rad               : {R_RAD:.4e}")
    print(f"  T*_cool             : {T_COOL_STAR:.5f}")
    print(f"  T*_env              : {T_ENV_STAR:.5f}")
    print(f"  C0 (Tref/ΔTref)     : {C0:.5f}")
    print(f"  C_env (Tenv/ΔTref)  : {C_ENV:.5f}")
    print(f"\n  --- Network ---")
    print(f"  Fourier scales (σ)  : {FF_SIGMA}")
    print(f"  Fourier features    : {FF_FEATURES} per scale")
    print(f"  Hidden layers       : {HIDDEN_LAYERS} × {HIDDEN_NEURONS}")
    print(f"\n  --- Training ---")
    print(f"  Total epochs        : {N_EPOCHS_TOTAL}")
    print(f"  Adam epochs         : {N_EPOCHS_ADAM}")
    print(f"  L-BFGS epochs       : {N_EPOCHS_LBFGS}")
    print(f"  Adaptive α          : {ADAPTIVE_ALPHA}")
    print(f"  Adaptive every      : {ADAPTIVE_SAMPLING_EVERY} epochs")
    print("=" * 65)