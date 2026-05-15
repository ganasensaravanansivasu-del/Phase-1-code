"""
=============================================================================
config.py — Configuration and Constants for W/Cu Monoblock PINN
=============================================================================
ALL CORRECTIONS APPLIED:
- Y domain: [-14, +14] mm (not [-28, +28])
- Single network (no decoupling)
- CPU only
- Increased dataset for overfitting (20k points)
- Smaller network (128 neurons, 4 layers)
- Dropout + weight decay
- Non-uniform time sampling
=============================================================================
"""

import torch

# ═════════════════════════════════════════════════════════════════════════════
# DEVICE CONFIGURATION (CPU ONLY)
# ═════════════════════════════════════════════════════════════════════════════
DEVICE = 'cpu'  # CPU only, no GPU
RANDOM_SEED = 42

# ═════════════════════════════════════════════════════════════════════════════
# GEOMETRY (DIMENSIONAL - all in SI units: meters)
# ═════════════════════════════════════════════════════════════════════════════
R_INNER  = 6.0e-3    # Inner coolant channel radius [m]
R_CuCrZr = 7.5e-3    # CuCrZr layer outer radius [m]
R_Cu     = 8.25e-3   # Cu interlayer outer radius [m]
R_FGM1   = 9.0e-3    # FGM-1 (25W-75Cu) outer radius [m]
R_FGM2   = 9.75e-3   # FGM-2 (50W-50Cu) outer radius [m]
R_FGM3   = 10.5e-3   # FGM-3 (75W-25Cu) outer radius [m]

# Domain extents
X_MAX    = 14.0e-3   # Half-width (x: 0 to 14mm) [m]
Y_MIN    = -14.0e-3  # CORRECTED: Bottom edge (y: -14mm) [m]
Y_MAX    = +14.0e-3  # CORRECTED: Top edge (y: +14mm) [m]

# ═════════════════════════════════════════════════════════════════════════════
# REFERENCE SCALES FOR DIMENSIONLESS EQUATIONS
# ═════════════════════════════════════════════════════════════════════════════
L_REF     = 14.0e-3                # Length scale [m]
T_REF     = 293.0                  # Reference temperature [K] (20°C)
DT_REF    = 1500.0                 # Temperature scale [K]
K_REF     = 200.0                  # Thermal conductivity [W/(m·K)]
RHO_REF   = 15000.0                # Density [kg/m³]
CP_REF    = 200.0                  # Specific heat [J/(kg·K)]
E_REF     = 2.0e11                 # Young's modulus [Pa]
ALPHA_REF = 1.0e-5                 # CTE [1/K]
U_REF     = ALPHA_REF * DT_REF * L_REF  # Displacement scale [m]

# Time scale (from Fourier number = 1)
t_REF = (RHO_REF * CP_REF * L_REF**2) / K_REF  # [s]

# ═════════════════════════════════════════════════════════════════════════════
# DIMENSIONLESS GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════
R_INNER_STAR  = R_INNER  / L_REF
R_CuCrZr_STAR = R_CuCrZr / L_REF
R_Cu_STAR     = R_Cu     / L_REF
R_FGM1_STAR   = R_FGM1   / L_REF
R_FGM2_STAR   = R_FGM2   / L_REF
R_FGM3_STAR   = R_FGM3   / L_REF

X_MAX_STAR    = X_MAX / L_REF     # = 1.0
Y_MIN_STAR    = Y_MIN / L_REF     # CORRECTED: = -1.0 (was -2.0)
Y_MAX_STAR    = Y_MAX / L_REF     # CORRECTED: = +1.0 (was +2.0)

# ═════════════════════════════════════════════════════════════════════════════
# BOUNDARY CONDITIONS (DIMENSIONAL)
# ═════════════════════════════════════════════════════════════════════════════
Q_TOP    = 10.0e6        # Heat flux on top surface [W/m²] - ONLY TOP!
T_COOL   = 293.0         # Coolant temperature [K]
T_COOL_STAR = (T_COOL - T_REF) / DT_REF  # Dimensionless coolant temp

# Dimensionless parameters
FO_INV = 1.0 / 1.0       # Inverse Fourier number (= 1 by design)

# Bottom and right surfaces: INSULATED (no radiation considered)

# ═════════════════════════════════════════════════════════════════════════════
# TIME DOMAIN
# ═════════════════════════════════════════════════════════════════════════════
T_MAX = 10.0             # Total simulation time [s]
T_STAR_MAX = T_MAX / t_REF  # Dimensionless max time

# ═════════════════════════════════════════════════════════════════════════════
# NETWORK ARCHITECTURE (REDUCED FOR OVERFITTING)
# ═════════════════════════════════════════════════════════════════════════════
HIDDEN_LAYERS  = 4       # REDUCED from 5 (less capacity to overfit)
HIDDEN_NEURONS = 96      # REDUCED from 128
DROPOUT_RATE   = 0.1     # 10% dropout for regularization

# Fourier Features
FF_SIGMA    = [1.0, 5.0, 10.0]  # Multi-scale frequency bands
FF_FEATURES = 96                 # Features per scale
# Total embedding dim: 2 * 96 * 3 = 576

# ═════════════════════════════════════════════════════════════════════════════
# COLLOCATION POINTS (INCREASED FOR OVERFITTING)
# ═════════════════════════════════════════════════════════════════════════════
N_INTERIOR   = 20000     # Full dataset kept in memory; PDE_BATCH_SIZE used during training
N_IC         = 2000      # Initial condition points (t*=0)

# Boundary points
N_BC_TOP     = 800       # INCREASED from 400
N_BC_BOTTOM  = 800       # INCREASED from 400
N_BC_INNER   = 1000      # INCREASED from 500
N_BC_LEFT    = 600       # INCREASED from 300
N_BC_RIGHT   = 600       # INCREASED from 300

# Interface points (5 material interfaces)
N_INTERFACE  = 500       # Points per interface

# Validation
N_VALIDATION = 5000      # INCREASED from 2000 (larger validation set)

# PDE mini-batch size — only this many points used per epoch for PDE losses.
# Prevents OOM caused by second-order autograd graph over all 20k points at once.
PDE_BATCH_SIZE = 2000

# Interface-biased sampling
INTERFACE_BIAS_FRACTION = 0.40  # 40% of interior points near interfaces
INTERFACE_ZONE_WIDTH    = 0.015 # ±0.015 dimensionless units (~0.2mm)

# ═════════════════════════════════════════════════════════════════════════════
# TRAINING HYPERPARAMETERS
# ═════════════════════════════════════════════════════════════════════════════
N_EPOCHS_ADAM   = 15000
N_EPOCHS_LBFGS  = 2000
N_EPOCHS_TOTAL  = N_EPOCHS_ADAM + N_EPOCHS_LBFGS

# Adam optimizer
LR_ADAM     = 1e-3       # Initial learning rate
LR_ADAM_MIN = 1e-5       # Minimum learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization (prevents overfitting)

# SWA (Stochastic Weight Averaging)
SWA_START   = 12000      # Start SWA at epoch 12000
SWA_LR      = 5e-5       # SWA learning rate
N_ADAM_AFTER_SWA = 100   # 100 Adam steps after SWA before L-BFGS

# ═════════════════════════════════════════════════════════════════════════════
# CURRICULUM LEARNING (SINGLE NETWORK)
# ═════════════════════════════════════════════════════════════════════════════
CURRICULUM_STAGES = {
    1: {'start': 1,     'end': 3000,  'losses': ['ic', 'thermal_bc', 'elastic_bc']},
    2: {'start': 3000,  'end': 8000,  'losses': ['ic', 'thermal_bc', 'elastic_bc', 'thermal_pde', 'elastic_pde']},
    3: {'start': 8000,  'end': 15000, 'losses': ['ic', 'thermal_bc', 'elastic_bc', 'thermal_pde', 'elastic_pde', 'interface']},
    4: {'start': 15000, 'end': 17000, 'losses': ['ic', 'thermal_bc', 'elastic_bc', 'thermal_pde', 'elastic_pde', 'interface']},  # L-BFGS
}

# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION AND EARLY STOPPING
# ═════════════════════════════════════════════════════════════════════════════
VALIDATION_EVERY  = 10         # Validate every 10 epochs
EARLY_STOP_PATIENCE = 500      # Stop if no improvement for 500 epochs
EARLY_STOP_MIN_DELTA = 1e-6    # Minimum improvement threshold
EARLY_STOP_LOSS_THRESHOLD = 1e-2  # Stop if both train and val < 1e-2

# Adam → L-BFGS switch criterion (wait for validation to converge)
LBFGS_SWITCH_TRAIN_LOSS = 1e-3  # Training loss threshold
LBFGS_SWITCH_VAL_RATIO = 10.0   # Validation must be within 10× of training

# Validation loss weight in training (prevents overfitting)
VAL_LOSS_WEIGHT = 0.1  # Add 10% of validation loss to training loss

# ═════════════════════════════════════════════════════════════════════════════
# INTERFACE NORMALIZATION
# ═════════════════════════════════════════════════════════════════════════════
INTERFACE_NORM_WARMUP  = 500    # Warmup epochs before normalization
INTERFACE_NORM_EPSILON = 1e-6   # Numerical stability

# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═════════════════════════════════════════════════════════════════════════════
SAVE_EVERY = 50         # Save checkpoint every 500 epochs
CKPT_DIR   = 'checkpoints'

# ═════════════════════════════════════════════════════════════════════════════
# PRINT CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════
def print_config():
    """Print key configuration parameters"""
    print("\n" + "="*70)
    print("  PINN CONFIGURATION (CORRECTED VERSION)")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"\nGeometry (Corrected):")
    print(f"  X domain: [0, {X_MAX*1e3:.1f}] mm")
    print(f"  Y domain: [{Y_MIN*1e3:.1f}, {Y_MAX*1e3:.1f}] mm (CORRECTED)")
    print(f"  Dimensionless: x* ∈ [0, 1], y* ∈ [-1, +1]")
    print(f"\nBoundary Conditions (Corrected):")
    print(f"  Top: Heat flux = {Q_TOP/1e6:.1f} MW/m²")
    print(f"  Bottom: Insulated (∂T/∂y = 0)")
    print(f"  Right: Insulated (∂T/∂x = 0)")
    print(f"  Left: Symmetry (∂T/∂x = 0)")
    print(f"  Inner: Convection (T_cool = {T_COOL:.0f} K)")
    print(f"\nNetwork Architecture (Anti-Overfitting):")
    print(f"  Layers: {HIDDEN_LAYERS}, Neurons: {HIDDEN_NEURONS}")
    print(f"  Dropout: {DROPOUT_RATE:.1%}")
    print(f"  Fourier features: {len(FF_SIGMA)} scales × {FF_FEATURES} = {2*FF_FEATURES*len(FF_SIGMA)}-dim")
    print(f"\nTraining (Anti-Overfitting Strategy):")
    print(f"  Interior points: {N_INTERIOR:,} (2.5× increase)")
    print(f"  Validation points: {N_VALIDATION:,} (2.5× increase)")
    print(f"  Weight decay (L2): {WEIGHT_DECAY}")
    print(f"  Validation loss weight: {VAL_LOSS_WEIGHT:.0%}")
    print(f"  Early stop threshold: {EARLY_STOP_LOSS_THRESHOLD}")
    print(f"\nTime Sampling (Non-Uniform):")
    print(f"  0-2s (transient): 50% of points")
    print(f"  2-4s (moderate): 30% of points")
    print(f"  4-10s (steady): 20% of points")
    print("="*70 + "\n")
