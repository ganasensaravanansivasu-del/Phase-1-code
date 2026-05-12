"""
=============================================================================
material_props.py — Temperature-dependent material properties
                    using pre-fitted polynomial coefficients
=============================================================================
All polynomials use Horner's method: coeffs[0]*T^n + ... + coeffs[-1]
Temperature T must be in DIMENSIONAL Kelvin (converted from T* before call).
All returned properties are DIMENSIONLESS (normalised by reference values).
=============================================================================
"""

import torch
from config import (DEVICE, K_REF, E_REF, ALPHA_REF, RHO_REF, CP_REF,
                    R_INNER, R_CuCrZr, R_Cu, R_FGM1, R_FGM2, R_FGM3,
                    T_REF, DT_REF)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  POLYNOMIAL COEFFICIENT TABLES
#     Format: highest-degree coefficient first (numpy polyval convention)
# ─────────────────────────────────────────────────────────────────────────────

# ── Thermal Conductivity K  [W/(m·K)] ────────────────────────────────────────
KWx        = [-6.8211068211068e-9,  5.40498e-5,  -0.134744,   208.011]
KCux       = [-1.0082743748739e-8,  2.0357e-5,   -0.083297,   423.912]
K50W_50Cux = [ 1.9578544061307e-9,  3.13021e-5,  -0.13179,    277.656]
K75W_25Cux = [ 1.1401095230903e-9,  3.54073e-5,  -0.129008,   236.397]
K25W_75Cux = [ 4.019820024199e-9,   1.79597e-5,  -0.119895,   335.108]
KCuCrZrx   = [-3.12865e-4,          0.378544,     233.946]

# ── CTE alpha  [1/K]  linear poly: a*T + b ───────────────────────────────────
AWx        = [4.256756756756e-10,   4.375277027027e-6]
ACux       = [3.448275862070e-9,    1.57897e-5]
A50W_50Cux = [1.9284482758620e-9,   1.0085e-5]
A75W_25Cux = [1.1685344827586e-9,   7.232619396551e-6]
A25W_75Cux = [2.688362068965e-9,    1.29373e-5]
ACuCrZrx   = [3.6842131578947e-9,   1.56205e-5]

# ── Young's Modulus E  [Pa] ───────────────────────────────────────────────────
EWx        = [0.00999516,   -39.8223,    26397.9,     -1.58551e7,  4.01307e11]
ECux       = [138.889,      -255417.0,    9.45149e7,   1.07741e11]
E50W_50Cux = [165.387,      -312203.0,    1.16888e8,   1.69233e11]
E75W_25Cux = [148.251,      -292628.0,    1.0947e8,    2.38002e11]
E25W_75Cux = [153.878,      -285626.0,    1.06269e8,   1.31591e11]
ECuCrZrx   = [-57017.6,      1.03421e7,   1.28865e11]

# ── Poisson's ratio nu ────────────────────────────────────────────────────────
VWx        = [-6.894650157070043e-9,  2.10011e-5,    0.274439]
VCux       = [0.33]
V50W_50Cux = [ 2.1392921960072e-8,  -1.63235e-5,    0.307946]
V75W_25Cux = [ 3.2089382940108e-8,  -2.44852e-5,    0.296919]
V25W_75Cux = [ 1.069646098003e-8,   -8.161728675135e-6, 0.318973]
VCuCrZrx   = [0.33]

# ── Density rho  [kg/m³] ──────────────────────────────────────────────────────
PWx        = [-1.7690618121557e-10,  7.250565605464e-7,
              -1.00667e-3,            0.454783,  19186.2]
PCux       = [-6.692946914699e-8,    1.22628e-4,  -0.163938,  8969.19]
P50W_50Cux = [-4.102389594676e-8,    7.65772e-5,  -0.106939,  14115.8]
P75W_25Cux = [-2.807395644283e-8,    5.35563e-5,  -0.0784423, 16689.1]
P25W_75Cux = [-5.397383545069e-8,    9.95982e-5,  -0.135436,  11542.5]
PCuCrZrx   = [ 2.19298e-4,          -0.501316,    9088.06]

# ── Specific heat cp  [J/(kg·K)] ─────────────────────────────────────────────
CWx        = [ 6.3178194061408e-9,  -2.81279e-5,   0.0513195,  119.219]
CCux       = [-9.046680167372e-7,    1.60153e-3,  -0.760216,   493.01]
C50W_50Cux = [-4.535572948175e-7,    8.02545e-4,  -0.364258,   307.738]
C75W_25Cux = [-2.2800180227868e-7,   4.03054e-4,  -0.166279,   215.103]
C25W_75Cux = [-6.791127873563e-7,    1.20203e-3,  -0.562236,   400.374]
CCuCrZrx   = [-1.90061e-4,           0.36781,      288.548]

# ── HTC polynomial  h(T) [W/(m²·K)] — cubic poly × 1000 ─────────────────────
#    Valid over T ∈ [293, 523] K
HTCx       = [1.786510591e-5, -1.7920819e-2, 6.117605546, -573.3485334]

# ─────────────────────────────────────────────────────────────────────────────
# 2.  ZONE → COEFFICIENT LOOKUP TABLE
# ─────────────────────────────────────────────────────────────────────────────
_ZONE_COEFFS = {
    #           K coeffs    A coeffs    E coeffs    V coeffs    P coeffs    C coeffs
    'CuCrZr': (KCuCrZrx,  ACuCrZrx,  ECuCrZrx,  VCuCrZrx,  PCuCrZrx,  CCuCrZrx),
    'Cu'    : (KCux,       ACux,       ECux,       VCux,       PCux,       CCux),
    'FGM1'  : (K25W_75Cux, A25W_75Cux, E25W_75Cux, V25W_75Cux, P25W_75Cux, C25W_75Cux),
    'FGM2'  : (K50W_50Cux, A50W_50Cux, E50W_50Cux, V50W_50Cux, P50W_50Cux, C50W_50Cux),
    'FGM3'  : (K75W_25Cux, A75W_25Cux, E75W_25Cux, V75W_25Cux, P75W_25Cux, C75W_25Cux),
    'W'     : (KWx,        AWx,        EWx,        VWx,        PWx,        CWx),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3.  DIFFERENTIABLE POLYNOMIAL EVALUATION  (Horner's method)
# ─────────────────────────────────────────────────────────────────────────────

def poly_eval(coeffs_list, T_tensor):
    """
    Evaluate polynomial on a torch tensor using Horner's method.
    coeffs_list : Python list [c0, c1, ..., cn]  (highest degree first)
    T_tensor    : (N,1) tensor  [Kelvin — dimensional]
    Returns     : (N,1) tensor of property values
    """
    result = torch.zeros_like(T_tensor)
    for c in coeffs_list:
        c_t = torch.tensor(float(c), dtype=torch.float32, device=DEVICE)
        result = result * T_tensor + c_t
    return result

# ─────────────────────────────────────────────────────────────────────────────
# 4.  HTC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_htc_star(T_star):
    """
    Compute dimensionless Biot number: Bi = h(T) * L_ref / K_ref
    T_star : (N,1) dimensionless temperature
    Returns : (N,1) Biot number (dimensionless)
    """
    # Convert T* back to dimensional K for polynomial evaluation
    T_dim = T_REF + T_star * DT_REF                          # (N,1) [K]
    T_dim = T_dim.clamp(293.0, 523.0)                        # clamp to valid range

    # Evaluate cubic polynomial and multiply by 1000
    h = poly_eval(HTCx, T_dim) * 1000.0                      # [W/(m²·K)]
    h = h.clamp(min=1.0)                                      # physical lower bound

    # Biot number
    Bi = h * (torch.tensor(float(
        __import__('config').L_REF), dtype=torch.float32, device=DEVICE)
    ) / float(K_REF)
    return Bi

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN MATERIAL PROPERTY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def get_props_star(x_star, y_star, T_star):
    """
    Compute DIMENSIONLESS material properties at given points.

    Parameters
    ----------
    x_star, y_star : (N,1) tensors  — dimensionless coordinates
    T_star         : (N,1) tensor   — dimensionless temperature

    Returns
    -------
    dict with dimensionless keys:
        K_star, rho_star, cp_star,
        lam_star, mu_star, beta_star
    All are (N,1) tensors.

    Procedure:
    1. Identify material zone from (x_star, y_star)
    2. Convert T_star → T_dimensional for polynomial evaluation
    3. Evaluate polynomials → dimensional properties
    4. Normalise by reference values → dimensionless properties
    5. Compute Lamé constants and beta in dimensionless form
    """
    import config as cfg

    # Convert dimensionless coords to dimensional for zone detection
    x_dim = x_star * cfg.L_REF
    y_dim = y_star * cfg.L_REF
    r_dim = torch.sqrt(x_dim**2 + y_dim**2)               # (N,1) [m]

    # Convert T* to dimensional K for polynomial evaluation
    T_dim = T_REF + T_star * DT_REF                        # (N,1) [K]
    # Clamp to physically meaningful temperature range
    T_dim = T_dim.clamp(293.0, 1800.0)

    N = x_star.shape[0]

    # Allocate dimensional property tensors
    K   = torch.zeros(N, 1, device=DEVICE)
    cte = torch.zeros(N, 1, device=DEVICE)
    E   = torch.zeros(N, 1, device=DEVICE)
    nu  = torch.zeros(N, 1, device=DEVICE)
    rho = torch.zeros(N, 1, device=DEVICE)
    cp  = torch.zeros(N, 1, device=DEVICE)

    # Zone boolean masks (using dimensional radius)
    zone_masks = {
        'CuCrZr': ((r_dim >= R_INNER)   & (r_dim < R_CuCrZr)).squeeze(-1),
        'Cu'    : ((r_dim >= R_CuCrZr)  & (r_dim < R_Cu     )).squeeze(-1),
        'FGM1'  : ((r_dim >= R_Cu)      & (r_dim < R_FGM1   )).squeeze(-1),
        'FGM2'  : ((r_dim >= R_FGM1)    & (r_dim < R_FGM2   )).squeeze(-1),
        'FGM3'  : ((r_dim >= R_FGM2)    & (r_dim < R_FGM3   )).squeeze(-1),
        'W'     : ( r_dim >= R_FGM3                           ).squeeze(-1),
    }

    for zone, idx in zone_masks.items():
        if not idx.any():
            continue
        T_z = T_dim[idx]                                   # (Nz,1)
        kc, ac, ec, vc, pc, cc = _ZONE_COEFFS[zone]

        K  [idx] = poly_eval(kc, T_z)
        cte[idx] = poly_eval(ac, T_z)
        E  [idx] = poly_eval(ec, T_z)
        nu [idx] = poly_eval(vc, T_z)
        rho[idx] = poly_eval(pc, T_z)
        cp [idx] = poly_eval(cc, T_z)

    # Physical clamping
    K   = K.clamp(min=1.0)
    cte = cte.clamp(min=1e-7)
    E   = E.clamp(min=1e8)
    nu  = nu.clamp(min=0.10, max=0.49)
    rho = rho.clamp(min=100.0)
    cp  = cp.clamp(min=50.0)

    # ── Dimensionless properties ──────────────────────────────────────────────
    K_star   = K   / K_REF
    rho_star = rho / RHO_REF
    cp_star  = cp  / CP_REF
    E_star   = E   / E_REF
    cte_star = cte / ALPHA_REF

    # ── Dimensionless Lamé parameters ─────────────────────────────────────────
    #  lambda* = nu/[(1+nu)(1-2nu)] * E*
    #  mu*     = E* / [2(1+nu)]
    #  beta*   = (cte* * E*) / (1 - 2*nu)
    lam_star  = (nu / ((1.0 + nu) * (1.0 - 2.0*nu))) * E_star
    mu_star   = E_star / (2.0 * (1.0 + nu))
    beta_star = (cte_star * E_star) / (1.0 - 2.0*nu)

    return dict(
        K_star   = K_star,
        rho_star = rho_star,
        cp_star  = cp_star,
        lam_star = lam_star,
        mu_star  = mu_star,
        beta_star= beta_star,
    )
