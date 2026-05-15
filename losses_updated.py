"""
=============================================================================
losses_updated.py — All Loss Functions for Single Network PINN
=============================================================================
CORRECTED VERSION:
- Single network (no Phase A/B split)
- Corrected BCs: top=flux, bottom/right/left=insulated, inner=convection
- All thermal and elastic losses computed together
=============================================================================
"""

import torch
from config import (DEVICE, T_COOL_STAR, FO_INV,
                    INTERFACE_NORM_WARMUP, INTERFACE_NORM_EPSILON)
from material_props import get_props_star, get_htc_star

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-DIFFERENTIATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def grad(output, inp):
    """∂output/∂inp — both (N,1). Retains graph for higher-order grads."""
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]


# ═════════════════════════════════════════════════════════════════════════════
# INITIAL CONDITION LOSS
# ═════════════════════════════════════════════════════════════════════════════

def loss_ic(model, x_s, y_s, t_s):
    """
    Initial condition - with hard IC enforcement, this should be ~0.
    Kept for monitoring.
    """
    T_s, u_s, v_s = model(x_s, y_s, t_s)
    return torch.mean(T_s**2 + u_s**2 + v_s**2)


# ═════════════════════════════════════════════════════════════════════════════
# THERMAL BOUNDARY CONDITION LOSSES (CORRECTED)
# ═════════════════════════════════════════════════════════════════════════════

def loss_bc_thermal_top(model, x_s, y_s, t_s):
    """Top surface: Heat flux -K* ∂T*/∂y* = 1"""
    y_r = y_s.detach().requires_grad_(True)
    T_s, _, _ = model(x_s, y_r, t_s)

    with torch.no_grad():
        props = get_props_star(x_s.detach(), y_r.detach(), T_s.detach())
    K_s = props['K_star']

    dT_dy = grad(T_s, y_r)
    return torch.mean((-K_s * dT_dy - 1.0)**2)


def loss_bc_thermal_bottom(model, x_s, y_s, t_s):
    """Bottom surface: INSULATED ∂T*/∂y* = 0 (CORRECTED - no heat flux)"""
    y_r = y_s.detach().requires_grad_(True)
    T_s, _, _ = model(x_s, y_r, t_s)
    dT_dy = grad(T_s, y_r)
    return torch.mean(dT_dy**2)


def loss_bc_thermal_left(model, x_s, y_s, t_s):
    """Left symmetry: ∂T*/∂x* = 0"""
    x_r = x_s.detach().requires_grad_(True)
    T_s, _, _ = model(x_r, y_s, t_s)
    dT_dx = grad(T_s, x_r)
    return torch.mean(dT_dx**2)


def loss_bc_thermal_right(model, x_s, y_s, t_s):
    """Right edge: INSULATED ∂T*/∂x* = 0 (CORRECTED - no radiation)"""
    x_r = x_s.detach().requires_grad_(True)
    T_s, _, _ = model(x_r, y_s, t_s)
    dT_dx = grad(T_s, x_r)
    return torch.mean(dT_dx**2)


def loss_bc_thermal_inner(model, x_s, y_s, t_s, nx, ny):
    """Inner wall: Robin convection BC"""
    x_r = x_s.detach().requires_grad_(True)
    y_r = y_s.detach().requires_grad_(True)
    T_s, _, _ = model(x_r, y_r, t_s)
    
    with torch.no_grad():
        props = get_props_star(x_r.detach(), y_r.detach(), T_s.detach())
        Bi = get_htc_star(T_s.detach())
    K_s = props['K_star']
    
    dT_dx = grad(T_s, x_r)
    dT_dy = grad(T_s, y_r)
    dT_dn = dT_dx * nx + dT_dy * ny
    T_cool_s = torch.tensor(float(T_COOL_STAR), dtype=torch.float32, device=DEVICE)
    return torch.mean((-K_s * dT_dn - Bi * (T_s - T_cool_s))**2)


# ═════════════════════════════════════════════════════════════════════════════
# ELASTIC BOUNDARY CONDITION LOSSES
# ═════════════════════════════════════════════════════════════════════════════

def loss_bc_elastic_top(model, x_s, y_s, t_s):
    """Top surface: traction-free"""
    x_r = x_s.detach().requires_grad_(True)
    y_r = y_s.detach().requires_grad_(True)
    
    T_s, u_s, v_s = model(x_r, y_r, t_s)
    
    with torch.no_grad():
        props = get_props_star(x_r.detach(), y_r.detach(), T_s.detach())
    
    lam_s = props['lam_star']
    mu_s = props['mu_star']
    beta_s = props['beta_star']
    
    du_dx = grad(u_s, x_r)
    dv_dy = grad(v_s, y_r)
    du_dy = grad(u_s, y_r)
    dv_dx = grad(v_s, x_r)
    
    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)
    
    s_yy = lam_s*(exx+eyy) + 2.0*mu_s*eyy - beta_s*T_s
    s_xy = 2.0*mu_s*exy
    
    return torch.mean(s_yy**2 + s_xy**2)


def loss_bc_elastic_bottom(model, x_s, y_s, t_s):
    """Bottom surface: traction-free"""
    x_r = x_s.detach().requires_grad_(True)
    y_r = y_s.detach().requires_grad_(True)
    
    T_s, u_s, v_s = model(x_r, y_r, t_s)
    
    with torch.no_grad():
        props = get_props_star(x_r.detach(), y_r.detach(), T_s.detach())
    
    lam_s = props['lam_star']
    mu_s = props['mu_star']
    beta_s = props['beta_star']
    
    du_dx = grad(u_s, x_r)
    dv_dy = grad(v_s, y_r)
    du_dy = grad(u_s, y_r)
    dv_dx = grad(v_s, x_r)
    
    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)
    
    s_yy = lam_s*(exx+eyy) + 2.0*mu_s*eyy - beta_s*T_s
    s_xy = 2.0*mu_s*exy
    
    return torch.mean(s_yy**2 + s_xy**2)


def loss_bc_elastic_left(model, x_s, y_s, t_s):
    """Left symmetry: u* = 0"""
    _, u_s, _ = model(x_s, y_s, t_s)
    return torch.mean(u_s**2)


def loss_bc_elastic_right(model, x_s, y_s, t_s):
    """Right edge: traction-free"""
    x_r = x_s.detach().requires_grad_(True)
    y_r = y_s.detach().requires_grad_(True)
    
    T_s, u_s, v_s = model(x_r, y_r, t_s)
    
    with torch.no_grad():
        props = get_props_star(x_r.detach(), y_r.detach(), T_s.detach())
    
    lam_s = props['lam_star']
    mu_s = props['mu_star']
    beta_s = props['beta_star']
    
    du_dx = grad(u_s, x_r)
    dv_dy = grad(v_s, y_r)
    du_dy = grad(u_s, y_r)
    dv_dx = grad(v_s, x_r)
    
    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)
    
    s_xx = lam_s*(exx+eyy) + 2.0*mu_s*exx - beta_s*T_s
    s_xy = 2.0*mu_s*exy
    
    return torch.mean(s_xx**2 + s_xy**2)


def loss_bc_elastic_inner(model, x_s, y_s, t_s, nx, ny):
    """Inner wall: traction-free"""
    x_r = x_s.detach().requires_grad_(True)
    y_r = y_s.detach().requires_grad_(True)
    
    T_s, u_s, v_s = model(x_r, y_r, t_s)
    
    with torch.no_grad():
        props = get_props_star(x_r.detach(), y_r.detach(), T_s.detach())
    
    lam_s = props['lam_star']
    mu_s = props['mu_star']
    beta_s = props['beta_star']
    
    du_dx = grad(u_s, x_r)
    dv_dy = grad(v_s, y_r)
    du_dy = grad(u_s, y_r)
    dv_dx = grad(v_s, x_r)
    
    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)
    
    s_xx = lam_s*(exx+eyy) + 2.0*mu_s*exx - beta_s*T_s
    s_yy = lam_s*(exx+eyy) + 2.0*mu_s*eyy - beta_s*T_s
    s_xy = 2.0*mu_s*exy
    
    t1 = s_xx * nx + s_xy * ny
    t2 = s_xy * nx + s_yy * ny
    
    return torch.mean(t1**2 + t2**2)


# ═════════════════════════════════════════════════════════════════════════════
# PDE LOSSES
# ═════════════════════════════════════════════════════════════════════════════

def loss_thermal_pde(model, x_s, y_s, t_s):
    """Thermal GDE: 1/Fo · ρ* cp* ∂T*/∂t* - ∇*(K* ∇*T*) = 0"""
    x_r = x_s.detach().requires_grad_(True)
    y_r = y_s.detach().requires_grad_(True)
    t_r = t_s.detach().requires_grad_(True)
    
    T_s, _, _ = model(x_r, y_r, t_r)
    
    with torch.no_grad():
        props = get_props_star(x_r.detach(), y_r.detach(), T_s.detach())
    K_s = props['K_star']
    rho_s = props['rho_star']
    cp_s = props['cp_star']
    
    dT_dt = grad(T_s, t_r)
    dT_dx = grad(T_s, x_r)
    dT_dy = grad(T_s, y_r)
    
    d_KdTdx_dx = grad(K_s * dT_dx, x_r)
    d_KdTdy_dy = grad(K_s * dT_dy, y_r)
    
    fo_inv = torch.tensor(float(FO_INV), dtype=torch.float32, device=DEVICE)
    R_T = fo_inv * rho_s * cp_s * dT_dt - (d_KdTdx_dx + d_KdTdy_dy)
    
    return torch.mean(R_T**2)


def loss_elastic_pde(model, x_s, y_s, t_s):
    """Elastic GDE with thermal forcing"""
    x_r = x_s.detach().requires_grad_(True)
    y_r = y_s.detach().requires_grad_(True)
    
    T_s, u_s, v_s = model(x_r, y_r, t_s)
    
    with torch.no_grad():
        props = get_props_star(x_r.detach(), y_r.detach(), T_s.detach())
    
    lam_s = props['lam_star']
    mu_s = props['mu_star']
    beta_s = props['beta_star']
    
    # First derivatives
    du_dx = grad(u_s, x_r)
    du_dy = grad(u_s, y_r)
    dv_dx = grad(v_s, x_r)
    dv_dy = grad(v_s, y_r)
    dT_dx = grad(T_s, x_r)
    dT_dy = grad(T_s, y_r)
    
    # Second derivatives
    d2u_dx2 = grad(du_dx, x_r)
    d2u_dy2 = grad(du_dy, y_r)
    d2v_dx2 = grad(dv_dx, x_r)
    d2v_dy2 = grad(dv_dy, y_r)
    d2u_dxdy = grad(du_dx, y_r)
    d2v_dxdy = grad(dv_dx, y_r)
    
    # Elastic GDE residuals
    R_ex = (-(lam_s + mu_s) * (d2u_dx2 + d2v_dxdy)
            - mu_s * (d2u_dx2 + d2u_dy2)
            + beta_s * dT_dx)
    
    R_ey = (-(lam_s + mu_s) * (d2u_dxdy + d2v_dy2)
            - mu_s * (d2v_dx2 + d2v_dy2)
            + beta_s * dT_dy)
    
    return torch.mean(R_ex**2 + R_ey**2)


# ═════════════════════════════════════════════════════════════════════════════
# INTERFACE LOSSES
# ═════════════════════════════════════════════════════════════════════════════

class InterfaceNormalizer:
    """Tracks running mean for interface loss components"""
    def __init__(self, warmup=INTERFACE_NORM_WARMUP, epsilon=INTERFACE_NORM_EPSILON):
        self.warmup = warmup
        self.epsilon = epsilon
        self.epoch = 0
        self.running_means = {'L1': 1.0, 'L2': 1.0, 'L3': 1.0, 'L4': 1.0}
        self.ema_alpha = 0.99
        
    def update(self, losses_dict):
        self.epoch += 1
        if self.epoch > self.warmup:
            for key in self.running_means:
                if key in losses_dict:
                    self.running_means[key] = (
                        self.ema_alpha * self.running_means[key] +
                        (1 - self.ema_alpha) * losses_dict[key]
                    )
    
    def normalize(self, losses_dict):
        if self.epoch < self.warmup:
            return losses_dict
        normalized = {}
        for key, val in losses_dict.items():
            mean = self.running_means.get(key, 1.0)
            normalized[key] = val / (mean + self.epsilon)
        return normalized


def loss_single_interface(model, xm, ym, xp, yp, t_if, nx, ny):
    """Interface conditions at single material boundary"""
    # Minus side
    xm_r = xm.detach().requires_grad_(True)
    ym_r = ym.detach().requires_grad_(True)
    
    Tm, um, vm = model(xm_r, ym_r, t_if)
    
    with torch.no_grad():
        pm = get_props_star(xm_r.detach(), ym_r.detach(), Tm.detach())
    
    Km = pm['K_star']
    lam_m = pm['lam_star']
    mu_m = pm['mu_star']
    beta_m = pm['beta_star']
    
    dTm_dx = grad(Tm, xm_r)
    dTm_dy = grad(Tm, ym_r)
    dum_dx = grad(um, xm_r)
    dum_dy = grad(um, ym_r)
    dvm_dx = grad(vm, xm_r)
    dvm_dy = grad(vm, ym_r)
    
    # Plus side
    xp_r = xp.detach().requires_grad_(True)
    yp_r = yp.detach().requires_grad_(True)
    
    Tp, up, vp = model(xp_r, yp_r, t_if)
    
    with torch.no_grad():
        pp = get_props_star(xp_r.detach(), yp_r.detach(), Tp.detach())
    
    Kp = pp['K_star']
    lam_p = pp['lam_star']
    mu_p = pp['mu_star']
    beta_p = pp['beta_star']
    
    dTp_dx = grad(Tp, xp_r)
    dTp_dy = grad(Tp, yp_r)
    dup_dx = grad(up, xp_r)
    dup_dy = grad(up, yp_r)
    dvp_dx = grad(vp, xp_r)
    dvp_dy = grad(vp, yp_r)
    
    # L1: Temperature continuity
    L1 = torch.mean((Tm - Tp)**2)
    
    # L2: Heat flux continuity
    flux_m = Km * (dTm_dx * nx + dTm_dy * ny)
    flux_p = Kp * (dTp_dx * nx + dTp_dy * ny)
    L2 = torch.mean((flux_m - flux_p)**2)
    
    # L3: Displacement continuity
    L3 = torch.mean((um - up)**2 + (vm - vp)**2)
    
    # L4: Traction continuity
    exxm = dum_dx
    eyym = dvm_dy
    exym = 0.5 * (dum_dy + dvm_dx)
    
    s_xxm = lam_m*(exxm+eyym) + 2.0*mu_m*exxm - beta_m*Tm
    s_yym = lam_m*(exxm+eyym) + 2.0*mu_m*eyym - beta_m*Tm
    s_xym = 2.0*mu_m*exym
    
    exxp = dup_dx
    eyyp = dvp_dy
    exyp = 0.5 * (dup_dy + dvp_dx)
    
    s_xxp = lam_p*(exxp+eyyp) + 2.0*mu_p*exxp - beta_p*Tp
    s_yyp = lam_p*(exxp+eyyp) + 2.0*mu_p*eyyp - beta_p*Tp
    s_xyp = 2.0*mu_p*exyp
    
    t1m = s_xxm * nx + s_xym * ny
    t2m = s_xym * nx + s_yym * ny
    t1p = s_xxp * nx + s_xyp * ny
    t2p = s_xyp * nx + s_yyp * ny
    
    L4 = torch.mean((t1m - t1p)**2 + (t2m - t2p)**2)
    
    return {'L1': L1.item(), 'L2': L2.item(), 'L3': L3.item(), 'L4': L4.item()}, (L1, L2, L3, L4)


def loss_all_interfaces(model, interfaces, normalizer):
    """Compute all interface losses with normalization"""
    L1_total = torch.tensor(0.0, device=DEVICE)
    L2_total = torch.tensor(0.0, device=DEVICE)
    L3_total = torch.tensor(0.0, device=DEVICE)
    L4_total = torch.tensor(0.0, device=DEVICE)
    
    for (xm, ym, xp, yp, t_if, nx, ny) in interfaces:
        loss_dict, (L1, L2, L3, L4) = loss_single_interface(
            model, xm, ym, xp, yp, t_if, nx, ny)
        
        L1_total = L1_total + L1
        L2_total = L2_total + L2
        L3_total = L3_total + L3
        L4_total = L4_total + L4
    
    # Average over 5 interfaces
    losses_dict = {
        'L1': L1_total.item() / 5,
        'L2': L2_total.item() / 5,
        'L3': L3_total.item() / 5,
        'L4': L4_total.item() / 5,
    }
    
    normalizer.update(losses_dict)
    normalized = normalizer.normalize(losses_dict)
    
    total = (normalized['L1'] * L1_total / 5 + 
             normalized['L2'] * L2_total / 5 +
             normalized['L3'] * L3_total / 5 + 
             normalized['L4'] * L4_total / 5)
    
    return total, losses_dict


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION LOSS
# ═════════════════════════════════════════════════════════════════════════════

def compute_validation_loss(model, x_v, y_v, t_v):
    """
    Validation loss: thermal + elastic PDE residuals
    """
    x_r = x_v.detach().requires_grad_(True)
    y_r = y_v.detach().requires_grad_(True)
    t_r = t_v.detach().requires_grad_(True)

    T_s, u_s, v_s = model(x_r, y_r, t_r)
    with torch.no_grad():
        props = get_props_star(x_r.detach(), y_r.detach(), T_s.detach())

    # Thermal residual
    K_s = props['K_star']
    rho_s = props['rho_star']
    cp_s = props['cp_star']

    dT_dt = grad(T_s, t_r)
    dT_dx = grad(T_s, x_r)
    dT_dy = grad(T_s, y_r)

    d_KdTdx_dx = grad(K_s * dT_dx, x_r)
    d_KdTdy_dy = grad(K_s * dT_dy, y_r)

    fo_inv = torch.tensor(float(FO_INV), dtype=torch.float32, device=DEVICE)
    R_T = fo_inv * rho_s * cp_s * dT_dt - (d_KdTdx_dx + d_KdTdy_dy)

    # Elastic residual
    lam_s = props['lam_star']
    mu_s = props['mu_star']
    beta_s = props['beta_star']

    du_dx = grad(u_s, x_r)
    du_dy = grad(u_s, y_r)
    dv_dx = grad(v_s, x_r)
    dv_dy = grad(v_s, y_r)

    d2u_dx2 = grad(du_dx, x_r)
    d2u_dy2 = grad(du_dy, y_r)
    d2v_dx2 = grad(dv_dx, x_r)
    d2v_dy2 = grad(dv_dy, y_r)
    d2u_dxdy = grad(du_dx, y_r)
    d2v_dxdy = grad(dv_dx, y_r)

    R_ex = (-(lam_s + mu_s) * (d2u_dx2 + d2v_dxdy)
            - mu_s * (d2u_dx2 + d2u_dy2)
            + beta_s * dT_dx)

    R_ey = (-(lam_s + mu_s) * (d2u_dxdy + d2v_dy2)
            - mu_s * (d2v_dx2 + d2v_dy2)
            + beta_s * dT_dy)

    return torch.mean(R_T**2) + torch.mean(R_ex**2 + R_ey**2)
