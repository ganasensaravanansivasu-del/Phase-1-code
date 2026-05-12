"""
=============================================================================
inference.py — Load Trained Model and Predict at Any (x, y, t)
=============================================================================
Usage:
    # Predict at single point
    python inference.py --checkpoint checkpoints/pinn_phaseB_final.pt --x 7.0 --y 0.0 --t 10.0
    
    # Generate field predictions
    python inference.py --checkpoint checkpoints/pinn_phaseB_final.pt --field --time 10.0
=============================================================================
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from config import DEVICE, L_REF, T_REF, DT_REF, U_REF, T_MAX, t_REF
from network_updated import PINNModel

def load_models(checkpoint_path):
    """Load both thermal and elastic models from Phase B checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # Load thermal model
    model_thermal = PINNModel(phase='A').to(DEVICE)
    model_thermal.load_state_dict(checkpoint['model_thermal_state_dict'])
    model_thermal.eval()
    
    # Load elastic model
    model_elastic = PINNModel(phase='B').to(DEVICE)
    model_elastic.load_state_dict(checkpoint['model_elastic_state_dict'])
    model_elastic.eval()
    
    return model_thermal, model_elastic

def predict_point(model_thermal, model_elastic, x_mm, y_mm, t_sec):
    """
    Predict T, u, v at a single point in dimensional units.
    
    Parameters:
        x_mm, y_mm: coordinates in mm
        t_sec: time in seconds
    
    Returns:
        T_C: temperature in °C
        u_um: x-displacement in µm
        v_um: y-displacement in µm
    """
    # Convert to dimensionless
    x_star = (x_mm * 1e-3) / L_REF
    y_star = (y_mm * 1e-3) / L_REF
    t_star = t_sec / t_REF
    
    # Create tensors
    x_t = torch.tensor([[x_star]], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor([[y_star]], dtype=torch.float32, device=DEVICE)
    t_t = torch.tensor([[t_star]], dtype=torch.float32, device=DEVICE)
    
    with torch.no_grad():
        T_star, _, _ = model_thermal(x_t, y_t, t_t)
        _, u_star, v_star = model_elastic(x_t, y_t, t_t)
    
    # Convert back to dimensional
    T_K = T_REF + T_star.item() * DT_REF
    T_C = T_K - 273.15
    u_um = u_star.item() * U_REF * 1e6
    v_um = v_star.item() * U_REF * 1e6
    
    return T_C, u_um, v_um

def predict_field(model_thermal, model_elastic, t_sec, resolution=100):
    """
    Predict fields over entire domain at given time.
    
    Returns:
        XX_mm, YY_mm: coordinate grids in mm
        T_C: temperature in °C
        u_um, v_um: displacements in µm
    """
    from sampling_updated import in_domain_star, X_MAX_STAR, Y_MIN_STAR, Y_MAX_STAR
    
    t_star = t_sec / t_REF
    
    x_lin = np.linspace(0.0, X_MAX_STAR, resolution)
    y_lin = np.linspace(Y_MIN_STAR, Y_MAX_STAR, resolution)
    XX_s, YY_s = np.meshgrid(x_lin, y_lin)
    
    Xf = XX_s.flatten()[:, None].astype(np.float32)
    Yf = YY_s.flatten()[:, None].astype(np.float32)
    Tf = np.full_like(Xf, t_star)
    
    mask = in_domain_star(Xf.flatten(), Yf.flatten())
    
    T_out = np.full(len(Xf), np.nan)
    U_out = np.full(len(Xf), np.nan)
    V_out = np.full(len(Xf), np.nan)
    
    if mask.any():
        Xv = torch.tensor(Xf[mask], dtype=torch.float32, device=DEVICE)
        Yv = torch.tensor(Yf[mask], dtype=torch.float32, device=DEVICE)
        Tv = torch.tensor(Tf[mask], dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            T_s, _, _ = model_thermal(Xv, Yv, Tv)
            _, u_s, v_s = model_elastic(Xv, Yv, Tv)
        
        T_out[mask] = (T_REF + T_s.cpu().numpy().flatten() * DT_REF) - 273.15
        U_out[mask] = u_s.cpu().numpy().flatten() * U_REF * 1e6
        V_out[mask] = v_s.cpu().numpy().flatten() * U_REF * 1e6
    
    XX_mm = XX_s * L_REF * 1e3
    YY_mm = YY_s * L_REF * 1e3
    
    return (XX_mm, YY_mm,
            T_out.reshape(resolution, resolution),
            U_out.reshape(resolution, resolution),
            V_out.reshape(resolution, resolution))

def main():
    parser = argparse.ArgumentParser(description='PINN Inference')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Phase B checkpoint')
    parser.add_argument('--field', action='store_true',
                        help='Generate full field prediction')
    parser.add_argument('--x', type=float, default=7.0,
                        help='x coordinate in mm (for point prediction)')
    parser.add_argument('--y', type=float, default=0.0,
                        help='y coordinate in mm (for point prediction)')
    parser.add_argument('--time', type=float, required=True,
                        help='Time in seconds')
    args = parser.parse_args()
    
    print(f"\nLoading models from: {args.checkpoint}")
    model_thermal, model_elastic = load_models(args.checkpoint)
    print("✓ Models loaded\n")
    
    if args.field:
        print(f"Generating field predictions at t = {args.time} s...")
        XX, YY, T_g, U_g, V_g = predict_field(model_thermal, model_elastic, args.time)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        cf0 = axes[0].contourf(XX, YY, T_g, levels=50, cmap='hot')
        axes[0].set_title(f'Temperature (°C) at t={args.time}s')
        plt.colorbar(cf0, ax=axes[0])
        
        cf1 = axes[1].contourf(XX, YY, U_g, levels=50, cmap='RdBu_r')
        axes[1].set_title(f'u-displacement (µm) at t={args.time}s')
        plt.colorbar(cf1, ax=axes[1])
        
        cf2 = axes[2].contourf(XX, YY, V_g, levels=50, cmap='RdBu_r')
        axes[2].set_title(f'v-displacement (µm) at t={args.time}s')
        plt.colorbar(cf2, ax=axes[2])
        
        for ax in axes:
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        filename = f'inference_t{args.time:.0f}s.png'
        plt.savefig(filename, dpi=150)
        print(f"✓ Plot saved: {filename}\n")
        
    else:
        print(f"Predicting at point ({args.x}, {args.y}) mm, t = {args.time} s")
        T_C, u_um, v_um = predict_point(model_thermal, model_elastic, 
                                        args.x, args.y, args.time)
        
        print("\nResults:")
        print(f"  Temperature: {T_C:>10.2f} °C")
        print(f"  u-displacement: {u_um:>10.4f} µm")
        print(f"  v-displacement: {v_um:>10.4f} µm")
        print()

if __name__ == '__main__':
    main()
