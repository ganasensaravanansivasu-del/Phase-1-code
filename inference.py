"""
=============================================================================
inference.py — Load Model and Predict
=============================================================================
Usage:
    python inference.py --checkpoint checkpoints/pinn_final.pt --field --time 5.0
=============================================================================
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from config import DEVICE, L_REF, T_REF, DT_REF, U_REF, t_REF
from network_updated import PINNModel
from sampling_updated import in_domain_star, X_MAX_STAR, Y_MIN_STAR, Y_MAX_STAR

def load_model(checkpoint_path):
    """Load trained model"""
    model = PINNModel().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_field(model, t_sec, resolution=100):
    """Predict fields at given time"""
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
            T_s, u_s, v_s = model(Xv, Yv, Tv)
        
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--field', action='store_true')
    parser.add_argument('--time', type=float, required=True)
    args = parser.parse_args()
    
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint)
    print("✓ Model loaded\n")
    
    if args.field:
        print(f"Generating field at t = {args.time} s...")
        XX, YY, T_g, U_g, V_g = predict_field(model, args.time)
        
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

if __name__ == '__main__':
    main()
