"""
=============================================================================
main.py — Main Training Script for W/Cu Monoblock PINN
=============================================================================
Usage:
    python main.py --phase A        # Train Phase A (thermal)
    python main.py --phase B        # Train Phase B (elastic, requires Phase A complete)
=============================================================================
"""

import sys

# Force UTF-8 output on Windows terminals (cp1252 default breaks Unicode symbols)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import torch._functorch.config
import numpy as np
import argparse
import os

# donated_buffer optimization in reduce-overhead mode conflicts with
# create_graph=True required for PDE second-order gradients
torch._functorch.config.donated_buffer = False

from config import DEVICE, RANDOM_SEED, print_config
from network_updated import PINNModel
from sampling_updated import prepare_data
from trainer import train_phase_A, train_phase_B
from plotting import plot_all_results

def check_gpu():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU detected: {name}  ({vram:.1f} GB VRAM)")
        torch.cuda.empty_cache()
    else:
        print("WARNING: No GPU detected. Training will be very slow on CPU.")
        print("In Colab: Runtime > Change runtime type > T4 GPU\n")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description='Train PINN for W/Cu Monoblock')
    parser.add_argument('--phase', type=str, required=True, choices=['A', 'B'],
                        help='Training phase: A (thermal) or B (elastic)')
    parser.add_argument('--thermal_checkpoint', type=str, default=None,
                        help='Path to Phase A checkpoint (required for Phase B)')
    args = parser.parse_args()
    
    # GPU check
    check_gpu()

    # Set random seed
    set_seed(RANDOM_SEED)

    # Print configuration
    print_config()
    
    # Enable torch.compile for speedup
    torch.set_float32_matmul_precision('high')
    
    if args.phase == 'A':
        print("\n" + "="*70)
        print("  PHASE A: THERMAL NETWORK TRAINING")
        print("="*70 + "\n")
        
        # Prepare data
        data = prepare_data(phase='A')
        
        # Create model
        model = PINNModel(phase='A').to(DEVICE)
        
        # torch.compile requires a C++ compiler on Windows and conflicts with
        # create_graph=True used in PDE losses; skip it on CPU.
        if torch.cuda.is_available():
            try:
                model = torch.compile(model, mode='reduce-overhead')
                print("✓ Model compiled with torch.compile\n")
            except Exception:
                print("⚠ torch.compile not available, proceeding without compilation\n")
        else:
            print("⚠ Skipping torch.compile on CPU (not beneficial)\n")
        
        # Count parameters
        n_net, n_wts, n_total = model.count_parameters()
        print(f"Network parameters: {n_net:,}")
        print(f"Attention parameters: {n_wts:,}")
        print(f"Total parameters: {n_total:,}\n")
        
        # Train
        model, history = train_phase_A(model, data)
        
        # Plot results
        print("\nGenerating plots...")
        plot_all_results(model, history, './results_phaseA', phase='A')
        
        print("\n" + "="*70)
        print("  PHASE A COMPLETE")
        print("  Next step: python main.py --phase B --thermal_checkpoint checkpoints/pinn_phaseA_final.pt")
        print("="*70 + "\n")
    
    elif args.phase == 'B':
        if args.thermal_checkpoint is None:
            args.thermal_checkpoint = 'checkpoints/pinn_phaseA_final.pt'
        
        if not os.path.exists(args.thermal_checkpoint):
            print(f"Error: Thermal checkpoint not found: {args.thermal_checkpoint}")
            print("Please complete Phase A training first.")
            return
        
        print("\n" + "="*70)
        print("  PHASE B: ELASTIC NETWORK TRAINING")
        print("="*70 + "\n")
        
        # Load thermal model
        print(f"Loading thermal model from: {args.thermal_checkpoint}")
        model_thermal = PINNModel(phase='A').to(DEVICE)
        checkpoint = torch.load(args.thermal_checkpoint, map_location=DEVICE)
        model_thermal.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Thermal model loaded\n")
        
        # Prepare data
        data = prepare_data(phase='B')
        
        # Create elastic model
        model_elastic = PINNModel(phase='B').to(DEVICE)
        
        # torch.compile requires a C++ compiler on Windows and conflicts with
        # create_graph=True used in PDE losses; skip it on CPU.
        if torch.cuda.is_available():
            try:
                model_elastic = torch.compile(model_elastic, mode='reduce-overhead')
                print("✓ Elastic model compiled\n")
            except Exception:
                print("⚠ torch.compile not available\n")
        else:
            print("⚠ Skipping torch.compile on CPU (not beneficial)\n")
        
        # Count parameters
        n_net, n_wts, n_total = model_elastic.count_parameters()
        print(f"Elastic network parameters: {n_net:,}")
        print(f"Attention parameters: {n_wts:,}")
        print(f"Total parameters: {n_total:,}\n")
        
        # Train
        model_elastic, history = train_phase_B(model_elastic, model_thermal, data)
        
        # Plot results
        print("\nGenerating plots...")
        plot_all_results((model_elastic, model_thermal), history, './results_phaseB', phase='B')
        
        print("\n" + "="*70)
        print("  PHASE B COMPLETE")
        print("  All training finished successfully!")
        print("="*70 + "\n")

if __name__ == '__main__':
    main()
