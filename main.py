"""
=============================================================================
main.py — Main Training Script (Single Network, CPU Only)
=============================================================================
Usage:
    python main.py
=============================================================================
"""

import torch
import numpy as np
import os

from config import DEVICE, RANDOM_SEED, print_config
from network_updated import PINNModel
from sampling_updated import prepare_data
from trainer import train_model
from plotting import plot_all_results

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    set_seed(RANDOM_SEED)
    print_config()
    
    print("\n" + "="*80)
    print("  SINGLE NETWORK PINN TRAINING")
    print("="*80 + "\n")
    
    # Prepare data
    data = prepare_data()
    
    # Create model
    model = PINNModel().to(DEVICE)
    
    # Count parameters
    n_net, n_wts, n_total = model.count_parameters()
    print(f"Network parameters: {n_net:,}")
    print(f"Attention parameters: {n_wts:,}")
    print(f"Total parameters: {n_total:,}\n")
    
    # Train
    model, history = train_model(model, data)
    
    # Plot results
    print("\nGenerating plots...")
    plot_all_results(model, history, './results')
    
    print("\n" + "="*80)
    print("  TRAINING COMPLETE")
    print("  Model saved: checkpoints/pinn_final.pt")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
