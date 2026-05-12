"""
=============================================================================
plotting.py — Visualization and Post-Processing
=============================================================================
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from config import (DEVICE, L_REF, T_REF, DT_REF, U_REF, T_MAX, t_REF,
                    Y_MIN_STAR, Y_MAX_STAR)

def plot_training_history(history, save_dir, phase='A'):
    """Plot training loss curves"""
    epochs = np.arange(1, len(history['total']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training History — Phase {phase}', fontsize=14, fontweight='bold')
    
    # Total loss
    axes[0,0].plot(epochs, history['total'], 'navy', lw=1.0, alpha=0.7)
    axes[0,0].set_title('Total Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('log')
    
    # Validation loss
    if history['validation']:
        val_epochs = np.arange(0, len(history['validation'])) * 500 + 1
        axes[0,1].plot(val_epochs[:len(history['validation'])], 
                       history['validation'], 'darkgreen', lw=1.5)
        axes[0,1].set_title('Validation Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Validation Loss')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].set_yscale('log')
    
    # Component losses
    if phase == 'A':
        axes[1,0].plot(epochs, history['thermal_pde'], 'crimson', label='Thermal PDE', lw=1.0)
        axes[1,0].plot(epochs, history['thermal_bc'], 'darkorange', label='Thermal BC', lw=1.0)
        axes[1,0].plot(epochs, history['ic'], 'darkgreen', label='IC', lw=1.0)
    else:
        axes[1,0].plot(epochs, history['elastic_pde'], 'purple', label='Elastic PDE', lw=1.0)
        axes[1,0].plot(epochs, history['elastic_bc'], 'blue', label='Elastic BC', lw=1.0)
        axes[1,0].plot(epochs, history['interface'], 'teal', label='Interface', lw=1.0)
    
    axes[1,0].set_title('Component Losses')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Loss')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_yscale('log')
    
    # Learning rate
    axes[1,1].plot(epochs, history['lr'], 'darkblue', lw=1.0)
    axes[1,1].set_title('Learning Rate Schedule')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Learning Rate')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    path = os.path.join(save_dir, f'training_history_phase{phase}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_interface_losses(history, save_dir):
    """Plot interface loss components (Phase B only)"""
    if 'L1' not in history or not history['L1']:
        return
    
    epochs = np.arange(1, len(history['L1']) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history['L1'], label='L1 (Temperature)', lw=1.2)
    ax.plot(epochs, history['L2'], label='L2 (Heat Flux)', lw=1.2)
    ax.plot(epochs, history['L3'], label='L3 (Displacement)', lw=1.2)
    ax.plot(epochs, history['L4'], label='L4 (Traction)', lw=1.2)
    
    ax.set_title('Interface Loss Components', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'interface_losses.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_all_results(model, history, save_dir, phase='A'):
    """Generate all plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("Generating plots...")
    
    # Training history
    plot_training_history(history, save_dir, phase)
    
    # Interface losses (Phase B only)
    if phase == 'B':
        plot_interface_losses(history, save_dir)
    
    print(f"All plots saved to: {save_dir}\n")
