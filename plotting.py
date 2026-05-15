"""
=============================================================================
plotting.py — Visualization
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_training_history(history, save_dir):
    """Plot training curves"""
    epochs = np.arange(1, len(history['total']) + 1)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')
    
    # Total loss
    axes[0,0].plot(epochs, history['total'], 'navy', lw=1.0)
    axes[0,0].set_title('Total Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True, alpha=0.3)
    
    # Validation loss + ratio
    if history['validation']:
        val_epochs = np.arange(1, len(history['validation']) + 1)
        axes[0,1].plot(val_epochs, history['validation'], 'darkgreen', lw=1.5, label='Validation')
        axes[0,1].plot(epochs, history['total'], 'navy', lw=1.0, alpha=0.5, label='Training')
        axes[0,1].set_title('Training vs Validation')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_yscale('log')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
    
    # Thermal losses
    axes[0,2].plot(epochs, history['thermal_pde'], 'crimson', label='Thermal PDE', lw=1.0)
    axes[0,2].plot(epochs, history['thermal_bc'], 'orange', label='Thermal BC', lw=1.0)
    axes[0,2].set_title('Thermal Losses')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_yscale('log')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Elastic losses
    axes[1,0].plot(epochs, history['elastic_pde'], 'purple', label='Elastic PDE', lw=1.0)
    axes[1,0].plot(epochs, history['elastic_bc'], 'blue', label='Elastic BC', lw=1.0)
    axes[1,0].set_title('Elastic Losses')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_yscale('log')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Interface losses
    if history['L1']:
        axes[1,1].plot(epochs, history['L1'], label='L1 (Temp)', lw=1.0)
        axes[1,1].plot(epochs, history['L2'], label='L2 (Flux)', lw=1.0)
        axes[1,1].plot(epochs, history['L3'], label='L3 (Disp)', lw=1.0)
        axes[1,1].plot(epochs, history['L4'], label='L4 (Trac)', lw=1.0)
        axes[1,1].set_title('Interface Losses')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_yscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1,2].plot(epochs, history['lr'], 'darkblue', lw=1.0)
    axes[1,2].set_title('Learning Rate')
    axes[1,2].set_xlabel('Epoch')
    axes[1,2].set_yscale('log')
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

def plot_all_results(model, history, save_dir):
    """Generate all plots"""
    os.makedirs(save_dir, exist_ok=True)
    plot_training_history(history, save_dir)
    print(f"All plots saved to: {save_dir}\n")
