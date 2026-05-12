"""
=============================================================================
resume_training.py — Resume Training from Checkpoint
=============================================================================
Usage:
    python resume_training.py --checkpoint checkpoints/pinn_phaseA_epoch_05000.pt
=============================================================================
"""

import torch
import numpy as np
import argparse
import os

from config import DEVICE, RANDOM_SEED
from network_updated import PINNModel
from sampling_updated import prepare_data
from trainer import train_phase_A, train_phase_B

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='Resume PINN training from checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    set_seed(RANDOM_SEED)
    
    # Detect phase from checkpoint name
    if 'phaseA' in args.checkpoint:
        phase = 'A'
    elif 'phaseB' in args.checkpoint:
        phase = 'B'
    else:
        print("Error: Cannot detect phase from checkpoint name")
        print("Checkpoint name should contain 'phaseA' or 'phaseB'")
        return
    
    print("\n" + "="*70)
    print(f"  RESUMING TRAINING — PHASE {phase}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("="*70 + "\n")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Resuming from epoch {start_epoch}")
    
    if phase == 'A':
        # Prepare data
        data = prepare_data(phase='A')
        
        # Create and load model
        model = PINNModel(phase='A').to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded, continuing training...\n")
        
        # Continue training
        model, history = train_phase_A(model, data, start_epoch=start_epoch)
        
    else:  # phase == 'B'
        # Need thermal model too
        thermal_ckpt = 'checkpoints/pinn_phaseA_final.pt'
        if not os.path.exists(thermal_ckpt):
            print(f"Error: Thermal checkpoint required: {thermal_ckpt}")
            return
        
        # Load thermal model
        model_thermal = PINNModel(phase='A').to(DEVICE)
        thermal_checkpoint = torch.load(thermal_ckpt, map_location=DEVICE)
        model_thermal.load_state_dict(thermal_checkpoint['model_state_dict'])
        
        # Prepare data
        data = prepare_data(phase='B')
        
        # Create and load elastic model
        model_elastic = PINNModel(phase='B').to(DEVICE)
        model_elastic.load_state_dict(checkpoint['model_state_dict'])
        
        
        print(f"Models loaded, continuing training...\n")
        
        # Continue training
        model_elastic, history = train_phase_B(
            model_elastic, model_thermal, data, start_epoch=start_epoch)
    
    print("\n" + "="*70)
    print("  TRAINING RESUMED AND COMPLETED")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
