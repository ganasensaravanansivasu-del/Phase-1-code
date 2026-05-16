"""
=============================================================================
resume_training.py — Resume from Checkpoint
=============================================================================
Usage:
    python resume_training.py --checkpoint checkpoints/pinn_epoch_05000.pt
=============================================================================
"""

import torch
import numpy as np
import argparse
import os

from config import DEVICE, RANDOM_SEED
from network_updated import PINNModel
from sampling_updated import prepare_data
from trainer import train_model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--stage', type=int, default=None,
                        help='Force starting stage (1 or 2). Overrides checkpoint value.')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    set_seed(RANDOM_SEED)
    
    print("\n" + "="*80)
    print(f"  RESUMING TRAINING")
    print(f"  Checkpoint: {args.checkpoint}")
    print("="*80 + "\n")
    
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    start_epoch = checkpoint['epoch'] + 1
    start_stage = checkpoint.get('current_stage', 1)
    start_stage2_count = checkpoint.get('stage2_converged_count', 0)

    if args.stage is not None:
        start_stage = args.stage
        start_stage2_count = 0
        print(f"  [Override] Forcing start stage to {start_stage}")

    data = prepare_data()
    model = PINNModel().to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Resuming from epoch {start_epoch}, stage {start_stage}\n")

    model, history = train_model(model, data, start_epoch=start_epoch,
                                 start_stage=start_stage,
                                 start_stage2_count=start_stage2_count)
    
    print("\n" + "="*80)
    print("  TRAINING COMPLETE")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
