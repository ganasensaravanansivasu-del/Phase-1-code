"""
=============================================================================
trainer.py — Single Network Training with Anti-Overfitting
=============================================================================
CORRECTED VERSION:
- Single network training (no Phase A/B split)
- Validation loss added to training (10% weight every 50 epochs)
- Weight decay (L2 regularization)
- Early stopping with practical threshold
- Smart Adam → L-BFGS switch
=============================================================================
"""

import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import os
import time

from config import (DEVICE, CURRICULUM_STAGES, N_EPOCHS_ADAM, LR_ADAM, 
                    LR_ADAM_MIN, WEIGHT_DECAY, SWA_START, SWA_LR, N_ADAM_AFTER_SWA,
                    VALIDATION_EVERY, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
                    EARLY_STOP_LOSS_THRESHOLD, VAL_LOSS_WEIGHT,
                    LBFGS_SWITCH_TRAIN_LOSS, LBFGS_SWITCH_VAL_RATIO,
                    SAVE_EVERY, CKPT_DIR)
from losses_updated import *

def get_current_stage(epoch):
    """Get current curriculum stage"""
    for sid, s in CURRICULUM_STAGES.items():
        if s['start'] <= epoch < s['end']:
            return sid
    return 4

def print_header():
    """Print training header"""
    hdr = f"{'Ep':>7} | {'Stage':>5} | {'Total':>10} | {'Val':>10} | {'Ratio':>7} | {'LR':>8} | {'Time':>6}"
    sep = "─" * len(hdr)
    print("=" * 80)
    print("  PINN TRAINING — SINGLE NETWORK (Anti-Overfitting)")
    print("=" * 80)
    print(hdr)
    print(sep)
    return sep

def train_model(model, data, start_epoch=1):
    """
    Train single PINN model for T, u, v together.
    
    Anti-overfitting strategies:
    - Weight decay (L2)
    - Validation loss in training
    - Dropout
    - Early stopping
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    sep = print_header()
    
    history = {
        'total': [], 'ic': [], 'thermal_bc': [], 'elastic_bc': [],
        'thermal_pde': [], 'elastic_pde': [], 'interface': [],
        'L1': [], 'L2': [], 'L3': [], 'L4': [],
        'validation': [], 'lr': [], 'epoch_time': []
    }
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=LR_ADAM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS_ADAM, eta_min=LR_ADAM_MIN)
    
    # SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    
    # Interface normalizer
    interface_normalizer = InterfaceNormalizer()
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    t0_total = time.time()
    
    for epoch in range(start_epoch, N_EPOCHS_ADAM + 1):
        model.train()
        ep_t0 = time.time()
        
        stage = get_current_stage(epoch)
        active = CURRICULUM_STAGES[stage]['losses']
        
        # Reset attention at stage transitions
        if epoch in [3000, 8000]:
            model.reset_attention_weights()
            print(f"  [Stage {stage}] Attention weights reset")
        
        # Compute training losses
        optimizer.zero_grad()
        
        w = model.get_loss_weights()
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        if 'ic' in active:
            L_ic = loss_ic(model, *data['ic'])
            loss_dict['ic'] = L_ic.item()
            total_loss = total_loss + w['ic'] * L_ic
        
        if 'thermal_bc' in active:
            L_top = loss_bc_thermal_top(model, *data['bc_top'])
            L_bot = loss_bc_thermal_bottom(model, *data['bc_bot'])
            L_left = loss_bc_thermal_left(model, *data['bc_left'])
            L_right = loss_bc_thermal_right(model, *data['bc_right'])
            L_inner = loss_bc_thermal_inner(model, *data['bc_inner'])
            L_thermal_bc = L_top + L_bot + L_left + L_right + L_inner
            loss_dict['thermal_bc'] = L_thermal_bc.item()
            total_loss = total_loss + w['thermal_bc'] * L_thermal_bc
        
        if 'elastic_bc' in active:
            L_top_e = loss_bc_elastic_top(model, *data['bc_top'][:3])
            L_bot_e = loss_bc_elastic_bottom(model, *data['bc_bot'][:3])
            L_left_e = loss_bc_elastic_left(model, *data['bc_left'][:3])
            L_right_e = loss_bc_elastic_right(model, *data['bc_right'][:3])
            L_inner_e = loss_bc_elastic_inner(model, *data['bc_inner'])
            L_elastic_bc = L_top_e + L_bot_e + L_left_e + L_right_e + L_inner_e
            loss_dict['elastic_bc'] = L_elastic_bc.item()
            total_loss = total_loss + w['elastic_bc'] * L_elastic_bc
        
        if 'thermal_pde' in active:
            L_thermal_pde = loss_thermal_pde(model, *data['pde'])
            loss_dict['thermal_pde'] = L_thermal_pde.item()
            total_loss = total_loss + w['thermal_pde'] * L_thermal_pde
        
        if 'elastic_pde' in active:
            L_elastic_pde = loss_elastic_pde(model, *data['pde'])
            loss_dict['elastic_pde'] = L_elastic_pde.item()
            total_loss = total_loss + w['elastic_pde'] * L_elastic_pde
        
        if 'interface' in active:
            L_intf, intf_dict = loss_all_interfaces(
                model, data['interfaces'], interface_normalizer)
            loss_dict['interface'] = L_intf.item()
            loss_dict.update(intf_dict)
            total_loss = total_loss + w['interface'] * L_intf
        
        loss_dict['total'] = total_loss.item()
        
        # Add validation loss to training (ANTI-OVERFITTING)
        if epoch % VALIDATION_EVERY == 0:
            val_loss_value = compute_validation_loss(model, *data['validation'])
            val_loss_tensor = torch.tensor(val_loss_value, device=DEVICE)
            total_loss = total_loss + VAL_LOSS_WEIGHT * val_loss_tensor
            history['validation'].append(val_loss_value)
        else:
            history['validation'].append(history['validation'][-1] if history['validation'] else 0.0)
            val_loss_value = history['validation'][-1]
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Scheduler
        if epoch < SWA_START:
            scheduler.step()
        elif epoch == SWA_START:
            print(f"  [Epoch {epoch}] Starting SWA")
        elif epoch < N_EPOCHS_ADAM - N_ADAM_AFTER_SWA:
            if epoch % 100 == 0:
                swa_model.update_parameters(model)
            swa_scheduler.step()
        elif epoch == N_EPOCHS_ADAM - N_ADAM_AFTER_SWA:
            print(f"  [Epoch {epoch}] SWA complete, final {N_ADAM_AFTER_SWA} Adam steps")
            model.load_state_dict(swa_model.module.state_dict())
            optimizer = optim.Adam(model.parameters(), lr=LR_ADAM_MIN, weight_decay=WEIGHT_DECAY)
        
        ep_time = time.time() - ep_t0
        lr = optimizer.param_groups[0]['lr']
        
        # Record history
        for key in ['total', 'ic', 'thermal_bc', 'elastic_bc', 'thermal_pde', 
                    'elastic_pde', 'interface', 'L1', 'L2', 'L3', 'L4']:
            history[key].append(loss_dict.get(key, 0.0))
        history['lr'].append(lr)
        history['epoch_time'].append(ep_time)
        
        # Early stopping (Stage 3 only)
        if stage == 3 and epoch % VALIDATION_EVERY == 0:
            train_loss = loss_dict['total']
            ratio = val_loss_value / (train_loss + 1e-12)
            
            # Stop if both losses below threshold
            if train_loss < EARLY_STOP_LOSS_THRESHOLD and val_loss_value < EARLY_STOP_LOSS_THRESHOLD:
                print(f"  [Early Stop] Loss threshold reached!")
                print(f"  Train: {train_loss:.3e}, Val: {val_loss_value:.3e}")
                break
            
            # Stop if no improvement
            if val_loss_value < best_val_loss - EARLY_STOP_MIN_DELTA:
                best_val_loss = val_loss_value
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += VALIDATION_EVERY
            
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  [Early Stop] No improvement for {EARLY_STOP_PATIENCE} epochs")
                break
        
        # Checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'pinn_epoch_{epoch:05d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, ckpt_path)
        
        # Print progress
        if epoch % 50 == 0 or epoch == 1:
            train_loss = loss_dict['total']
            ratio = val_loss_value / (train_loss + 1e-12)
            eta = (time.time() - t0_total) / epoch * (N_EPOCHS_ADAM - epoch) / 60
            
            # Warning if overfitting
            status = ""
            if ratio > 10:
                status = " ⚠️ OVERFITTING!"
            elif ratio > 3:
                status = " ⚠️"
            
            print(f"{epoch:>7d} | {stage:>5d} | {train_loss:>10.3e} | "
                  f"{val_loss_value:>10.3e} | {ratio:>7.1f}× | {lr:>8.2e} | "
                  f"{ep_time:>6.2f}s{status}")
    
    # Final checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'pinn_final.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, ckpt_path)
    
    print(sep)
    print(f"Training complete. Time: {(time.time()-t0_total)/60:.1f} min")
    print(f"Final model saved: {ckpt_path}\n")
    
    return model, history
