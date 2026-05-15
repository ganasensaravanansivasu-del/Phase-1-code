"""
=============================================================================
trainer.py — Single Network Training with Anti-Overfitting + L-BFGS Switch
=============================================================================
CORRECTED VERSION WITH L-BFGS:
- Single network training (no Phase A/B split)
- Validation loss added to training (10% weight every 50 epochs)
- Weight decay (L2 regularization)
- Early stopping with practical threshold
- **AUTOMATIC Adam → L-BFGS switch when loss < 1e-3 AND ratio < 10×**
=============================================================================
"""

import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import os
import time

from config import (DEVICE, CURRICULUM_STAGES, N_EPOCHS_ADAM, N_EPOCHS_LBFGS,
                    LR_ADAM, LR_ADAM_MIN, WEIGHT_DECAY, SWA_START, SWA_LR,
                    N_ADAM_AFTER_SWA, VALIDATION_EVERY, VAL_EVAL_EVERY,
                    EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA, EARLY_STOP_LOSS_THRESHOLD,
                    LBFGS_SWITCH_TRAIN_LOSS, LBFGS_SWITCH_VAL_RATIO,
                    SAVE_EVERY, CKPT_DIR, PDE_BATCH_SIZE)
from losses_updated import *

def _pde_minibatch(data):
    """Return a random PDE_BATCH_SIZE subset of data['pde'] to avoid OOM."""
    x, y, t = data['pde']
    n = x.shape[0]
    if n <= PDE_BATCH_SIZE:
        return x, y, t
    idx = torch.randperm(n, device=DEVICE)[:PDE_BATCH_SIZE]
    return x[idx], y[idx], t[idx]


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
    print("  PINN TRAINING — SINGLE NETWORK (Anti-Overfitting + L-BFGS)")
    print("=" * 80)
    print(hdr)
    print(sep)
    return sep


def compute_all_losses(model, data, active, w, interface_normalizer):
    """
    Compute all active losses and return total.
    Used for both regular training and L-BFGS closure.
    """
    total_loss = torch.tensor(0.0, device=DEVICE)
    loss_dict = {}
    
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
    
    if 'thermal_pde' in active or 'elastic_pde' in active:
        pde_batch = _pde_minibatch(data)

    if 'thermal_pde' in active:
        L_thermal_pde = loss_thermal_pde(model, *pde_batch)
        loss_dict['thermal_pde'] = L_thermal_pde.item()
        total_loss = total_loss + w['thermal_pde'] * L_thermal_pde

    if 'elastic_pde' in active:
        L_elastic_pde = loss_elastic_pde(model, *pde_batch)
        loss_dict['elastic_pde'] = L_elastic_pde.item()
        total_loss = total_loss + w['elastic_pde'] * L_elastic_pde
    
    if 'interface' in active:
        L_intf, intf_dict = loss_all_interfaces(
            model, data['interfaces'], interface_normalizer)
        loss_dict['interface'] = L_intf.item()
        loss_dict.update(intf_dict)
        total_loss = total_loss + w['interface'] * L_intf
    
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


def train_with_adam(model, data, optimizer, scheduler, swa_model, swa_scheduler,
                    interface_normalizer, history, start_epoch, max_epoch, sep):
    """
    Train with Adam optimizer until switch to L-BFGS.
    
    Returns:
        epoch: last epoch trained
        switched: whether L-BFGS switch occurred
    """
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    cached_val_loss_tensor = None
    cached_val_loss_value = 0.0
    cached_ratio = 0.0
    switched_to_lbfgs = False
    
    t0_total = time.time()
    
    for epoch in range(start_epoch, max_epoch + 1):
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
        
        total_loss, loss_dict = compute_all_losses(model, data, active, w, interface_normalizer)
        
        # Recompute validation every VAL_EVAL_EVERY epochs; reuse cached values in between
        train_loss = loss_dict['total']
        if epoch % VAL_EVAL_EVERY == 0:
            cached_val_loss_tensor = compute_validation_loss(model, *data['validation'])
            cached_val_loss_value = cached_val_loss_tensor.item()
            cached_ratio = cached_val_loss_value / (train_loss + 1e-12)

        val_loss_value = cached_val_loss_value
        history['validation'].append(val_loss_value)
        ratio = cached_ratio

        if ratio > 10:
            val_weight = 0.2
        elif ratio > 5:
            val_weight = 0.1
        else:
            val_weight = 0.05 if epoch % VAL_EVAL_EVERY == 0 else 0.0

        if val_weight > 0.0 and cached_val_loss_tensor is not None:
            total_loss = total_loss + val_weight * cached_val_loss_tensor
        
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
        
        # Calculate ratio for monitoring
        train_loss = loss_dict['total']
        ratio = val_loss_value / (train_loss + 1e-12)
        
        # ═══════════════════════════════════════════════════════════════════
        # CHECK FOR L-BFGS SWITCH (NEW!)
        # ═══════════════════════════════════════════════════════════════════
        if (stage >= 3 and 
            epoch >= 8000 and 
            not switched_to_lbfgs and
            train_loss < LBFGS_SWITCH_TRAIN_LOSS and
            ratio < LBFGS_SWITCH_VAL_RATIO):
            
            print(f"\n{'='*80}")
            print(f"  🚀 SWITCHING TO L-BFGS OPTIMIZER")
            print(f"{'='*80}")
            print(f"  Epoch: {epoch}")
            print(f"  Training loss: {train_loss:.3e} < {LBFGS_SWITCH_TRAIN_LOSS:.3e} ✓")
            print(f"  Validation ratio: {ratio:.1f}× < {LBFGS_SWITCH_VAL_RATIO:.1f}× ✓")
            print(f"  Switching from Adam to L-BFGS for final refinement...")
            print(f"{'='*80}\n")
            
            switched_to_lbfgs = True
            
            # Save checkpoint before switching
            ckpt_path = os.path.join(CKPT_DIR, f'pinn_before_lbfgs_epoch_{epoch:05d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
            }, ckpt_path)
            print(f"  Checkpoint saved before L-BFGS: {ckpt_path}\n")
            
            return epoch, True, optimizer
        
        # ═══════════════════════════════════════════════════════════════════
        # EARLY STOPPING (Stage 3 only)
        # ═══════════════════════════════════════════════════════════════════
        if stage == 3 and epoch % VALIDATION_EVERY == 0:
            # Stop if both losses below threshold
            if train_loss < EARLY_STOP_LOSS_THRESHOLD and val_loss_value < EARLY_STOP_LOSS_THRESHOLD:
                print(f"\n{'='*80}")
                print(f"  ✓ EARLY STOPPING: Loss Threshold Reached!")
                print(f"{'='*80}")
                print(f"  Training loss: {train_loss:.3e} < {EARLY_STOP_LOSS_THRESHOLD:.3e} ✓")
                print(f"  Validation loss: {val_loss_value:.3e} < {EARLY_STOP_LOSS_THRESHOLD:.3e} ✓")
                print(f"  Model has converged sufficiently.")
                print(f"{'='*80}\n")
                return epoch, False, optimizer
            
            # Stop if no improvement
            if val_loss_value < best_val_loss - EARLY_STOP_MIN_DELTA:
                best_val_loss = val_loss_value
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += VALIDATION_EVERY
            
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n{'='*80}")
                print(f"  ✓ EARLY STOPPING: No Improvement")
                print(f"{'='*80}")
                print(f"  No improvement for {EARLY_STOP_PATIENCE} epochs")
                print(f"  Best epoch: {best_epoch}, Best val loss: {best_val_loss:.3e}")
                print(f"{'='*80}\n")
                return epoch, False, optimizer
        
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
            eta = (time.time() - t0_total) / (epoch - start_epoch + 1) * (max_epoch - epoch) / 60
            
            # Warning if overfitting
            status = ""
            if ratio > 100:
                status = " 🚨 CRITICAL!"
            elif ratio > 10:
                status = " ⚠️ OVERFITTING!"
            elif ratio > 5:
                status = " ⚠️"
            
            print(f"{epoch:>7d} | {stage:>5d} | {train_loss:>10.3e} | "
                  f"{val_loss_value:>10.3e} | {ratio:>7.1f}× | {lr:>8.2e} | "
                  f"{ep_time:>6.2f}s{status}")
    
    return max_epoch, False, optimizer


def train_with_lbfgs(model, data, interface_normalizer, history, start_epoch, max_epochs, sep):
    """
    Train with L-BFGS optimizer for final refinement.
    """
    print(f"\n{'='*80}")
    print(f"  L-BFGS REFINEMENT (Epochs {start_epoch} - {start_epoch + max_epochs})")
    print(f"{'='*80}\n")
    
    # Create L-BFGS optimizer
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        max_eval=25,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=100
    )
    
    for epoch in range(start_epoch, start_epoch + max_epochs):
        model.train()
        ep_t0 = time.time()
        
        stage = 4
        active = CURRICULUM_STAGES[stage]['losses']
        w = model.get_loss_weights()
        
        # L-BFGS requires closure
        def closure():
            optimizer.zero_grad()
            total_loss, _ = compute_all_losses(model, data, active, w, interface_normalizer)
            total_loss.backward()
            return total_loss
        
        optimizer.step(closure)
        
        # Compute losses for logging
        with torch.no_grad():
            total_loss, loss_dict = compute_all_losses(model, data, active, w, interface_normalizer)
        
        ep_time = time.time() - ep_t0
        
        # Validation
        if epoch % VALIDATION_EVERY == 0:
            val_loss_value = compute_validation_loss(model, *data['validation'])
            history['validation'].append(val_loss_value)
        else:
            history['validation'].append(history['validation'][-1] if history['validation'] else 0.0)
            val_loss_value = history['validation'][-1]
        
        # Record history
        for key in ['total', 'ic', 'thermal_bc', 'elastic_bc', 'thermal_pde', 
                    'elastic_pde', 'interface', 'L1', 'L2', 'L3', 'L4']:
            history[key].append(loss_dict.get(key, 0.0))
        history['lr'].append(1.0)
        history['epoch_time'].append(ep_time)
        
        # Print
        if epoch % 10 == 0 or epoch == start_epoch:
            train_loss = loss_dict['total']
            ratio = val_loss_value / (train_loss + 1e-12)
            print(f"{epoch:>7d} | {stage:>5d} | {train_loss:>10.3e} | "
                  f"{val_loss_value:>10.3e} | {ratio:>7.1f}× | L-BFGS | "
                  f"{ep_time:>6.2f}s")
        
        # Checkpoint
        if (epoch - start_epoch) % 100 == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'pinn_lbfgs_epoch_{epoch:05d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
            }, ckpt_path)
    
    return epoch


def train_model(model, data, start_epoch=1):
    """
    Main training with automatic Adam → L-BFGS switch.
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    sep = print_header()
    
    history = {
        'total': [], 'ic': [], 'thermal_bc': [], 'elastic_bc': [],
        'thermal_pde': [], 'elastic_pde': [], 'interface': [],
        'L1': [], 'L2': [], 'L3': [], 'L4': [],
        'validation': [], 'lr': [], 'epoch_time': []
    }
    
    optimizer = optim.Adam(model.parameters(), lr=LR_ADAM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS_ADAM, eta_min=LR_ADAM_MIN)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    interface_normalizer = InterfaceNormalizer()
    
    t0_total = time.time()
    
    # Adam training
    final_adam_epoch, switched, optimizer = train_with_adam(
        model, data, optimizer, scheduler, swa_model, swa_scheduler,
        interface_normalizer, history, start_epoch, N_EPOCHS_ADAM, sep
    )
    
    # L-BFGS refinement (if switched)
    if switched:
        final_epoch = train_with_lbfgs(
            model, data, interface_normalizer, history,
            final_adam_epoch + 1, N_EPOCHS_LBFGS, sep
        )
    else:
        final_epoch = final_adam_epoch
    
    # Final checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'pinn_final.pt')
    torch.save({
        'epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, ckpt_path)
    
    print(sep)
    print(f"Training complete! Time: {(time.time()-t0_total)/60:.1f} min")
    print(f"Final epoch: {final_epoch}, Loss: {history['total'][-1]:.3e}")
    print(f"Model saved: {ckpt_path}")
    print(sep + "\n")
    
    return model, history
