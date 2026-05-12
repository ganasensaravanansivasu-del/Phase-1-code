"""
=============================================================================
trainer.py — Training Loops for Phase A (Thermal) and Phase B (Elastic)
=============================================================================
Implements:
- Curriculum Learning (4 stages per phase)
- Cosine Annealing LR
- Adam → SWA → 100 Adam steps → L-BFGS
- Validation with Early Stopping (Stage 3 only)
- Checkpoint saving every 50 epochs
- Adaptive sampling reweighting every 500 epochs
=============================================================================
"""

import torch
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
import numpy as np
import os
import time

from config import (DEVICE, CURRICULUM_STAGES_A, CURRICULUM_STAGES_B,
                    N_EPOCHS_ADAM, N_EPOCHS_LBFGS, N_EPOCHS_TOTAL,
                    LR_ADAM, LR_ADAM_MIN, SWA_START, SWA_LR, N_ADAM_AFTER_SWA,
                    VALIDATION_EVERY, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
                    ADAPTIVE_SAMPLING_EVERY, SAVE_EVERY, CKPT_DIR)
from losses_updated import *
from sampling_updated import compute_importance_weights

def get_current_stage(epoch, curriculum):
    """Get current curriculum stage"""
    for sid, s in curriculum.items():
        if s['start'] <= epoch < s['end']:
            return sid
    return 4

def print_header(phase):
    """Print training header"""
    hdr = f"{'Ep':>7} | {'Stage':>5} | {'Total':>10} | {'Val':>10} | {'LR':>8} | {'Time':>6}"
    sep = "─" * len(hdr)
    print("=" * 70)
    print(f"  PINN Training — Phase {phase}")
    print("=" * 70)
    print(hdr)
    print(sep)
    return sep

# ═════════════════════════════════════════════════════════════════════════════
# PHASE A — THERMAL TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_phase_A(model, data, start_epoch=1):
    """
    Phase A: Train thermal network only.

    Returns:
        model: trained thermal model
        history: training history dict
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    sep = print_header('A')
    
    history = {
        'total': [], 'ic': [], 'thermal_bc': [], 'thermal_pde': [],
        'validation': [], 'lr': [], 'epoch_time': []
    }
    
    optimizer = optim.Adam(model.parameters(), lr=LR_ADAM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS_ADAM, eta_min=LR_ADAM_MIN)
    
    # SWA
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # Adaptive sampling state
    residual_weights = None
    
    t0_total = time.time()
    
    for epoch in range(start_epoch, N_EPOCHS_ADAM + 1):
        model.train()
        ep_t0 = time.time()
        
        # Get current stage
        stage = get_current_stage(epoch, CURRICULUM_STAGES_A)
        active = CURRICULUM_STAGES_A[stage]['losses']
        
        # Reset attention weights at stage transitions
        if epoch in [3000, 8000, 15000]:
            model.reset_attention_weights()
            print(f"  [Stage {stage}] Attention weights reset")
        
        # Compute losses
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
            L_bc = L_top + L_bot + L_left + L_right + L_inner
            loss_dict['thermal_bc'] = L_bc.item()
            total_loss = total_loss + w['thermal_bc'] * L_bc
        
        if 'thermal_pde' in active:
            L_pde = loss_thermal_pde(model, *data['pde'])
            loss_dict['thermal_pde'] = L_pde.item()
            total_loss = total_loss + w['thermal_pde'] * L_pde
        
        loss_dict['total'] = total_loss.item()
        
        # Backward and step
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update scheduler
        if epoch < SWA_START:
            scheduler.step()
        elif epoch == SWA_START:
            print(f"  [Epoch {epoch}] Starting SWA")
        elif epoch < N_EPOCHS_ADAM - N_ADAM_AFTER_SWA:
            if epoch % 100 == 0:
                swa_model.update_parameters(model)
            swa_scheduler.step()
        elif epoch == N_EPOCHS_ADAM - N_ADAM_AFTER_SWA:
            # Switch from SWA to 100 Adam steps
            print(f"  [Epoch {epoch}] SWA complete, final 100 Adam steps")
            torch.optim.swa_utils.update_bn(torch.utils.data.DataLoader(
                [(data['pde'][0][:100], data['pde'][1][:100], data['pde'][2][:100])],
                batch_size=100), swa_model)
            model.load_state_dict(swa_model.module.state_dict())
            optimizer = optim.Adam(model.parameters(), lr=LR_ADAM_MIN)
        
        ep_time = time.time() - ep_t0
        lr = optimizer.param_groups[0]['lr']
        
        # Record history
        history['total'].append(loss_dict['total'])
        history['ic'].append(loss_dict.get('ic', 0.0))
        history['thermal_bc'].append(loss_dict.get('thermal_bc', 0.0))
        history['thermal_pde'].append(loss_dict.get('thermal_pde', 0.0))
        history['lr'].append(lr)
        history['epoch_time'].append(ep_time)
        
        # Validation
        if epoch % VALIDATION_EVERY == 0:
            val_loss = compute_validation_loss_A(model, *data['validation'])
            history['validation'].append(val_loss)
            
            # Early stopping (only in Stage 3)
            if stage == 3:
                if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += VALIDATION_EVERY
                    
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  [Early Stop] No improvement for {EARLY_STOP_PATIENCE} epochs")
                    print(f"  Best epoch: {best_epoch}, Best val loss: {best_val_loss:.4e}")
                    break
        else:
            history['validation'].append(history['validation'][-1] if history['validation'] else 0.0)
        
        # Adaptive sampling reweighting
        if epoch % ADAPTIVE_SAMPLING_EVERY == 0 and 'thermal_pde' in active:
            # Compute residuals on PDE points (autograd required — no torch.no_grad)
            x_p, y_p, t_p = data['pde']
            x_r = x_p.detach().requires_grad_(True)
            y_r = y_p.detach().requires_grad_(True)
            t_r = t_p.detach().requires_grad_(True)

            T_s, _, _ = model(x_r, y_r, t_r)
            props = get_props_star(x_r, y_r, T_s)

            dT_dt = grad(T_s, t_r)
            dT_dx = grad(T_s, x_r)
            dT_dy = grad(T_s, y_r)

            d_KdTdx_dx = grad(props['K_star'] * dT_dx, x_r)
            d_KdTdy_dy = grad(props['K_star'] * dT_dy, y_r)

            fo_inv = torch.tensor(float(FO_INV), dtype=torch.float32, device=DEVICE)
            R_T = fo_inv * props['rho_star'] * props['cp_star'] * dT_dt - (d_KdTdx_dx + d_KdTdy_dy)

            with torch.no_grad():
                residuals = R_T.squeeze(-1)**2
                residual_weights = compute_importance_weights(residuals)
        
        # Checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'pinn_phaseA_epoch_{epoch:05d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if epoch < SWA_START else None,
                'history': history,
                'loss_dict': loss_dict,
            }, ckpt_path)
        
        # Print progress
        if epoch % 50 == 0 or epoch == 1:
            val = history['validation'][-1]
            eta = (time.time() - t0_total) / epoch * (N_EPOCHS_ADAM - epoch) / 60
            print(f"{epoch:>7d} | {stage:>5d} | {loss_dict['total']:>10.3e} | "
                  f"{val:>10.3e} | {lr:>8.2e} | {ep_time:>6.2f}s  ETA:{eta:.1f}m")
    
    # Final checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'pinn_phaseA_final.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'history': history,
    }, ckpt_path)
    
    print(sep)
    print(f"Phase A complete. Time: {(time.time()-t0_total)/60:.1f} min")
    print(f"Final model saved: {ckpt_path}\n")
    
    return model, history


# ═════════════════════════════════════════════════════════════════════════════
# PHASE B — ELASTIC TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_phase_B(model_elastic, model_thermal_frozen, data, start_epoch=1):
    """
    Phase B: Train elastic network using frozen thermal predictions.

    Returns:
        model_elastic: trained elastic model
        history: training history dict
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    sep = print_header('B')
    
    # Freeze thermal model
    model_thermal_frozen.freeze_for_phase_B()
    
    history = {
        'total': [], 'elastic_bc': [], 'elastic_pde': [], 'interface': [],
        'L1': [], 'L2': [], 'L3': [], 'L4': [],
        'validation': [], 'lr': [], 'epoch_time': []
    }
    
    optimizer = optim.Adam(model_elastic.parameters(), lr=LR_ADAM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=N_EPOCHS_ADAM, eta_min=LR_ADAM_MIN)
    
    # Interface normalizer
    interface_normalizer = InterfaceNormalizer()
    
    # SWA
    swa_model = AveragedModel(model_elastic)
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    t0_total = time.time()
    
    for epoch in range(start_epoch, N_EPOCHS_ADAM + 1):
        model_elastic.train()
        ep_t0 = time.time()
        
        stage = get_current_stage(epoch, CURRICULUM_STAGES_B)
        active = CURRICULUM_STAGES_B[stage]['losses']
        
        if epoch in [3000, 8000, 15000]:
            model_elastic.reset_attention_weights()
            print(f"  [Stage {stage}] Attention weights reset")
        
        optimizer.zero_grad()
        
        w = model_elastic.get_loss_weights()
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=DEVICE)
        
        if 'elastic_bc' in active:
            L_top = loss_bc_elastic_top(model_elastic, model_thermal_frozen, *data['bc_top'])
            L_bot = loss_bc_elastic_bottom(model_elastic, model_thermal_frozen, *data['bc_bot'])
            L_left = loss_bc_elastic_left(model_elastic, *data['bc_left'])
            L_right = loss_bc_elastic_right(model_elastic, model_thermal_frozen, *data['bc_right'])
            L_inner = loss_bc_elastic_inner(model_elastic, model_thermal_frozen, *data['bc_inner'])
            L_bc = L_top + L_bot + L_left + L_right + L_inner
            loss_dict['elastic_bc'] = L_bc.item()
            total_loss = total_loss + w['elastic_bc'] * L_bc
        
        if 'elastic_pde' in active:
            L_pde = loss_elastic_pde(model_elastic, model_thermal_frozen, *data['pde'])
            loss_dict['elastic_pde'] = L_pde.item()
            total_loss = total_loss + w['elastic_pde'] * L_pde
        
        if 'interface' in active:
            L_intf, intf_dict = loss_all_interfaces(
                model_elastic, model_thermal_frozen, data['interfaces'], interface_normalizer)
            loss_dict['interface'] = L_intf.item()
            loss_dict.update(intf_dict)
            total_loss = total_loss + w['interface'] * L_intf
        
        loss_dict['total'] = total_loss.item()
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_elastic.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch < SWA_START:
            scheduler.step()
        elif epoch == SWA_START:
            print(f"  [Epoch {epoch}] Starting SWA")
        elif epoch < N_EPOCHS_ADAM - N_ADAM_AFTER_SWA:
            if epoch % 100 == 0:
                swa_model.update_parameters(model_elastic)
            swa_scheduler.step()
        elif epoch == N_EPOCHS_ADAM - N_ADAM_AFTER_SWA:
            print(f"  [Epoch {epoch}] SWA complete, final 100 Adam steps")
            model_elastic.load_state_dict(swa_model.module.state_dict())
            optimizer = optim.Adam(model_elastic.parameters(), lr=LR_ADAM_MIN)
        
        ep_time = time.time() - ep_t0
        lr = optimizer.param_groups[0]['lr']
        
        # Record history
        for key in ['total', 'elastic_bc', 'elastic_pde', 'interface', 'L1', 'L2', 'L3', 'L4']:
            history[key].append(loss_dict.get(key, 0.0))
        history['lr'].append(lr)
        history['epoch_time'].append(ep_time)
        
        # Validation
        if epoch % VALIDATION_EVERY == 0:
            val_loss = compute_validation_loss_B(model_elastic, model_thermal_frozen, *data['validation'])
            history['validation'].append(val_loss)
            
            if stage == 3:
                if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += VALIDATION_EVERY
                    
                if patience_counter >= EARLY_STOP_PATIENCE:
                    print(f"  [Early Stop] No improvement for {EARLY_STOP_PATIENCE} epochs")
                    break
        else:
            history['validation'].append(history['validation'][-1] if history['validation'] else 0.0)
        
        # Checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'pinn_phaseB_epoch_{epoch:05d}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_elastic.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'interface_normalizer': interface_normalizer,
            }, ckpt_path)
        
        # Print progress
        if epoch % 500 == 0 or epoch == 1:
            val = history['validation'][-1]
            eta = (time.time() - t0_total) / epoch * (N_EPOCHS_ADAM - epoch) / 60
            print(f"{epoch:>7d} | {stage:>5d} | {loss_dict['total']:>10.3e} | "
                  f"{val:>10.3e} | {lr:>8.2e} | {ep_time:>6.2f}s  ETA:{eta:.1f}m")
    
    # Final checkpoint
    ckpt_path = os.path.join(CKPT_DIR, 'pinn_phaseB_final.pt')
    torch.save({
        'epoch': epoch,
        'model_elastic_state_dict': model_elastic.state_dict(),
        'model_thermal_state_dict': model_thermal_frozen.state_dict(),
        'history': history,
    }, ckpt_path)
    
    print(sep)
    print(f"Phase B complete. Time: {(time.time()-t0_total)/60:.1f} min")
    print(f"Final model saved: {ckpt_path}\n")
    
    return model_elastic, history
