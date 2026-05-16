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
                    N_ADAM_AFTER_SWA, VAL_EVAL_EVERY,
                    STAGE1_LOSS_THRESHOLD,
                    STAGE2_LOSS_THRESHOLD, STAGE2_RATIO_THRESHOLD, STAGE2_CONVERGE_EPOCHS,
                    LBFGS_LOSS_THRESHOLD, LBFGS_RATIO_THRESHOLD, LBFGS_CONVERGE_EPOCHS,
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
                    interface_normalizer, history, start_epoch, max_epoch, sep,
                    start_stage=1, start_stage2_count=0):
    """
    Adam training with criteria-based stage transitions.
    Stage 1 (BC):      exits when loss < STAGE1_LOSS_THRESHOLD and ratio < STAGE1_RATIO_THRESHOLD
    Stage 2 (BC+PDE):  switches to L-BFGS after STAGE2_CONVERGE_EPOCHS consecutive epochs
                       with loss < STAGE2_LOSS_THRESHOLD and ratio < STAGE2_RATIO_THRESHOLD
    start_stage / start_stage2_count are restored from checkpoint on resume.
    """
    cached_val_loss_value = 0.0
    cached_ratio = 0.0
    current_stage = start_stage
    stage2_converged_count = start_stage2_count

    for epoch in range(start_epoch, max_epoch + 1):
        model.train()
        ep_t0 = time.time()

        active = CURRICULUM_STAGES[current_stage]['losses']

        optimizer.zero_grad()
        w = model.get_loss_weights()
        total_loss, loss_dict = compute_all_losses(model, data, active, w, interface_normalizer)
        train_loss = loss_dict['total']

        # Validation every VAL_EVAL_EVERY epochs (gradient applied only on eval epochs)
        if epoch % VAL_EVAL_EVERY == 0:
            val_loss_tensor = compute_validation_loss(model, *data['validation'])
            cached_val_loss_value = val_loss_tensor.item()
            cached_ratio = cached_val_loss_value / (train_loss + 1e-12)

            if cached_ratio > 10:
                val_weight = 0.35
            elif cached_ratio > 5:
                val_weight = 0.2
            else:
                val_weight = 0.1
            total_loss = total_loss + val_weight * val_loss_tensor

        val_loss_value = cached_val_loss_value
        ratio = cached_ratio
        history['validation'].append(val_loss_value)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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

        for key in ['total', 'ic', 'thermal_bc', 'elastic_bc', 'thermal_pde',
                    'elastic_pde', 'interface', 'L1', 'L2', 'L3', 'L4']:
            history[key].append(loss_dict.get(key, 0.0))
        history['lr'].append(lr)
        history['epoch_time'].append(ep_time)

        # ── Stage 1 exit: loss < 2e-3 only — ratio is excluded because validation
        #    measures PDE residuals that haven't been trained yet in stage 1,
        #    so a high ratio here is expected and not meaningful.
        if current_stage == 1 and train_loss < STAGE1_LOSS_THRESHOLD:
            current_stage = 2
            model.reset_attention_weights()
            print(f"\n  [Stage 1 Done] Epoch {epoch}: loss={train_loss:.3e} "
                  f"-> Stage 2 (BC + PDE)\n")

        # ── Stage 2 exit: 10 consecutive epochs with loss < 1e-3 AND ratio < 7 ─
        elif current_stage == 2:
            if train_loss < STAGE2_LOSS_THRESHOLD and ratio < STAGE2_RATIO_THRESHOLD:
                stage2_converged_count += 1
                if stage2_converged_count >= STAGE2_CONVERGE_EPOCHS:
                    print(f"\n{'='*80}")
                    print(f"  SWITCHING TO L-BFGS  (Stage 2 converged for {STAGE2_CONVERGE_EPOCHS} epochs)")
                    print(f"  Epoch {epoch}: loss={train_loss:.3e}, ratio={ratio:.1f}x")
                    print(f"{'='*80}\n")
                    ckpt_path = os.path.join(CKPT_DIR, f'pinn_before_lbfgs_epoch_{epoch:05d}.pt')
                    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'history': history}, ckpt_path)
                    return epoch, True, optimizer
            else:
                stage2_converged_count = 0

        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'pinn_epoch_{epoch:05d}.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'history': history,
                        'current_stage': current_stage,
                        'stage2_converged_count': stage2_converged_count}, ckpt_path)

        if epoch % 50 == 0 or epoch == 1:
            status = ""
            if ratio > 100:
                status = "  CRITICAL!"
            elif ratio > 10:
                status = "  OVERFIT!"
            elif ratio > 5:
                status = "  warn"
            print(f"{epoch:>7d} | {current_stage:>5d} | {train_loss:>10.3e} | "
                  f"{val_loss_value:>10.3e} | {ratio:>7.1f}x | {lr:>8.2e} | "
                  f"{ep_time:>6.2f}s{status}")

    return max_epoch, False, optimizer


def train_with_lbfgs(model, data, interface_normalizer, history, start_epoch, max_epochs, sep):
    """
    L-BFGS refinement (Stage 3).
    Stops when loss < LBFGS_LOSS_THRESHOLD AND ratio < LBFGS_RATIO_THRESHOLD
    for LBFGS_CONVERGE_EPOCHS consecutive epochs.
    """
    print(f"\n{'='*80}")
    print(f"  L-BFGS REFINEMENT — Stage 3 (start epoch {start_epoch})")
    print(f"  Stop when loss < {LBFGS_LOSS_THRESHOLD:.0e} and ratio < {LBFGS_RATIO_THRESHOLD} "
          f"for {LBFGS_CONVERGE_EPOCHS} consecutive epochs")
    print(f"{'='*80}\n")

    optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, max_eval=25,
                             tolerance_grad=1e-9, tolerance_change=1e-12, history_size=100)

    active = CURRICULUM_STAGES[4]['losses']
    lbfgs_converged_count = 0
    cached_val_loss_value = 0.0
    final_epoch = start_epoch

    for epoch in range(start_epoch, start_epoch + max_epochs):
        model.train()
        ep_t0 = time.time()

        def closure():
            optimizer.zero_grad()
            w = model.get_loss_weights()  # recomputed each call — avoids stale graph
            loss, _ = compute_all_losses(model, data, active, w, interface_normalizer)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Recompute for logging — cannot use no_grad because PDE losses need autograd
        w = model.get_loss_weights()
        _, loss_dict = compute_all_losses(model, data, active, w, interface_normalizer)
        train_loss = loss_dict['total']
        ep_time = time.time() - ep_t0

        # Validation (scalar only — not added to L-BFGS loss)
        if epoch % VAL_EVAL_EVERY == 0:
            cached_val_loss_value = compute_validation_loss(model, *data['validation']).item()
        val_loss_value = cached_val_loss_value
        ratio = val_loss_value / (train_loss + 1e-12)

        history['validation'].append(val_loss_value)
        for key in ['total', 'ic', 'thermal_bc', 'elastic_bc', 'thermal_pde',
                    'elastic_pde', 'interface', 'L1', 'L2', 'L3', 'L4']:
            history[key].append(loss_dict.get(key, 0.0))
        history['lr'].append(1.0)
        history['epoch_time'].append(ep_time)

        if epoch % 10 == 0 or epoch == start_epoch:
            print(f"{epoch:>7d} | {'  3':>5} | {train_loss:>10.3e} | "
                  f"{val_loss_value:>10.3e} | {ratio:>7.1f}x | L-BFGS | {ep_time:>6.2f}s")

        if (epoch - start_epoch) % 100 == 0:
            ckpt_path = os.path.join(CKPT_DIR, f'pinn_lbfgs_epoch_{epoch:05d}.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'history': history}, ckpt_path)

        # ── Stage 3 stop: LBFGS_CONVERGE_EPOCHS consecutive epochs meeting both criteria ─
        if train_loss < LBFGS_LOSS_THRESHOLD and ratio < LBFGS_RATIO_THRESHOLD:
            lbfgs_converged_count += 1
            if lbfgs_converged_count >= LBFGS_CONVERGE_EPOCHS:
                print(f"\n  [L-BFGS Done] Epoch {epoch}: loss={train_loss:.3e}, "
                      f"ratio={ratio:.1f}x  (converged {lbfgs_converged_count} epochs)\n")
                final_epoch = epoch
                break
        else:
            lbfgs_converged_count = 0

        final_epoch = epoch

    return final_epoch


def train_model(model, data, start_epoch=1, start_stage=1, start_stage2_count=0):
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
        interface_normalizer, history, start_epoch, N_EPOCHS_ADAM, sep,
        start_stage=start_stage, start_stage2_count=start_stage2_count
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
