#!/usr/bin/env python3
"""
Training script for OT-CFM precipitation downscaling.

Simplified from SF2M:
  - No score head or score loss (removed the unstable ~5800 loss term)
  - Uses ConditionalFlowMatcher (I-CFM) to preserve paired conditioning
  - OT-CFM was breaking the (x0, x1) ↔ conditioning alignment
  - Ensemble generation at inference via different x0 initializations
  - 3M parameters (was 33M)

Phase 1: Multi-scale generalization
  - Losses: flow matching + cycle-consistency + spectral PSD
  - AEF projections frozen for first 10 epochs

Phase 2: Rollout robustness
  - Additional: rollout consistency loss
  - AEF projections frozen

Usage:
    python -u train_cfm_ode_v3.py \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --output-dir checkpoints/cfm \
        --phase 1
"""

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Auxiliary losses
# ══════════════════════════════════════════════════════════════════════════════

def cycle_consistency_loss(x_fine_hat, x_coarse):
    """
    Physical constraint: pooling the fine prediction back to the original
    coarse resolution should recover the original coarse field.

    x_fine_hat: (B, 1, H_fine, W_fine) — predicted fine-resolution field
    x_coarse:   (B, 1, H_coarse, W_coarse) — original coarse field (NOT upsampled)

    The pool target size is the coarse grid dimensions.
    """
    H_c, W_c = x_coarse.shape[2], x_coarse.shape[3]
    fine_pooled = F.adaptive_avg_pool2d(x_fine_hat, (H_c, W_c))
    return ((fine_pooled - x_coarse) ** 2).mean()


def spectral_psd_loss(r_hat, r_target):
    fft_hat = torch.fft.rfft2(r_hat.squeeze(1))
    fft_tgt = torch.fft.rfft2(r_target.squeeze(1))
    psd_hat = (fft_hat.abs() ** 2).mean(dim=0)
    psd_tgt = (fft_tgt.abs() ** 2).mean(dim=0)
    log_psd_hat = torch.log(psd_hat + 1e-8)
    log_psd_tgt = torch.log(psd_tgt + 1e-8)
    H_freq = log_psd_hat.shape[0]
    subgrid_start = H_freq // 4
    return ((log_psd_hat[subgrid_start:] - log_psd_tgt[subgrid_start:]) ** 2).mean()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def pad_t_like_x(t, x):
    return t.view(-1, *([1] * (x.ndim - 1)))


def estimate_x1_from_flow(xt, vt, t):
    return xt + (1 - pad_t_like_x(t, xt)) * vt


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


# ══════════════════════════════════════════════════════════════════════════════
# Training steps
# ══════════════════════════════════════════════════════════════════════════════

def train_step_phase1(model, batch, fm, device, epoch):
    x_coarse = batch["x_coarse"].to(device)
    x_coarse_up = batch["x_coarse_up"].to(device)
    residual = batch["residual"].to(device)
    alpha_c = batch["alpha_coarse"].to(device)
    alpha_f = batch["alpha_fine"].to(device)

    x0 = torch.randn_like(residual)
    x1 = residual

    t, xt, ut = fm.sample_location_and_conditional_flow(
        x0.flatten(1), x1.flatten(1)
    )
    xt = xt.view_as(x0)
    ut = ut.view_as(x0)

    x_input = torch.cat([x_coarse_up, xt], dim=1)
    v_pred = model(t, x_input, alpha_c, alpha_f)

    # Flow matching loss
    L_flow = ((v_pred - ut) ** 2).mean()

    # Cycle-consistency: pool predicted fine back to coarse, compare to original coarse
    r_hat = estimate_x1_from_flow(xt, v_pred, t)
    x_fine_hat = x_coarse_up + r_hat
    L_cycle = cycle_consistency_loss(x_fine_hat, x_coarse)

    # Spectral PSD (from epoch 5)
    if epoch >= 5:
        L_psd = spectral_psd_loss(r_hat, residual)
    else:
        L_psd = torch.tensor(0.0, device=device)

    loss = L_flow + 0.8 * L_cycle
    if epoch >= 5:
        loss = loss + 0.4 * L_psd

    metrics = {
        "flow": L_flow.item(),
        "cycle": L_cycle.item(),
        "psd": L_psd.item(),
        "total": loss.item(),
    }
    return loss, metrics


def train_step_phase2(model, batch_A, batch_B, fm, device):
    """
    Phase 2: 2-step rollout using Pair A (50→25km) → Pair B (25→12.5km).

    Step 1: Model predicts residual at 25km (Pair A), reconstructs x_25km_hat.
    Step 2: Uses x_25km_hat as input, bicubic upsamples to 12.5km grid,
            model predicts residual at 12.5km.
    Rollout loss: pool x_12.5km_hat back to 50km, compare to original x_50km.

    Gradients flow through both steps for full rollout training.
    """
    # Pair A data (50→25km)
    x_coarse_A = batch_A["x_coarse"].to(device)       # (B, 1, H_50, W_50) original 50km
    x_coarse_up_A = batch_A["x_coarse_up"].to(device)  # (B, 1, H_25, W_25) upsampled
    residual_A = batch_A["residual"].to(device)         # (B, 1, H_25, W_25)
    alpha_c_A = batch_A["alpha_coarse"].to(device)
    alpha_f_A = batch_A["alpha_fine"].to(device)

    # Pair B data (25→12.5km) — for AEF conditioning at step 2
    alpha_c_B = batch_B["alpha_coarse"].to(device)
    alpha_f_B = batch_B["alpha_fine"].to(device)
    x_coarse_B = batch_B["x_coarse"].to(device)       # (B, 1, H_25, W_25) = Pair A fine
    residual_B = batch_B["residual"].to(device)

    B = x_coarse_up_A.shape[0]

    # ── Step 1: Flow matching on Pair A ──────────────────────────────────
    x0_A = torch.randn_like(residual_A)
    t_A, xt_A, ut_A = fm.sample_location_and_conditional_flow(
        x0_A.flatten(1), residual_A.flatten(1)
    )
    xt_A = xt_A.view_as(x0_A)
    ut_A = ut_A.view_as(x0_A)

    x_input_A = torch.cat([x_coarse_up_A, xt_A], dim=1)
    v_pred_A = model(t_A, x_input_A, alpha_c_A, alpha_f_A)

    L_flow_A = ((v_pred_A - ut_A) ** 2).mean()

    # Reconstruct step 1 output
    r_hat_A = estimate_x1_from_flow(xt_A, v_pred_A, t_A)
    x_25km_hat = x_coarse_up_A + r_hat_A

    # Cycle loss step 1: pool x_25km_hat back to 50km, compare to x_50km
    L_cycle_A = cycle_consistency_loss(x_25km_hat, x_coarse_A)

    # PSD step 1
    L_psd_A = spectral_psd_loss(r_hat_A, residual_A)

    # ── Step 2: Rollout — use step 1 output as input to step 2 ──────────
    # Bicubic upsample step 1 output to 12.5km grid
    H_B, W_B = alpha_c_B.shape[2], alpha_c_B.shape[3]
    x_25km_up = F.interpolate(x_25km_hat, size=(H_B, W_B),
                               mode="bicubic", align_corners=False)

    # Flow matching on step 2
    x0_B = torch.randn(B, 1, H_B, W_B, device=device)
    t_B, xt_B, ut_B = fm.sample_location_and_conditional_flow(
        x0_B.flatten(1), residual_B.flatten(1)
    )
    xt_B = xt_B.view_as(x0_B)
    ut_B = ut_B.view_as(x0_B)

    x_input_B = torch.cat([x_25km_up, xt_B], dim=1)
    v_pred_B = model(t_B, x_input_B, alpha_c_B, alpha_f_B)

    L_flow_B = ((v_pred_B - ut_B) ** 2).mean()

    # Reconstruct step 2 output
    r_hat_B = estimate_x1_from_flow(xt_B, v_pred_B, t_B)
    x_12km_hat = x_25km_up + r_hat_B

    # Cycle loss step 2: pool x_12.5km_hat back to 25km, compare to x_25km (Pair B coarse)
    L_cycle_B = cycle_consistency_loss(x_12km_hat, x_coarse_B)

    # PSD step 2
    L_psd_B = spectral_psd_loss(r_hat_B, residual_B)

    # ── Rollout consistency: pool x_12.5km_hat all the way back to 50km ──
    H_50, W_50 = x_coarse_A.shape[2], x_coarse_A.shape[3]
    x_repooled = F.adaptive_avg_pool2d(x_12km_hat, (H_50, W_50))
    L_rollout = ((x_repooled - x_coarse_A) ** 2).mean()

    # ── Combined loss ────────────────────────────────────────────────────
    L_flow = (L_flow_A + L_flow_B) / 2
    L_cycle = (L_cycle_A + L_cycle_B) / 2
    L_psd = (L_psd_A + L_psd_B) / 2

    loss = L_flow + 0.8 * L_cycle + 0.4 * L_psd + 0.1 * L_rollout

    metrics = {
        "flow": L_flow.item(),
        "cycle": L_cycle.item(),
        "psd": L_psd.item(),
        "rollout": L_rollout.item(),
        "total": loss.item(),
    }
    return loss, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Validation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate(model, loader, fm, device):
    model.eval()
    total_loss = 0
    n = 0
    for batch in loader:
        x_coarse_up = batch["x_coarse_up"].to(device)
        residual = batch["residual"].to(device)
        alpha_c = batch["alpha_coarse"].to(device)
        alpha_f = batch["alpha_fine"].to(device)

        x0 = torch.randn_like(residual)
        t, xt, ut = fm.sample_location_and_conditional_flow(
            x0.flatten(1), residual.flatten(1)
        )
        xt = xt.view_as(x0)
        ut = ut.view_as(x0)

        x_input = torch.cat([x_coarse_up, xt], dim=1)
        v_pred = model(t, x_input, alpha_c, alpha_f)
        total_loss += ((v_pred - ut) ** 2).mean().item()
        n += 1

    model.train()
    return total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Training loops
# ══════════════════════════════════════════════════════════════════════════════

def train_phase1(model, loaders, fm, optimizer, scheduler, device, args):
    from dataset import InterleavedPairIterator

    ema_model = copy.deepcopy(model)
    best_val_loss = float("inf")
    out_dir = Path(args.output_dir)
    history = []

    aef_param_names = set()
    for name, param in model.named_parameters():
        if "cross" in name.lower() or "kv_proj" in name.lower():
            aef_param_names.add(name)

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        t_epoch = time.time()

        freeze_aef = epoch < 10
        for name, param in model.named_parameters():
            if name in aef_param_names:
                param.requires_grad = not freeze_aef

        train_iter = InterleavedPairIterator(loaders["train_A"], loaders["train_B"])
        epoch_metrics = {}
        n_batches = 0

        for batch in train_iter:
            loss, metrics = train_step_phase1(model, batch, fm, device, epoch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            update_ema(ema_model, model)

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        val_A = validate(model, loaders["test_A"], fm, device)
        val_B = validate(model, loaders["test_B"], fm, device)
        val_loss = (val_A + val_B) / 2

        elapsed = time.time() - t_epoch

        epoch_record = {
            "epoch": epoch + 1,
            "phase": 1,
            "elapsed_s": round(elapsed, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "aef_frozen": freeze_aef,
            "val_A": val_A,
            "val_B": val_B,
            "val_avg": val_loss,
            **{f"train_{k}": v for k, v in epoch_metrics.items()},
        }
        history.append(epoch_record)

        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        log.info(
            f"Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) — "
            f"flow={epoch_metrics['flow']:.6f} "
            f"cycle={epoch_metrics['cycle']:.6f} psd={epoch_metrics['psd']:.6f} "
            f"total={epoch_metrics['total']:.6f} "
            f"val_A={val_A:.6f} val_B={val_B:.6f} "
            f"{'[AEF frozen]' if freeze_aef else ''}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, out_dir / "best_model_phase1.pt")
            log.info(f"  Saved best model (val={val_loss:.6f})")

        if (epoch + 1) % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, out_dir / f"checkpoint_phase1_ep{epoch+1}.pt")

    return model, ema_model


def train_phase2(model, ema_model, loaders, fm, optimizer, device, args):
    """Phase 2: 2-step rollout with paired Pair A + Pair B batches."""
    out_dir = Path(args.output_dir)

    history_path = out_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []

    for name, param in model.named_parameters():
        if "cross" in name.lower() or "kv_proj" in name.lower():
            param.requires_grad = False

    best_val_loss = float("inf")

    for epoch in range(args.phase2_epochs):
        model.train()
        t_epoch = time.time()

        # Iterate over paired A+B batches (same day, different scales)
        epoch_metrics = {}
        n_batches = 0

        for batch_A, batch_B in zip(loaders["train_A"], loaders["train_B"]):
            loss, metrics = train_step_phase2(model, batch_A, batch_B, fm, device)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(ema_model, model)

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1

        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        val_A = validate(model, loaders["test_A"], fm, device)
        val_B = validate(model, loaders["test_B"], fm, device)
        val_loss = (val_A + val_B) / 2

        elapsed = time.time() - t_epoch

        epoch_record = {
            "epoch": epoch + 1,
            "phase": 2,
            "elapsed_s": round(elapsed, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "val_A": val_A,
            "val_B": val_B,
            "val_avg": val_loss,
            **{f"train_{k}": v for k, v in epoch_metrics.items()},
        }
        history.append(epoch_record)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        log.info(
            f"Phase2 Epoch {epoch+1}/{args.phase2_epochs} ({elapsed:.0f}s) — "
            f"flow={epoch_metrics['flow']:.6f} "
            f"cycle={epoch_metrics['cycle']:.6f} "
            f"rollout={epoch_metrics.get('rollout', 0):.6f} "
            f"total={epoch_metrics['total']:.6f} "
            f"val={val_loss:.6f}"
        )

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
            }, out_dir / f"checkpoint_phase2_ep{epoch+1}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
            }, out_dir / "best_model_phase2.pt")
            log.info(f"  Saved best phase2 model (val={val_loss:.6f})")

    return model, ema_model


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train OT-CFM precipitation downscaling model."
    )
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--phase2-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="OT-CFM sigma (0.0 for deterministic OT paths)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.dataset_ode_v3 import build_datasets, build_paired_dataloaders
    from src.models.model_ode import DownscalingUNet

    # Build datasets
    log.info("Building datasets...")
    datasets = build_datasets(args.era5_dir, args.aef_dir)
    loaders = build_paired_dataloaders(
        datasets, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    sample_A = next(iter(loaders["train_A"]))
    sample_B = next(iter(loaders["train_B"]))
    log.info("Pair A shapes:")
    for k, v in sample_A.items():
        log.info(f"  {k}: {tuple(v.shape)}")
    log.info("Pair B shapes:")
    for k, v in sample_B.items():
        log.info(f"  {k}: {tuple(v.shape)}")

    # Build model
    model = DownscalingUNet(
        in_channels=2,
        out_channels=1,
        base_channels=48,
        channel_mult=(1, 2, 4),
        aef_dim=64,
        time_dim=64,
        num_heads=2,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # Flow matcher — deterministic OT-CFM
    # I-CFM: preserves (x0, x1) pairing so conditioning stays aligned.
    # OT-CFM re-pairs within the batch, which breaks the correspondence
    # between x_coarse_up and the interpolated residual.
    fm = ConditionalFlowMatcher(sigma=args.sigma)
    log.info(f"Flow matcher: I-CFM, σ={args.sigma}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.start_epoch = 0

    if args.phase == 1:
        total_steps = args.epochs * (len(loaders["train_A"]) + len(loaders["train_B"]))
        warmup_steps = min(500, total_steps // 10)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # Resume
    if args.resume:
        log.info(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and args.phase == 1:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "epoch" in ckpt and args.phase == 1:
            args.start_epoch = ckpt["epoch"] + 1
            log.info(f"  Resuming from epoch {args.start_epoch}")
        else:
            log.info(f"  Loaded weights (epoch {ckpt.get('epoch', '?')})")

    # Train
    if args.phase == 1:
        log.info("=" * 60)
        log.info("PHASE 1: Multi-Scale Generalization (OT-CFM)")
        log.info("=" * 60)
        model, ema_model = train_phase1(
            model, loaders, fm, optimizer, scheduler, device, args,
        )
    else:
        log.info("=" * 60)
        log.info("PHASE 2: Rollout Robustness")
        log.info("=" * 60)
        ema_model = copy.deepcopy(model)
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * 0.1
        model, ema_model = train_phase2(
            model, ema_model, loaders, fm, optimizer, device, args,
        )

    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "args": vars(args),
    }, out_dir / f"final_phase{args.phase}.pt")

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    log.info(f"Training complete. Saved to {out_dir}")


if __name__ == "__main__":
    main()