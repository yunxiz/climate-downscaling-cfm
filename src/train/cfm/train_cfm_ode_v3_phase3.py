#!/usr/bin/env python3
"""
Phase 3: PRISM-supervised fine-tuning.

The model has learned single-step residuals (Phase 1) and is rollout-stable
(Phase 2), but has never seen precipitation at scales finer than 12.5km.
Phase 3 closes this gap by comparing the full 4-step recursive output
against PRISM observations at 1.5625km.

Training loop:
  1. For each training day, load ERA5 25km field
  2. Run 4-step recursive downscale: 25→12.5→6.25→3.125→1.5625 km
     - Steps 0-2: no_grad (frozen, just forward pass)
     - Step 3: with grad (backprop through this step only)
  3. Load PRISM for that day, coarsen to 1.5625km grid
  4. Loss = MSE(model_output, PRISM) + flow_loss on Pair A/B

This teaches the model what fine-scale precipitation actually looks like
while keeping it grounded via the single-step flow matching objective.

No data leakage: trained on 2018-2024 PRISM, evaluated on 2025 PRISM.

Usage:
    python -u train_phase3.py \
        --resume checkpoints/cfm_ode_v3/best_model_phase2.pt \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_tif_2018_2024 \
        --output-dir checkpoints/cfm_ode_v3 \
        --epochs 30 --batch-size 8 --lr 1e-4
"""

import argparse
import copy
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
import xarray as xr

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AEF_YEAR_OFFSET = 2017
M_TO_MM = 1000.0

# Recursive downscaling steps with AEF resolutions
DOWNSCALE_STEPS = [
    (25.0,   12.5,    25.0,   12.5),    # Step 0: Pair A equivalent
    (12.5,    6.25,   12.5,    6.25),   # Step 1: Pair B equivalent
    (6.25,    3.125,   6.25,   3.125),  # Step 2: extrapolation
    (3.125,   1.5625,  3.125,  1.5625), # Step 3: extrapolation (grad here)
]


# ── AEF helpers ──────────────────────────────────────────────────────────────

def _format_res(res_km):
    if res_km == int(res_km):
        return str(int(res_km))
    return str(res_km)


def find_aef_file(aef_dir, t_idx, res_km):
    res_str = _format_res(res_km)
    candidates = [
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{res_str}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{res_str}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{res_km}km.nc",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_aef_as_tensor(aef_dir, t_idx, res_km, target_h, target_w, device):
    path = find_aef_file(aef_dir, t_idx, res_km)
    if path is None:
        return torch.zeros(1, 64, target_h, target_w, device=device)
    ds = xr.open_dataset(path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32).transpose(1, 2, 0)
    ds.close()
    D = arr.shape[2]
    resized = np.stack([
        zoom(arr[:, :, d], (target_h / arr.shape[0], target_w / arr.shape[1]), order=1)
        for d in range(D)
    ], axis=0).astype(np.float32)
    return torch.from_numpy(resized).unsqueeze(0).to(device)


# ── PRISM loading ────────────────────────────────────────────────────────────

def load_prism_day(prism_dir, date_str, target_lats, target_lons):
    """Load PRISM TIF, crop to IL, regrid to target. Returns mm/day or None."""
    import rasterio

    date_clean = date_str.replace("-", "")
    year = date_clean[:4]
    tif_path = Path(prism_dir) / year / f"prism_ppt_us_30s_{date_clean}.tif"

    if not tif_path.exists():
        return None

    with rasterio.open(tif_path) as src:
        prism_transform = src.transform
        prism_data = src.read(1).astype(np.float32)
        H_p, W_p = prism_data.shape
        prism_lons = np.array([prism_transform[2] + prism_transform[0] * (j + 0.5)
                               for j in range(W_p)])
        prism_lats = np.array([prism_transform[5] + prism_transform[4] * (i + 0.5)
                               for i in range(H_p)])

    prism_data[prism_data < -900] = 0

    target_lons_shifted = -np.abs(np.where(target_lons > 180, target_lons - 360, target_lons))
    lat_min, lat_max = target_lats.min() - 0.1, target_lats.max() + 0.1
    lon_min, lon_max = target_lons_shifted.min() - 0.1, target_lons_shifted.max() + 0.1

    lat_mask = (prism_lats >= lat_min) & (prism_lats <= lat_max)
    lon_mask = (prism_lons >= lon_min) & (prism_lons <= lon_max)

    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        return None

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    cropped = prism_data[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

    target_h, target_w = len(target_lats), len(target_lons)
    regridded = zoom(cropped, (target_h / cropped.shape[0], target_w / cropped.shape[1]),
                     order=1).astype(np.float32)
    return np.maximum(regridded, 0)


# ── ODE integration (single step, with or without grad) ─────────────────────

def ode_step(model, coarse_up, alpha_c, alpha_f, n_steps=20, device="cuda"):
    """
    Single-sample ODE integration from t=0 to t=1.
    Returns the predicted residual. Supports gradient flow.

    coarse_up: (1, 1, H, W)
    """
    x = torch.randn(1, 1, coarse_up.shape[2], coarse_up.shape[3], device=device)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step * dt
        t_batch = torch.full((1,), t_val, device=device)
        x_input = torch.cat([coarse_up, x], dim=1)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            v = model(t_batch, x_input, alpha_c, alpha_f)

        x = x + v.to(torch.float32) * dt

    return x  # predicted residual


# ── Helpers ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def pad_t_like_x(t, x):
    return t.view(-1, *([1] * (x.ndim - 1)))


# ── Phase 3 training ─────────────────────────────────────────────────────────

def train_phase3_epoch(
    model, era5_25km, dates, train_idx, aef_dir, prism_dir,
    era5_lats, era5_lons, fm, optimizer, device, args,
    loaders=None,
):
    """
    One epoch of Phase 3 training.

    Alternates between two objectives:
      A) PRISM rollout: run 4-step recursive downscale, compare against PRISM
         at 1.5625km. This teaches fine-scale structure.
      B) Flow matching: standard single-step flow loss on Pair A/B data.
         This anchors the model to predict correct residuals and prevents
         collapse to the climatological mean.

    Schedule: for every PRISM day, run K flow matching minibatches.
    This keeps the model grounded while learning from PRISM.
    """
    model.train()
    rng = np.random.RandomState(int(time.time()) % 2**31)

    # Shuffle training indices and subsample for speed
    shuffled_idx = train_idx.copy()
    rng.shuffle(shuffled_idx)
    if args.max_days_per_epoch and len(shuffled_idx) > args.max_days_per_epoch:
        shuffled_idx = shuffled_idx[:args.max_days_per_epoch]
        log.info(f"  Subsampled to {len(shuffled_idx)} days this epoch")

    total_loss = 0
    total_prism_loss = 0
    total_log_loss = 0
    total_flow_loss = 0
    n_prism = 0
    n_flow = 0

    # Create iterators for flow matching batches
    flow_iter_A = iter(loaders["train_A"]) if loaders else None
    flow_iter_B = iter(loaders["train_B"]) if loaders else None

    FLOW_STEPS_PER_PRISM = 3  # run 3 flow matching steps per PRISM day

    for batch_i, t_idx in enumerate(shuffled_idx):

        # ── Part B: Flow matching regularization ─────────────────────────
        # Run a few standard flow matching steps to keep the model grounded
        if flow_iter_A is not None:
            for _ in range(FLOW_STEPS_PER_PRISM):
                try:
                    batch_A = next(flow_iter_A)
                except StopIteration:
                    flow_iter_A = iter(loaders["train_A"])
                    batch_A = next(flow_iter_A)

                try:
                    batch_B = next(flow_iter_B)
                except StopIteration:
                    flow_iter_B = iter(loaders["train_B"])
                    batch_B = next(flow_iter_B)

                # Pick A or B randomly
                batch = batch_A if rng.rand() > 0.5 else batch_B

                x_coarse_up = batch["x_coarse_up"].to(device)
                residual_gt = batch["residual"].to(device)
                alpha_c = batch["alpha_coarse"].to(device)
                alpha_f = batch["alpha_fine"].to(device)

                x0 = torch.randn_like(residual_gt)
                t, xt, ut = fm.sample_location_and_conditional_flow(
                    x0.flatten(1), residual_gt.flatten(1)
                )
                xt = xt.view_as(x0)
                ut = ut.view_as(x0)

                x_input = torch.cat([x_coarse_up, xt], dim=1)
                v_pred = model(t, x_input, alpha_c, alpha_f)

                L_flow = ((v_pred - ut) ** 2).mean()

                optimizer.zero_grad()
                L_flow.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_flow_loss += L_flow.item()
                n_flow += 1

                del x_coarse_up, residual_gt, alpha_c, alpha_f
                del x0, xt, ut, x_input, v_pred, L_flow

        # ── Part A: PRISM rollout ────────────────────────────────────────
        date_str = str(dates[t_idx])[:10]
        year = int(date_str[:4])
        t_aef = year - AEF_YEAR_OFFSET

        era5_field = era5_25km[t_idx]  # meters/day, (H25, W25)
        field_mm = era5_field * M_TO_MM  # mm/day

        H25, W25 = era5_field.shape
        H_final, W_final = H25 * 16, W25 * 16
        final_lats = np.linspace(era5_lats[0], era5_lats[-1], H_final)
        final_lons = np.linspace(era5_lons[0], era5_lons[-1], W_final)

        prism = None
        try:
            prism = load_prism_day(prism_dir, date_str, final_lats, final_lons)
        except Exception as e:
            log.warning(f"    PRISM failed for {date_str}: {e}")

        if prism is None:
            continue

        if prism.shape != (H_final, W_final):
            prism = zoom(prism, (H_final / prism.shape[0], W_final / prism.shape[1]),
                         order=1).astype(np.float32)
            prism = np.maximum(prism, 0)

        prism_t = torch.from_numpy(prism).unsqueeze(0).unsqueeze(0).to(device)

        # Recursive rollout: steps 0-2 frozen, step 3 with grad
        current = torch.from_numpy(field_mm.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            for step_idx in range(3):
                in_res, out_res, aef_c_res, aef_f_res = DOWNSCALE_STEPS[step_idx]
                H_in, W_in = current.shape[2], current.shape[3]
                H_out, W_out = H_in * 2, W_in * 2

                coarse_up = F.interpolate(current, size=(H_out, W_out),
                                          mode="bicubic", align_corners=False)
                alpha_c = load_aef_as_tensor(Path(aef_dir), t_aef, aef_c_res,
                                             H_out, W_out, device)
                alpha_f = load_aef_as_tensor(Path(aef_dir), t_aef, aef_f_res,
                                             H_out, W_out, device)
                residual = ode_step(model, coarse_up, alpha_c, alpha_f,
                                    n_steps=10, device=device)
                current = torch.clamp(coarse_up + residual, min=0)

                del coarse_up, alpha_c, alpha_f, residual
                torch.cuda.empty_cache()

        # Step 3: WITH gradient
        step_idx = 3
        in_res, out_res, aef_c_res, aef_f_res = DOWNSCALE_STEPS[step_idx]
        H_in, W_in = current.shape[2], current.shape[3]
        H_out, W_out = H_in * 2, W_in * 2

        coarse_up = F.interpolate(current.detach(), size=(H_out, W_out),
                                  mode="bicubic", align_corners=False)
        alpha_c = load_aef_as_tensor(Path(aef_dir), t_aef, aef_c_res,
                                     H_out, W_out, device)
        alpha_f = load_aef_as_tensor(Path(aef_dir), t_aef, aef_f_res,
                                     H_out, W_out, device)

        residual = ode_step(model, coarse_up, alpha_c, alpha_f,
                            n_steps=20, device=device)
        x_final = torch.clamp(coarse_up + residual, min=0)

        # PRISM loss
        if x_final.shape != prism_t.shape:
            prism_t = F.interpolate(prism_t, size=x_final.shape[2:],
                                    mode="bilinear", align_corners=False)

        L_prism = ((x_final - prism_t) ** 2).mean()
        L_log = ((torch.log1p(x_final) - torch.log1p(prism_t)) ** 2).mean()

        # Weight PRISM loss lower than flow matching to prevent collapse
        loss = 0.1 * L_prism + 0.05 * L_log

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_prism_loss += L_prism.item()
        total_log_loss += L_log.item()
        total_loss += loss.item()
        n_prism += 1

        del current, coarse_up, alpha_c, alpha_f, residual, x_final, prism_t, loss
        del L_prism, L_log
        torch.cuda.empty_cache()

        if (batch_i + 1) % 50 == 0:
            avg_flow = total_flow_loss / max(n_flow, 1)
            avg_prism = total_prism_loss / max(n_prism, 1)
            log.info(f"    Day {batch_i+1}/{len(shuffled_idx)}: "
                     f"flow={avg_flow:.4f} PRISM={avg_prism:.4f}")

    metrics = {
        "prism_loss": total_prism_loss / max(n_prism, 1),
        "log_loss": total_log_loss / max(n_prism, 1),
        "flow_loss": total_flow_loss / max(n_flow, 1),
        "total_loss": total_loss / max(n_prism, 1),
        "n_prism_days": n_prism,
        "n_flow_steps": n_flow,
    }
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: PRISM-supervised fine-tuning"
    )
    parser.add_argument("--resume", required=True,
                        help="Path to Phase 2 best checkpoint")
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--prism-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (should be small for fine-tuning)")
    parser.add_argument("--max-days-per-epoch", type=int, default=500,
                        help="Randomly sample this many days per epoch (for speed)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from src.models.model_ode import DownscalingUNet

    # Load model from Phase 2
    log.info(f"Loading Phase 2 checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device, weights_only=False)

    model = DownscalingUNet(
        in_channels=2, out_channels=1,
        base_channels=48, channel_mult=(1, 2, 4),
        aef_dim=64, time_dim=64, num_heads=2,
    ).to(device)

    # Prefer EMA weights as starting point
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        log.info("  Loaded EMA weights from Phase 2")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("  Loaded model weights from Phase 2")

    ema_model = copy.deepcopy(model)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {n_params:,}")

    # Freeze AEF cross-attention (only train flow backbone)
    for name, param in model.named_parameters():
        if "cross" in name.lower() or "kv_proj" in name.lower():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Trainable parameters: {trainable:,}")

    # Load ERA5 data
    era5_dir = Path(args.era5_dir)
    era5_lats = np.load(era5_dir / "era5_lats.npy")
    era5_lons = np.load(era5_dir / "era5_lons.npy")
    A_fine = np.load(era5_dir / "pair_A_fine.npy")  # 25km, meters/day
    A_dates = np.load(era5_dir / "valid_times_A.npy")
    A_train_idx = np.load(era5_dir / "train_indices_A.npy")
    A_test_idx = np.load(era5_dir / "test_indices_A.npy")

    log.info(f"Training days: {len(A_train_idx)}, Test days: {len(A_test_idx)}")

    # Build paired dataloaders for flow matching regularization
    from dataset import build_datasets, build_paired_dataloaders
    log.info("Building flow matching dataloaders...")
    datasets = build_datasets(args.era5_dir, args.aef_dir)
    loaders = build_paired_dataloaders(
        datasets, batch_size=16, num_workers=4,
    )
    log.info(f"  train_A: {len(datasets['train_A'])} samples")
    log.info(f"  train_B: {len(datasets['train_B'])} samples")

    # Flow matcher for regularization
    fm = ConditionalFlowMatcher(sigma=0.0)

    # Optimizer — small LR for fine-tuning
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )

    # Load existing history
    history_path = out_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    else:
        history = []

    best_prism_loss = float("inf")

    log.info("=" * 60)
    log.info("PHASE 3: PRISM-Supervised Fine-Tuning")
    log.info(f"  LR: {args.lr}, Epochs: {args.epochs}")
    log.info(f"  PRISM dir: {args.prism_dir}")
    log.info(f"  Rollout: 25km → 1.5625km (4 steps, grad on step 3 only)")
    log.info("=" * 60)

    for epoch in range(args.epochs):
        t_epoch = time.time()

        metrics = train_phase3_epoch(
            model, A_fine, A_dates, A_train_idx,
            args.aef_dir, args.prism_dir,
            era5_lats, era5_lons, fm, optimizer, device, args,
            loaders=loaders,
        )

        update_ema(ema_model, model)

        elapsed = time.time() - t_epoch

        epoch_record = {
            "epoch": epoch + 1,
            "phase": 3,
            "elapsed_s": round(elapsed, 1),
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in metrics.items()},
        }
        history.append(epoch_record)

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        prism_loss = metrics["prism_loss"]
        log.info(
            f"Phase3 Epoch {epoch+1}/{args.epochs} ({elapsed:.0f}s) — "
            f"flow={metrics['flow_loss']:.4f} "
            f"PRISM_mse={prism_loss:.4f} "
            f"log_mse={metrics['log_loss']:.4f} "
            f"n_prism={metrics['n_prism_days']} "
            f"n_flow={metrics['n_flow_steps']}"
        )

        # Save best
        if prism_loss < best_prism_loss:
            best_prism_loss = prism_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "prism_loss": prism_loss,
            }, out_dir / "best_model_phase3.pt")
            log.info(f"  Saved best Phase 3 model (PRISM loss={prism_loss:.6f})")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
            }, out_dir / f"checkpoint_phase3_ep{epoch+1}.pt")

    # Final save
    torch.save({
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "args": vars(args),
    }, out_dir / "final_phase3.pt")

    log.info(f"Phase 3 complete. Best PRISM loss: {best_prism_loss:.6f}")
    log.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()