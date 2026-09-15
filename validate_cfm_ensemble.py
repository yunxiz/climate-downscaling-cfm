#!/usr/bin/env python3
"""
Recursive ensemble downscaling with CFM (Conditional Flow Matching),
validated against PRISM 800m observations.

Mirrors the XGBoost baseline evaluation for direct comparison.

Pipeline:
  1. Start from ERA5 25km (native) for each test day
  2. Recursively downscale: 25→12.5→6.25→3.125→1.5625 km
  3. At each step, generate N ensemble members via Schrödinger Bridge
     SDE integration: dx = [v_θ + σ²·s_θ] dt + σ dW
  4. Load PRISM 800m TIF, downsample to 1.5625km grid
  5. Evaluate ensemble vs PRISM: CRPS, spread-skill, coverage, RMSE
  6. Plot intermediate resolutions to show progressive refinement

Usage:
    python -u validate_cfm_ensemble_on_prism.py \
        --checkpoint checkpoints/cfm/best_model_phase2.pt \
        --era5-dir data/processed \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_tif_2018_2024 \
        --output-dir results/cfm_ensemble \
        --n-ensemble 100 \
        --n-days 30 \
        --n-steps 50
"""

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import torch
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AEF_YEAR_OFFSET = 2017

# Recursive downscaling steps: (input_res_km, output_res_km, aef_coarse_km, aef_fine_km)
DOWNSCALE_STEPS = [
    (25.0,   12.5,    25.0,   12.5),
    (12.5,    6.25,   12.5,    6.25),
    (6.25,    3.125,   6.25,   3.125),
    (3.125,   1.5625,  3.125,  1.5625),
]


# ── AEF helpers ──────────────────────────────────────────────────────────────

def load_aef_nc(nc_path):
    """Load pre-pooled AEF → (H, W, D) float32."""
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32).transpose(1, 2, 0)
    ds.close()
    return arr


def find_aef_file(aef_dir, t_idx, res_km):
    """Find AEF file matching naming conventions."""
    candidates = [
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{res_km}km.nc",
        aef_dir / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"aef_illinois_{res_km}km.nc",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"AEF not found: t{t_idx}, {res_km}km. "
                            f"Searched: {[str(c) for c in candidates]}")


def load_aef_as_tensor(aef_dir, t_idx, res_km, target_h, target_w, device):
    """
    Load AEF, resize to target grid, return as (1, D, H, W) tensor.
    Returns zeros if file not found.
    """
    try:
        aef = load_aef_nc(find_aef_file(aef_dir, t_idx, res_km))
        D = aef.shape[2]
        # Resize each channel to target grid
        aef_resized = np.stack([
            zoom(aef[:, :, d], (target_h / aef.shape[0], target_w / aef.shape[1]),
                 order=1)
            for d in range(D)
        ], axis=0).astype(np.float32)  # (D, H, W)
        return torch.from_numpy(aef_resized).unsqueeze(0).to(device)  # (1, D, H, W)
    except FileNotFoundError as e:
        log.warning(f"  {e}, using zeros")
        return torch.zeros(1, 64, target_h, target_w, device=device)


# ── SDE integration ──────────────────────────────────────────────────────────

@torch.no_grad()
def sde_integrate(model, coarse_up, alpha_c, alpha_f, sigma,
                  n_ensemble, n_steps=50, device="cuda"):
    """
    Euler-Maruyama integration of the Schrödinger Bridge SDE.

    dx = [v_θ(t,x) + σ² · s_θ(t,x)] dt + σ dW

    Integrates from t=0 (noise) to t=1 (predicted residual).
    Both the velocity head v_θ and score head s_θ are used:
      - v_θ drives transport from noise toward the target distribution
      - s_θ corrects the drift to account for the stochastic diffusion
      - σ dW injects noise at each step, adding ensemble diversity
        beyond just the initial x_0

    Args:
        model: DownscalingUNet (returns v_θ and s_θ)
        coarse_up: (1, 1, H, W) bicubic-upsampled coarse field
        alpha_c: (1, D, H, W) coarse AEF
        alpha_f: (1, D, H, W) fine AEF
        sigma: float, Schrödinger bridge noise scale
        n_ensemble: number of ensemble members
        n_steps: number of Euler-Maruyama steps
        device: torch device

    Returns:
        residuals: (n_ensemble, 1, H, W) predicted residuals
    """
    B = n_ensemble
    _, _, H, W = coarse_up.shape

    # Expand conditioning to batch of ensemble members
    coarse_up_batch = coarse_up.expand(B, -1, -1, -1)        # (B, 1, H, W)
    alpha_c_batch = alpha_c.expand(B, -1, -1, -1)            # (B, D, H, W)
    alpha_f_batch = alpha_f.expand(B, -1, -1, -1)            # (B, D, H, W)

    # Initialize: each member starts from different noise
    x = torch.randn(B, 1, H, W, device=device)

    dt = 1.0 / n_steps
    sqrt_dt = dt ** 0.5

    for step in range(n_steps):
        t_val = step * dt
        t_batch = torch.full((B,), t_val, device=device)

        # Model input: [coarse_up, x_t]
        x_input = torch.cat([coarse_up_batch, x], dim=1)  # (B, 2, H, W)
        v, s = model(t_batch, x_input, alpha_c_batch, alpha_f_batch)

        # SDE drift: v_θ + σ² · s_θ
        drift = v + sigma ** 2 * s

        # Euler-Maruyama: x_{t+dt} = x_t + drift · dt + σ · √dt · z
        noise = torch.randn_like(x)
        x = x + drift * dt + sigma * sqrt_dt * noise

    return x  # (B, 1, H, W) — predicted residuals at t=1


# ── PRISM loading (from baseline script) ─────────────────────────────────────

def load_prism_day(prism_dir, date, target_lats, target_lons):
    """
    Load a PRISM daily TIF, crop to Illinois, and downsample to target grid.
    PRISM is in mm/day → convert to meters/day.
    """
    import rasterio

    date_str = str(date)[:10].replace("-", "")
    year = date_str[:4]
    tif_path = Path(prism_dir) / year / f"prism_ppt_us_30s_{date_str}.tif"

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

    lat_min = target_lats.min() - 0.1
    lat_max = target_lats.max() + 0.1
    lon_min = target_lons.min() - 0.1
    lon_max = target_lons.max() + 0.1

    lat_mask = (prism_lats >= lat_min) & (prism_lats <= lat_max)
    lon_mask = (prism_lons >= lon_min) & (prism_lons <= lon_max)

    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        return None

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    cropped = prism_data[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
    cropped = cropped / 1000.0  # mm → m

    target_h = len(target_lats)
    target_w = len(target_lons)
    regridded = zoom(cropped, (target_h / cropped.shape[0], target_w / cropped.shape[1]),
                     order=1).astype(np.float32)
    return np.maximum(regridded, 0)


# ── Evaluation metrics (from baseline script) ────────────────────────────────

def crps_ensemble(ensemble, observation):
    """CRPS = E[|X - y|] - 0.5 * E[|X - X'|]"""
    N = len(ensemble)
    mae = np.abs(ensemble - observation).mean()
    sorted_ens = np.sort(ensemble)
    diff_sum = sum((2 * i - N) * sorted_ens[i] for i in range(N))
    spread = 2 * diff_sum / (N * N)
    return mae - 0.5 * spread


def crps_grid(ensemble_grids, observation_grid):
    """Mean CRPS over all grid cells."""
    H, W = observation_grid.shape
    crps_values = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            crps_values[i, j] = crps_ensemble(
                ensemble_grids[:, i, j], observation_grid[i, j])
    return crps_values.mean(), crps_values


def ensemble_coverage(ensemble_grids, observation_grid, levels=(0.5, 0.8, 0.9)):
    """Fraction of observations within ensemble quantile intervals."""
    results = {}
    for level in levels:
        alpha = (1 - level) / 2
        lo = np.quantile(ensemble_grids, alpha, axis=0)
        hi = np.quantile(ensemble_grids, 1 - alpha, axis=0)
        covered = ((observation_grid >= lo) & (observation_grid <= hi)).mean()
        results[f"coverage_{int(level*100)}"] = float(covered)
    return results


# ── Recursive downscaling ────────────────────────────────────────────────────

def recursive_downscale_cfm(
    era5_25km,       # (H25, W25) single day ERA5 at 25km
    year,
    model,           # DownscalingUNet (on device)
    sigma,           # Schrödinger bridge σ
    aef_dir,         # Path to AEF directory
    era5_lats,       # (H25,) 25km lats
    era5_lons,       # (W25,) 25km lons
    n_ensemble=100,
    n_steps=50,
    device="cuda",
    ensemble_batch=None,  # process ensemble in chunks to save GPU memory
):
    """
    Recursively downscale from 25km to 1.5625km with CFM ensemble generation.

    At each 2× step:
      1. Bicubic upsample current field
      2. Load AEF at coarse & fine resolution
      3. Integrate Schrödinger Bridge SDE (Euler-Maruyama) from t=0→1
         to predict residual with input-conditional uncertainty
      4. Fine = coarse_up + residual

    Ensemble diversity comes from two sources:
      - Different initial noise x_0 ~ N(0, I) per member
      - Stochastic diffusion σ dW injected at each integration step

    Returns:
        ensemble_final: (n_ensemble, H_final, W_final) at 1.5625km
        intermediates: list of dicts with ensemble stats at each step
    """
    t_idx = year - AEF_YEAR_OFFSET

    # Initialize: all members start from same ERA5 field
    # Shape: (n_ensemble, H, W)
    current_fields = np.stack([era5_25km] * n_ensemble)
    current_lats = era5_lats.copy()
    current_lons = era5_lons.copy()

    intermediates = []

    # Record the input (25km) as step -1
    intermediates.append({
        "step": -1,
        "res_km": 25.0,
        "mean": era5_25km.copy(),
        "std": np.zeros_like(era5_25km),
        "lats": current_lats.copy(),
        "lons": current_lons.copy(),
        "shape": era5_25km.shape,
    })

    if ensemble_batch is None:
        ensemble_batch = n_ensemble  # process all at once

    for step_idx, (in_res, out_res, aef_c_res, aef_f_res) in enumerate(DOWNSCALE_STEPS):
        t_step = time.time()
        H_in, W_in = current_fields.shape[1], current_fields.shape[2]
        H_out = H_in * 2
        W_out = W_in * 2

        out_lats = np.linspace(current_lats[0], current_lats[-1], H_out)
        out_lons = np.linspace(current_lons[0], current_lons[-1], W_out)

        # Load AEF at both scales
        alpha_c = load_aef_as_tensor(Path(aef_dir), t_idx, aef_c_res,
                                     H_out, W_out, device)
        alpha_f = load_aef_as_tensor(Path(aef_dir), t_idx, aef_f_res,
                                     H_out, W_out, device)

        new_fields = np.empty((n_ensemble, H_out, W_out), dtype=np.float32)

        # Process ensemble in batches to manage GPU memory
        for batch_start in range(0, n_ensemble, ensemble_batch):
            batch_end = min(batch_start + ensemble_batch, n_ensemble)
            batch_size = batch_end - batch_start

            # Bicubic upsample each member in this batch
            coarse_up_list = []
            for i in range(batch_start, batch_end):
                cu = zoom(current_fields[i],
                          (H_out / H_in, W_out / W_in),
                          order=3).astype(np.float32)
                coarse_up_list.append(cu)
            coarse_up_np = np.stack(coarse_up_list)  # (batch_size, H_out, W_out)

            # To tensor: (batch_size, 1, H_out, W_out)
            coarse_up_t = torch.from_numpy(coarse_up_np).unsqueeze(1).to(device)

            # Expand AEF to batch
            alpha_c_batch = alpha_c.expand(batch_size, -1, -1, -1)
            alpha_f_batch = alpha_f.expand(batch_size, -1, -1, -1)

            # SDE integration: Euler-Maruyama of Schrödinger Bridge
            residuals = sde_integrate(
                model, coarse_up_t, alpha_c_batch, alpha_f_batch,
                sigma=sigma, n_ensemble=batch_size, n_steps=n_steps,
                device=device,
            )

            # Reconstruct fine fields
            fine_t = coarse_up_t + residuals  # (batch_size, 1, H_out, W_out)
            fine_np = fine_t.squeeze(1).cpu().numpy()
            fine_np = np.maximum(fine_np, 0)  # precipitation >= 0

            new_fields[batch_start:batch_end] = fine_np

        current_fields = new_fields
        current_lats = out_lats
        current_lons = out_lons

        elapsed = time.time() - t_step
        log.info(f"  Step {step_idx} ({in_res}→{out_res} km): "
                 f"({H_out},{W_out}), {elapsed:.1f}s, "
                 f"mean={current_fields.mean():.6f}, std={current_fields.std():.6f}")

        intermediates.append({
            "step": step_idx,
            "res_km": out_res,
            "mean": current_fields.mean(axis=0),
            "std": current_fields.std(axis=0),
            "lats": current_lats.copy(),
            "lons": current_lons.copy(),
            "shape": (H_out, W_out),
        })

    return current_fields, current_lats, current_lons, intermediates


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_resolution_progression(intermediates, prism, date_str, out_dir):
    """
    Plot ensemble mean and spread at each resolution step,
    with PRISM truth at the final resolution for comparison.

    Creates a 2-row figure:
      Row 1: Ensemble mean at each resolution (25km → 12.5 → 6.25 → 3.125 → 1.5625 + PRISM)
      Row 2: Ensemble spread at each resolution
    """
    n_steps = len(intermediates)  # includes the 25km input
    n_cols = n_steps + (1 if prism is not None else 0)  # +1 for PRISM

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7),
                             constrained_layout=True)

    # Global colorscale from all means + PRISM
    all_means = [inter["mean"] for inter in intermediates]
    vmin = min(m.min() for m in all_means)
    vmax = max(m.max() for m in all_means)
    if prism is not None:
        vmin = min(vmin, prism.min())
        vmax = max(vmax, prism.max())

    for col, inter in enumerate(intermediates):
        extent = [inter["lons"][0], inter["lons"][-1],
                  inter["lats"][-1], inter["lats"][0]]

        res = inter["res_km"]
        step = inter["step"]
        label = "Input (ERA5)" if step == -1 else f"Step {step}"

        # Row 0: ensemble mean
        im = axes[0, col].imshow(inter["mean"], cmap="YlGnBu",
                                 vmin=vmin, vmax=vmax,
                                 extent=extent, aspect="auto")
        axes[0, col].set_title(f"{label}\n{res} km\n{inter['shape']}",
                               fontsize=9)

        # Row 1: ensemble spread
        if inter["std"].max() > 0:
            axes[1, col].imshow(inter["std"], cmap="Oranges",
                                extent=extent, aspect="auto")
            axes[1, col].set_title(f"Spread\nmean={inter['std'].mean():.6f}",
                                   fontsize=9)
        else:
            axes[1, col].set_title("Spread\n(input — no spread)", fontsize=9)
            axes[1, col].imshow(inter["std"], cmap="Oranges",
                                extent=extent, aspect="auto")

    # PRISM column (if available)
    if prism is not None:
        col = n_steps
        final = intermediates[-1]
        extent = [final["lons"][0], final["lons"][-1],
                  final["lats"][-1], final["lats"][0]]

        axes[0, col].imshow(prism, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                            extent=extent, aspect="auto")
        axes[0, col].set_title(f"PRISM (truth)\n~0.8 km\n{prism.shape}",
                               fontsize=9, fontweight="bold")

        # Difference map (ensemble mean - PRISM)
        diff = final["mean"] - prism
        abs_max = max(abs(diff.min()), abs(diff.max()))
        if abs_max > 0:
            axes[1, col].imshow(diff, cmap="RdBu_r",
                                vmin=-abs_max, vmax=abs_max,
                                extent=extent, aspect="auto")
        axes[1, col].set_title(f"Mean − PRISM\nRMSE={np.sqrt((diff**2).mean()):.6f}",
                               fontsize=9)

    # Labels
    for ax in axes[0, :]:
        ax.set_ylabel("Lat")
    for ax in axes[-1, :]:
        ax.set_xlabel("Lon")

    axes[0, 0].set_ylabel("Ensemble Mean", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("Ensemble Spread", fontsize=11, fontweight="bold")

    fig.suptitle(f"CFM Recursive Downscaling — {date_str}", fontsize=13,
                 fontweight="bold")

    save_path = out_dir / f"cfm_progression_{date_str}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {save_path.name}")


def plot_comparison_with_baseline(
    cfm_ensemble, bicubic_final, prism, crps_map,
    final_lats, final_lons, date_str, metrics, out_dir,
):
    """
    5-panel comparison plot matching the XGBoost baseline format:
    PRISM | CFM Ensemble Mean | Bicubic | Ensemble Spread | CRPS Map
    """
    ens_mean = cfm_ensemble.mean(axis=0)
    ens_std = cfm_ensemble.std(axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)

    vmin = min(prism.min(), ens_mean.min(), bicubic_final.min())
    vmax = max(prism.max(), ens_mean.max(), bicubic_final.max())
    extent = [final_lons[0], final_lons[-1], final_lats[-1], final_lats[0]]

    axes[0].imshow(prism, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                   extent=extent, aspect="auto")
    axes[0].set_title("PRISM (truth)", fontsize=9, fontweight="bold")

    axes[1].imshow(ens_mean, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                   extent=extent, aspect="auto")
    axes[1].set_title(f"CFM Ensemble Mean\nRMSE={metrics['rmse_mean']:.6f}",
                      fontsize=9)

    axes[2].imshow(bicubic_final, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                   extent=extent, aspect="auto")
    axes[2].set_title(f"Bicubic\nRMSE={metrics['rmse_bicubic']:.6f}",
                      fontsize=9)

    axes[3].imshow(ens_std, cmap="Oranges", extent=extent, aspect="auto")
    axes[3].set_title(f"Ensemble Spread\nmean={ens_std.mean():.6f}",
                      fontsize=9)

    axes[4].imshow(crps_map, cmap="Reds", extent=extent, aspect="auto")
    axes[4].set_title(f"CRPS Map\nmean={metrics['crps']:.6f}", fontsize=9)

    for ax in axes:
        ax.set_xlabel("Lon")
    axes[0].set_ylabel("Lat")

    fig.suptitle(f"Ensemble Downscaling — {date_str}", fontsize=12)
    save_path = out_dir / f"cfm_ensemble_{date_str}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {save_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CFM recursive ensemble downscaling + PRISM validation."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained CFM checkpoint (.pt)")
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--prism-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-ensemble", type=int, default=100)
    parser.add_argument("--n-days", type=int, default=30,
                        help="Number of test days to evaluate")
    parser.add_argument("--n-steps", type=int, default=50,
                        help="Number of Euler-Maruyama steps for SDE integration")
    parser.add_argument("--sigma", type=float, default=None,
                        help="Schrödinger bridge σ (default: read from checkpoint)")
    parser.add_argument("--use-ema", action="store_true", default=True,
                        help="Use EMA model weights (default: True)")
    parser.add_argument("--ensemble-batch", type=int, default=None,
                        help="Process ensemble in chunks (for GPU memory)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    # Model architecture args (should match training)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--channel-mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--aef-dim", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-res-blocks", type=int, default=2)
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load model ───────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model import DownscalingUNet

    log.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Read σ from checkpoint or use override
    sigma = args.sigma
    if sigma is None:
        if "args" in ckpt and "sigma" in ckpt["args"]:
            sigma = ckpt["args"]["sigma"]
        else:
            sigma = 0.1
            log.warning(f"  σ not found in checkpoint, using default {sigma}")
    log.info(f"  σ = {sigma}")

    model = DownscalingUNet(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        channel_mult=tuple(args.channel_mult),
        aef_dim=args.aef_dim,
        time_dim=args.time_dim,
        num_heads=args.num_heads,
        num_res_blocks=args.num_res_blocks,
    ).to(device)

    # Load weights (prefer EMA)
    if args.use_ema and "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        log.info("  Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("  Loaded model weights")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Model parameters: {n_params:,}")

    # ── Load ERA5 data ───────────────────────────────────────────────────
    era5_dir = Path(args.era5_dir)
    era5_lats = np.load(era5_dir / "era5_lats.npy")
    era5_lons = np.load(era5_dir / "era5_lons.npy")
    A_fine = np.load(era5_dir / "pair_A_fine.npy")    # (T, 23, 17) 25km
    A_dates = np.load(era5_dir / "valid_times_A.npy")
    A_test_idx = np.load(era5_dir / "test_indices_A.npy")

    log.info(f"Test days available: {len(A_test_idx)}")

    # ── Select test days ─────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    n_eval = min(args.n_days, len(A_test_idx))
    eval_indices = rng.choice(A_test_idx, size=n_eval, replace=False)
    eval_indices.sort()
    log.info(f"Evaluating {n_eval} test days")

    # ── Run inference + PRISM validation ─────────────────────────────────
    all_crps = []
    all_rmse_mean = []
    all_rmse_bicubic = []
    all_coverage = {f"coverage_{l}": [] for l in [50, 80, 90]}
    all_spread_skill = []
    per_day_results = []

    for day_i, t_idx in enumerate(eval_indices):
        date = A_dates[t_idx]
        date_str = str(date)[:10]
        year = int(date_str[:4])

        log.info(f"Day {day_i + 1}/{n_eval}: {date_str}")

        era5_field = A_fine[t_idx]  # (23, 17)

        # ── Recursive downscale with CFM ─────────────────────────────────
        t0 = time.time()
        ensemble, final_lats, final_lons, intermediates = \
            recursive_downscale_cfm(
                era5_field, year, model, sigma,
                args.aef_dir, era5_lats, era5_lons,
                n_ensemble=args.n_ensemble,
                n_steps=args.n_steps,
                device=device,
                ensemble_batch=args.ensemble_batch,
            )
        log.info(f"  Total downscaling: {time.time() - t0:.1f}s, "
                 f"ensemble shape: {ensemble.shape}")

        # ── Load PRISM ───────────────────────────────────────────────────
        H_ens, W_ens = ensemble.shape[1], ensemble.shape[2]
        try:
            prism = load_prism_day(
                args.prism_dir, date, final_lats, final_lons)
        except Exception as e:
            log.warning(f"  PRISM load failed: {e}")
            prism = None

        if prism is not None and prism.shape != (H_ens, W_ens):
            prism = zoom(prism,
                         (H_ens / prism.shape[0], W_ens / prism.shape[1]),
                         order=1).astype(np.float32)
            prism = np.maximum(prism, 0)

        # ── Multi-resolution progression plot ────────────────────────────
        plot_resolution_progression(intermediates, prism, date_str, out_dir)

        # ── Evaluate against PRISM ───────────────────────────────────────
        if prism is None:
            log.warning(f"  No PRISM data for {date_str}, skipping metrics")
            continue

        ens_mean = ensemble.mean(axis=0)
        ens_std = ensemble.std(axis=0)

        # RMSE of ensemble mean
        rmse_mean = float(np.sqrt(((ens_mean - prism) ** 2).mean()))
        all_rmse_mean.append(rmse_mean)

        # RMSE of bicubic baseline
        bicubic_final = zoom(
            era5_field,
            (H_ens / era5_field.shape[0], W_ens / era5_field.shape[1]),
            order=3).astype(np.float32)
        bicubic_final = np.maximum(bicubic_final, 0)
        rmse_bicubic = float(np.sqrt(((bicubic_final - prism) ** 2).mean()))
        all_rmse_bicubic.append(rmse_bicubic)

        # CRPS
        mean_crps, crps_map = crps_grid(ensemble, prism)
        all_crps.append(mean_crps)

        # Coverage
        cov = ensemble_coverage(ensemble, prism)
        for k, v in cov.items():
            all_coverage[k].append(v)

        # Spread-skill
        spread = float(ens_std.mean())
        skill = float(np.abs(ens_mean - prism).mean())
        ss_ratio = spread / (skill + 1e-10)
        all_spread_skill.append(ss_ratio)

        day_metrics = {
            "rmse_mean": rmse_mean,
            "rmse_bicubic": rmse_bicubic,
            "crps": float(mean_crps),
            "spread": spread,
            "skill": skill,
            "spread_skill": ss_ratio,
            **cov,
        }

        log.info(f"  RMSE(ens mean): {rmse_mean:.6f}, "
                 f"RMSE(bicubic): {rmse_bicubic:.6f}")
        log.info(f"  CRPS: {mean_crps:.6f}, Spread/Skill: {ss_ratio:.3f}")
        log.info(f"  Coverage: "
                 + ", ".join(f"{k}={v:.3f}" for k, v in cov.items()))

        per_day_results.append({"date": date_str, **day_metrics})

        # ── Baseline-format comparison plot ──────────────────────────────
        if day_i < 10:
            plot_comparison_with_baseline(
                ensemble, bicubic_final, prism, crps_map,
                final_lats, final_lons, date_str, day_metrics, out_dir,
            )

    # ── Aggregate results ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("AGGREGATE RESULTS")

    results = {
        "model": str(args.checkpoint),
        "sigma": sigma,
        "n_steps": args.n_steps,
        "n_ensemble": args.n_ensemble,
        "use_ema": args.use_ema,
        "n_days_evaluated": len(all_crps),
        "rmse_ensemble_mean": {
            "mean": float(np.mean(all_rmse_mean)) if all_rmse_mean else None,
            "std": float(np.std(all_rmse_mean)) if all_rmse_mean else None,
        },
        "rmse_bicubic": {
            "mean": float(np.mean(all_rmse_bicubic)) if all_rmse_bicubic else None,
            "std": float(np.std(all_rmse_bicubic)) if all_rmse_bicubic else None,
        },
        "crps": {
            "mean": float(np.mean(all_crps)) if all_crps else None,
            "std": float(np.std(all_crps)) if all_crps else None,
        },
        "spread_skill_ratio": {
            "mean": float(np.mean(all_spread_skill)) if all_spread_skill else None,
            "std": float(np.std(all_spread_skill)) if all_spread_skill else None,
            "note": "ideal is ~1.0",
        },
        "coverage": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in all_coverage.items() if v
        },
        "per_day": per_day_results,
    }

    if all_crps:
        log.info(f"  RMSE (ens mean): {results['rmse_ensemble_mean']['mean']:.6f} "
                 f"± {results['rmse_ensemble_mean']['std']:.6f}")
        log.info(f"  RMSE (bicubic):  {results['rmse_bicubic']['mean']:.6f} "
                 f"± {results['rmse_bicubic']['std']:.6f}")
        log.info(f"  CRPS:            {results['crps']['mean']:.6f} "
                 f"± {results['crps']['std']:.6f}")
        log.info(f"  Spread/Skill:    {results['spread_skill_ratio']['mean']:.3f} "
                 f"± {results['spread_skill_ratio']['std']:.3f}")
        for k, v in results["coverage"].items():
            log.info(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")

    with open(out_dir / "cfm_ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nTotal time: {time.time() - t_start:.0f}s")
    log.info(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()