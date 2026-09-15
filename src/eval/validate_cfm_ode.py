#!/usr/bin/env python3
"""
Recursive ensemble downscaling with ODE-CFM, validated against PRISM.

Ensemble generation: run the deterministic ODE from different x0 ~ N(0,I).
Each initial noise vector produces a different sample from the learned
conditional distribution. No score head or SDE needed.

Pipeline:
  1. Start from ERA5 25km for each test day
  2. Recursively downscale: 25→12.5→6.25→3.125→1.5625 km
  3. At each step, generate N ensemble members via ODE from different x0
  4. Load PRISM 800m TIF, downsample to 1.5625km grid
  5. Evaluate: CRPS, spread-skill, coverage, RMSE

Note: model outputs are in mm/day (dataset scales ×1000).
      PRISM is in mm/day after conversion from the TIF.
      All comparisons are in mm/day.

Usage:
    python -u validate_cfm_ode.py \
        --checkpoint checkpoints/cfm_ode_v2/best_model_phase2.pt \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_tif_2018_2024 \
        --output-dir results/cfm_ode \
        --n-ensemble 20 \
        --n-days 10
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
import torch.nn.functional as F
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AEF_YEAR_OFFSET = 2017
M_TO_MM = 1000.0

DOWNSCALE_STEPS = [
    (25.0,   12.5,    25.0,   12.5),
    (12.5,    6.25,   12.5,    6.25),
    (6.25,    3.125,   6.25,   3.125),
    (3.125,   1.5625,  3.125,  1.5625),
]


def _format_res(res_km):
    if res_km == int(res_km):
        return str(int(res_km))
    return str(res_km)


def load_aef_nc(nc_path):
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32).transpose(1, 2, 0)
    ds.close()
    return arr


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
    raise FileNotFoundError(f"AEF not found: t{t_idx}, {res_km}km")


def load_aef_as_tensor(aef_dir, t_idx, res_km, target_h, target_w, device):
    try:
        aef = load_aef_nc(find_aef_file(aef_dir, t_idx, res_km))
        D = aef.shape[2]
        aef_resized = np.stack([
            zoom(aef[:, :, d], (target_h / aef.shape[0], target_w / aef.shape[1]),
                 order=1)
            for d in range(D)
        ], axis=0).astype(np.float32)
        return torch.from_numpy(aef_resized).unsqueeze(0).to(device)
    except FileNotFoundError as e:
        log.warning(f"  {e}, using zeros")
        return torch.zeros(1, 64, target_h, target_w, device=device)


@torch.no_grad()
def ode_integrate(model, coarse_up, alpha_c, alpha_f,
                  n_ensemble, n_steps=50, device="cuda"):
    """
    Euler integration of the flow ODE from t=0 (noise) to t=1 (residual).

    dx/dt = v_θ(t, [coarse_up, x_t], α_c, α_f)

    Each ensemble member starts from a different x0 ~ N(0, I).
    The ODE is deterministic given x0, so diversity comes entirely
    from the initial noise.

    Args:
        model: DownscalingUNet returning v_theta only
        coarse_up: (1, 1, H, W) in mm/day
        alpha_c, alpha_f: (1, D, H, W)
        n_ensemble: number of members
        n_steps: Euler steps

    Returns:
        residuals: (n_ensemble, 1, H, W) predicted residuals in mm/day
    """
    B = n_ensemble
    _, _, H, W = coarse_up.shape

    coarse_up_batch = coarse_up.expand(B, -1, -1, -1)
    alpha_c_batch = alpha_c.expand(B, -1, -1, -1)
    alpha_f_batch = alpha_f.expand(B, -1, -1, -1)

    # Each member starts from different noise
    x = torch.randn(B, 1, H, W, device=device)

    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step * dt
        t_batch = torch.full((B,), t_val, device=device)

        x_input = torch.cat([coarse_up_batch, x], dim=1)
        
        # Cast to bfloat16 for the forward pass - this allows the use of FlashAttention
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            v = model(t_batch, x_input, alpha_c_batch, alpha_f_batch)

        # The ODE step remains in float32 for accuracy
        x = x + v.to(torch.float32) * dt

    return x


def load_prism_day(prism_dir, date, target_lats, target_lons):
    """Load PRISM TIF, crop to IL, regrid. Returns mm/day."""
    import rasterio

    date_str = str(date)[:10].replace("-", "")
    year = date_str[:4]
    tif_path = Path(prism_dir) / year / f"prism_ppt_us_30s_{date_str}.tif"

    if not tif_path.exists():
        log.warning(f"  PRISM tif file not found at: {str(tif_path)} ")
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

    # 1. Handle [0, 360] to [-180, 180] conversion if needed
    target_lons_shifted = np.where(target_lons > 180, target_lons - 360, target_lons)
    
    # 2. Force Western Hemisphere longitudes to be negative (fixes Degrees West vs East)
    # If the array is showing [87.4, 91.6], this flips it to [-87.4, -91.6]
    target_lons_shifted = -np.abs(target_lons_shifted)

    lat_min, lat_max = target_lats.min() - 0.1, target_lats.max() + 0.1
    # Note: Because they are negative, min() and max() behavior flips natively
    lon_min, lon_max = target_lons_shifted.min() - 0.1, target_lons_shifted.max() + 0.1

    lat_mask = (prism_lats >= lat_min) & (prism_lats <= lat_max)
    lon_mask = (prism_lons >= lon_min) & (prism_lons <= lon_max)

    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        log.warning(f"  PRISM data sum = 0 (Bounds: Lat {lat_min:.2f} to {lat_max:.2f}, Lon {lon_min:.2f} to {lon_max:.2f})")
        return None

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    cropped = prism_data[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

    # PRISM is already in mm/day — no conversion needed
    target_h, target_w = len(target_lats), len(target_lons)
    regridded = zoom(cropped, (target_h / cropped.shape[0], target_w / cropped.shape[1]),
                     order=1).astype(np.float32)
    return np.maximum(regridded, 0)


def crps_ensemble(ensemble, observation):
    N = len(ensemble)
    mae = np.abs(ensemble - observation).mean()
    sorted_ens = np.sort(ensemble)
    diff_sum = sum((2 * i - N) * sorted_ens[i] for i in range(N))
    spread = 2 * diff_sum / (N * N)
    return mae - 0.5 * spread


def crps_grid(ensemble_grids, observation_grid):
    H, W = observation_grid.shape
    crps_values = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            crps_values[i, j] = crps_ensemble(
                ensemble_grids[:, i, j], observation_grid[i, j])
    return crps_values.mean(), crps_values


def ensemble_coverage(ensemble_grids, observation_grid, levels=(0.5, 0.8, 0.9)):
    results = {}
    for level in levels:
        alpha = (1 - level) / 2
        lo = np.quantile(ensemble_grids, alpha, axis=0)
        hi = np.quantile(ensemble_grids, 1 - alpha, axis=0)
        covered = ((observation_grid >= lo) & (observation_grid <= hi)).mean()
        results[f"coverage_{int(level*100)}"] = float(covered)
    return results


def recursive_downscale(
    era5_25km,       # (H25, W25) in meters/day
    year,
    model,
    aef_dir,
    era5_lats,
    era5_lons,
    n_ensemble=20,
    n_steps=50,
    device="cuda",
    ensemble_batch=None,
):
    """
    Recursively downscale from 25km to 1.5625km.

    Input is in meters/day (raw ERA5). Converted to mm/day for the model.
    Output is in mm/day for comparison with PRISM.
    """
    t_idx = year - AEF_YEAR_OFFSET

    if ensemble_batch is None:
        ensemble_batch = n_ensemble

    # Convert ERA5 from meters to mm
    era5_mm = era5_25km * M_TO_MM

    current_fields = np.stack([era5_mm] * n_ensemble)  # (N, H, W) in mm/day
    current_lats = era5_lats.copy()
    current_lons = era5_lons.copy()

    intermediates = [{
        "step": -1, "res_km": 25.0,
        "mean": era5_mm.copy(), "std": np.zeros_like(era5_mm),
        "lats": current_lats.copy(), "lons": current_lons.copy(),
        "shape": era5_mm.shape,
    }]

    for step_idx, (in_res, out_res, aef_c_res, aef_f_res) in enumerate(DOWNSCALE_STEPS):
        t_step = time.time()
        H_in, W_in = current_fields.shape[1], current_fields.shape[2]
        H_out, W_out = H_in * 2, W_in * 2

        out_lats = np.linspace(current_lats[0], current_lats[-1], H_out)
        out_lons = np.linspace(current_lons[0], current_lons[-1], W_out)

        alpha_c = load_aef_as_tensor(Path(aef_dir), t_idx, aef_c_res,
                                     H_out, W_out, device)
        alpha_f = load_aef_as_tensor(Path(aef_dir), t_idx, aef_f_res,
                                     H_out, W_out, device)

        new_fields = np.empty((n_ensemble, H_out, W_out), dtype=np.float32)

        for batch_start in range(0, n_ensemble, ensemble_batch):
            batch_end = min(batch_start + ensemble_batch, n_ensemble)
            batch_size = batch_end - batch_start

            # Bicubic upsample each member
            coarse_up_list = []
            for i in range(batch_start, batch_end):
                cu = zoom(current_fields[i],
                          (H_out / H_in, W_out / W_in),
                          order=3).astype(np.float32)
                coarse_up_list.append(cu)
            coarse_up_np = np.stack(coarse_up_list)

            # Model expects (B, 1, H, W) in mm/day
            coarse_up_t = torch.from_numpy(coarse_up_np).unsqueeze(1).to(device)

            alpha_c_batch = alpha_c.expand(batch_size, -1, -1, -1)
            alpha_f_batch = alpha_f.expand(batch_size, -1, -1, -1)

            # ODE integration → residuals in mm/day
            residuals = ode_integrate(
                model, coarse_up_t, alpha_c_batch, alpha_f_batch,
                n_ensemble=batch_size, n_steps=n_steps, device=device,
            )

            fine_t = coarse_up_t + residuals
            fine_np = fine_t.squeeze(1).cpu().numpy()
            fine_np = np.maximum(fine_np, 0)

            new_fields[batch_start:batch_end] = fine_np

        current_fields = new_fields
        current_lats = out_lats
        current_lons = out_lons

        elapsed = time.time() - t_step
        log.info(f"  Step {step_idx} ({in_res}→{out_res}km): "
                 f"({H_out},{W_out}), {elapsed:.1f}s, "
                 f"mean={current_fields.mean():.3f} mm/day")

        intermediates.append({
            "step": step_idx, "res_km": out_res,
            "mean": current_fields.mean(axis=0),
            "std": current_fields.std(axis=0),
            "lats": current_lats.copy(), "lons": current_lons.copy(),
            "shape": (H_out, W_out),
        })

    return current_fields, current_lats, current_lons, intermediates


def plot_ensemble(ensemble, bicubic_final, prism, crps_map,
                  final_lats, final_lons, date_str, metrics, out_dir):
    ens_mean = ensemble.mean(axis=0)
    ens_std = ensemble.std(axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)

    vmin = min(prism.min(), ens_mean.min(), bicubic_final.min())
    vmax = max(prism.max(), ens_mean.max(), bicubic_final.max())
    extent = [final_lons[0], final_lons[-1], final_lats[-1], final_lats[0]]

    axes[0].imshow(prism, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                   extent=extent, aspect="auto")
    axes[0].set_title("PRISM (truth)", fontsize=9, fontweight="bold")

    axes[1].imshow(ens_mean, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                   extent=extent, aspect="auto")
    axes[1].set_title(f"CFM Ens Mean\nRMSE={metrics['rmse_mean']:.3f} mm", fontsize=9)

    axes[2].imshow(bicubic_final, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                   extent=extent, aspect="auto")
    axes[2].set_title(f"Bicubic\nRMSE={metrics['rmse_bicubic']:.3f} mm", fontsize=9)

    axes[3].imshow(ens_std, cmap="Oranges", extent=extent, aspect="auto")
    axes[3].set_title(f"Ens Spread\nmean={ens_std.mean():.3f} mm", fontsize=9)

    axes[4].imshow(crps_map, cmap="Reds", extent=extent, aspect="auto")
    axes[4].set_title(f"CRPS\nmean={metrics['crps']:.3f} mm", fontsize=9)

    for ax in axes:
        ax.set_xlabel("Lon")
    axes[0].set_ylabel("Lat")

    fig.suptitle(f"CFM ODE Ensemble — {date_str} (all mm/day)", fontsize=12)
    fig.savefig(out_dir / f"cfm_ode_{date_str}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved cfm_ode_{date_str}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--prism-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-ensemble", type=int, default=20)
    parser.add_argument("--n-days", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--ensemble-batch", type=int, default=None)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model_ode import DownscalingUNet

    log.info(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model = DownscalingUNet(
        in_channels=2, out_channels=1,
        base_channels=48, channel_mult=(1, 2, 4),
        aef_dim=64, time_dim=64, num_heads=2,
    ).to(device)

    if args.use_ema and "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        log.info("  Loaded EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("  Loaded model weights")
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {n_params:,}")
    log.info(f"  Ensemble Batch size: {args.ensemble_batch}")

    era5_dir = Path(args.era5_dir)
    era5_lats = np.load(era5_dir / "era5_lats.npy")
    era5_lons = np.load(era5_dir / "era5_lons.npy")
    A_fine = np.load(era5_dir / "pair_A_fine.npy")  # meters/day
    A_dates = np.load(era5_dir / "valid_times_A.npy")
    A_test_idx = np.load(era5_dir / "test_indices_A.npy")

    log.info(f"Test days: {len(A_test_idx)}")

    rng = np.random.RandomState(args.seed)
    n_eval = min(args.n_days, len(A_test_idx))
    eval_indices = rng.choice(A_test_idx, size=n_eval, replace=False)
    eval_indices.sort()

    # Run evaluation
    all_metrics = {k: [] for k in ["crps", "rmse_mean", "rmse_bicubic",
                                    "spread_skill", "coverage_50",
                                    "coverage_80", "coverage_90"]}
    per_day = []

    for day_i, t_idx in enumerate(eval_indices):
        date = A_dates[t_idx]
        date_str = str(date)[:10]
        year = int(date_str[:4])
        log.info(f"Day {day_i+1}/{n_eval}: {date_str}")

        era5_field = A_fine[t_idx]  # meters/day

        t0 = time.time()
        ensemble, final_lats, final_lons, intermediates = recursive_downscale(
            era5_field, year, model, args.aef_dir,
            era5_lats, era5_lons,
            n_ensemble=args.n_ensemble, n_steps=args.n_steps,
            device=device, ensemble_batch=args.ensemble_batch,
        )
        log.info(f"  Downscaled in {time.time()-t0:.1f}s, shape={ensemble.shape}")

        # ensemble is in mm/day
        H_ens, W_ens = ensemble.shape[1], ensemble.shape[2]

        # Load PRISM (mm/day)
        try:
            prism = load_prism_day(args.prism_dir, date, final_lats, final_lons)
        except Exception as e:
            log.warning(f"  PRISM failed: {e}")
            prism = None

        if prism is not None and prism.shape != (H_ens, W_ens):
            prism = zoom(prism, (H_ens / prism.shape[0], W_ens / prism.shape[1]),
                         order=1).astype(np.float32)
            prism = np.maximum(prism, 0)

        if prism is None:
            log.warning(f"  No PRISM for {date_str}")
            continue

        # All in mm/day now
        ens_mean = ensemble.mean(axis=0)
        ens_std = ensemble.std(axis=0)

        rmse_mean = float(np.sqrt(((ens_mean - prism) ** 2).mean()))

        # Bicubic baseline (meters → mm)
        bicubic = zoom(era5_field,
                       (H_ens / era5_field.shape[0], W_ens / era5_field.shape[1]),
                       order=3).astype(np.float32)
        bicubic = np.maximum(bicubic, 0) * M_TO_MM  # to mm/day
        rmse_bicubic = float(np.sqrt(((bicubic - prism) ** 2).mean()))

        mean_crps, crps_map = crps_grid(ensemble, prism)
        cov = ensemble_coverage(ensemble, prism)

        spread = float(ens_std.mean())
        skill = float(np.abs(ens_mean - prism).mean())
        ss = spread / (skill + 1e-10)

        day_metrics = {
            "rmse_mean": rmse_mean, "rmse_bicubic": rmse_bicubic,
            "crps": float(mean_crps), "spread_skill": ss, **cov,
        }

        log.info(f"  RMSE(ens)={rmse_mean:.3f}, RMSE(bic)={rmse_bicubic:.3f}, "
                 f"CRPS={mean_crps:.3f}, S/S={ss:.3f}")
        log.info(f"  Coverage: " + ", ".join(f"{k}={v:.3f}" for k, v in cov.items()))

        for k, v in day_metrics.items():
            if k in all_metrics:
                all_metrics[k].append(v)
        per_day.append({"date": date_str, **day_metrics})

        if day_i < 5:
            plot_ensemble(ensemble, bicubic, prism, crps_map,
                          final_lats, final_lons, date_str, day_metrics, out_dir)

    # Aggregate
    log.info("=" * 60)
    log.info("AGGREGATE (all mm/day)")
    results = {"per_day": per_day}
    for k, vals in all_metrics.items():
        if vals:
            results[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            log.info(f"  {k}: {results[k]['mean']:.4f} ± {results[k]['std']:.4f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"Total: {time.time()-t_start:.0f}s")
    log.info(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()