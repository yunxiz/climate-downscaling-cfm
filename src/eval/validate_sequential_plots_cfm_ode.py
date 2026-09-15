#!/usr/bin/env python3
"""
Same as validate_cfm_ode.py but adds sequential progression plots
showing the full recursive pipeline at each resolution step.

Usage:
    python -u validate_cfm_ode_sequential.py \
        --checkpoint checkpoints/cfm_ode_v2/best_model_phase2.pt \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_tif_2025 \
        --output-dir results/cfm_ode_sequential \
        --n-ensemble 10 --n-days 5 --n-steps 50 \
        --ensemble-batch 1
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
    B = n_ensemble
    _, _, H, W = coarse_up.shape

    coarse_up_batch = coarse_up.expand(B, -1, -1, -1)
    alpha_c_batch = alpha_c.expand(B, -1, -1, -1)
    alpha_f_batch = alpha_f.expand(B, -1, -1, -1)

    x = torch.randn(B, 1, H, W, device=device)
    dt = 1.0 / n_steps

    for step in range(n_steps):
        t_val = step * dt
        t_batch = torch.full((B,), t_val, device=device)
        x_input = torch.cat([coarse_up_batch, x], dim=1)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            v = model(t_batch, x_input, alpha_c_batch, alpha_f_batch)

        x = x + v.to(torch.float32) * dt

    return x


def load_prism_day(prism_dir, date, target_lats, target_lons):
    import rasterio

    date_str = str(date)[:10].replace("-", "")
    year = date_str[:4]
    tif_path = Path(prism_dir) / year / f"prism_ppt_us_30s_{date_str}.tif"

    if not tif_path.exists():
        log.warning(f"  PRISM not found: {tif_path}")
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

    target_lons_shifted = np.where(target_lons > 180, target_lons - 360, target_lons)
    target_lons_shifted = -np.abs(target_lons_shifted)

    lat_min, lat_max = target_lats.min() - 0.1, target_lats.max() + 0.1
    lon_min, lon_max = target_lons_shifted.min() - 0.1, target_lons_shifted.max() + 0.1

    lat_mask = (prism_lats >= lat_min) & (prism_lats <= lat_max)
    lon_mask = (prism_lons >= lon_min) & (prism_lons <= lon_max)

    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        log.warning(f"  PRISM crop empty")
        return None

    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    cropped = prism_data[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]

    target_h, target_w = len(target_lats), len(target_lons)
    regridded = zoom(cropped, (target_h / cropped.shape[0], target_w / cropped.shape[1]),
                     order=1).astype(np.float32)
    return np.maximum(regridded, 0)


def recursive_downscale(
    era5_25km, year, model, aef_dir, era5_lats, era5_lons,
    n_ensemble=20, n_steps=50, device="cuda", ensemble_batch=None,
):
    t_idx = year - AEF_YEAR_OFFSET
    if ensemble_batch is None:
        ensemble_batch = n_ensemble

    era5_mm = era5_25km * M_TO_MM
    current_fields = np.stack([era5_mm] * n_ensemble)
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

            coarse_up_list = []
            for i in range(batch_start, batch_end):
                cu = zoom(current_fields[i],
                          (H_out / H_in, W_out / W_in),
                          order=3).astype(np.float32)
                coarse_up_list.append(cu)
            coarse_up_np = np.stack(coarse_up_list)

            coarse_up_t = torch.from_numpy(coarse_up_np).unsqueeze(1).to(device)
            alpha_c_batch = alpha_c.expand(batch_size, -1, -1, -1)
            alpha_f_batch = alpha_f.expand(batch_size, -1, -1, -1)

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


# ── Progression plot ─────────────────────────────────────────────────────────

def plot_progression(intermediates, prism, date_str, out_dir):
    """
    Row 1: Ensemble mean at each resolution + PRISM truth
    Row 2: Ensemble spread at each resolution + error vs PRISM

    Columns: Input(25km) | 12.5km | 6.25km | 3.125km | 1.5625km | PRISM
    """
    n_steps = len(intermediates)
    has_prism = prism is not None
    n_cols = n_steps + (1 if has_prism else 0)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7),
                             constrained_layout=True)

    all_means = [inter["mean"] for inter in intermediates]
    vmin = min(m.min() for m in all_means)
    vmax = max(m.max() for m in all_means)
    if has_prism:
        vmin = min(vmin, prism.min())
        vmax = max(vmax, prism.max())
    if vmax <= vmin:
        vmax = vmin + 0.1

    for col, inter in enumerate(intermediates):
        extent = [inter["lons"][0], inter["lons"][-1],
                  inter["lats"][-1], inter["lats"][0]]
        H, W = inter["shape"]
        res = inter["res_km"]
        step = inter["step"]

        label = f"ERA5 Input\n{res}km ({H}×{W})" if step == -1 else f"Step {step}\n{res}km ({H}×{W})"

        axes[0, col].imshow(inter["mean"], cmap="YlGnBu",
                            vmin=vmin, vmax=vmax,
                            extent=extent, aspect="auto")
        axes[0, col].set_title(label, fontsize=8)
        axes[0, col].tick_params(labelsize=6)

        std = inter["std"]
        axes[1, col].imshow(std, cmap="Oranges", extent=extent, aspect="auto")
        if std.max() > 0:
            axes[1, col].set_title(f"Spread\nμ={std.mean():.2f} mm", fontsize=8)
        else:
            axes[1, col].set_title("(no spread)", fontsize=8)
        axes[1, col].tick_params(labelsize=6)

    if has_prism:
        col = n_steps
        final = intermediates[-1]
        extent = [final["lons"][0], final["lons"][-1],
                  final["lats"][-1], final["lats"][0]]

        axes[0, col].imshow(prism, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                            extent=extent, aspect="auto")
        axes[0, col].set_title(f"PRISM Truth\n~0.8km ({prism.shape[0]}×{prism.shape[1]})",
                               fontsize=8, fontweight="bold", color="darkgreen")
        axes[0, col].tick_params(labelsize=6)

        diff = final["mean"] - prism
        rmse = np.sqrt((diff ** 2).mean())
        abs_max = max(abs(diff.min()), abs(diff.max()), 0.01)
        axes[1, col].imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                            extent=extent, aspect="auto")
        axes[1, col].set_title(f"Error vs PRISM\nRMSE={rmse:.2f} mm",
                               fontsize=8, color="red")
        axes[1, col].tick_params(labelsize=6)

    axes[0, 0].set_ylabel("Ensemble Mean\n(mm/day)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Ensemble Spread\n(mm/day)", fontsize=10, fontweight="bold")

    fig.suptitle(f"Recursive Downscaling Pipeline — {date_str}",
                 fontsize=13, fontweight="bold")

    save_path = out_dir / f"progression_{date_str}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {save_path.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--prism-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-ensemble", type=int, default=10)
    parser.add_argument("--n-days", type=int, default=5)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--ensemble-batch", type=int, default=1)
    parser.add_argument("--use-ema", action="store_true", default=True)
    parser.add_argument("--rainy-only", action="store_true", default=False,
                        help="Only sample days with domain-mean precip > 0.1 mm/day")
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
    from src.models.model_ode import DownscalingUNet

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
    log.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load ERA5
    era5_dir = Path(args.era5_dir)
    era5_lats = np.load(era5_dir / "era5_lats.npy")
    era5_lons = np.load(era5_dir / "era5_lons.npy")
    A_fine = np.load(era5_dir / "pair_A_fine.npy")
    A_dates = np.load(era5_dir / "valid_times_A.npy")
    A_test_idx = np.load(era5_dir / "test_indices_A.npy")

    log.info(f"Test days: {len(A_test_idx)}")

    # Filter to rainy days if requested
    if args.rainy_only:
        candidate_idx = []
        for t in A_test_idx:
            if A_fine[t].mean() > 0.0001:  # > 0.1 mm/day
                candidate_idx.append(t)
        candidate_idx = np.array(candidate_idx)
        log.info(f"Rainy test days: {len(candidate_idx)} / {len(A_test_idx)}")
    else:
        candidate_idx = A_test_idx

    rng = np.random.RandomState(args.seed)
    n_eval = min(args.n_days, len(candidate_idx))
    eval_indices = rng.choice(candidate_idx, size=n_eval, replace=False)
    eval_indices.sort()

    for day_i, t_idx in enumerate(eval_indices):
        date = A_dates[t_idx]
        date_str = str(date)[:10]
        year = int(date_str[:4])
        log.info(f"Day {day_i+1}/{n_eval}: {date_str} "
                 f"(domain mean={A_fine[t_idx].mean()*M_TO_MM:.2f} mm/day)")

        t0 = time.time()
        ensemble, final_lats, final_lons, intermediates = recursive_downscale(
            A_fine[t_idx], year, model, args.aef_dir,
            era5_lats, era5_lons,
            n_ensemble=args.n_ensemble, n_steps=args.n_steps,
            device=device, ensemble_batch=args.ensemble_batch,
        )
        log.info(f"  Downscaled in {time.time()-t0:.1f}s, shape={ensemble.shape}")

        H_ens, W_ens = ensemble.shape[1], ensemble.shape[2]

        # Load PRISM
        try:
            prism = load_prism_day(args.prism_dir, date, final_lats, final_lons)
        except Exception as e:
            log.warning(f"  PRISM failed: {e}")
            prism = None

        if prism is not None and prism.shape != (H_ens, W_ens):
            prism = zoom(prism, (H_ens / prism.shape[0], W_ens / prism.shape[1]),
                         order=1).astype(np.float32)
            prism = np.maximum(prism, 0)

        # Always plot progression (works with or without PRISM)
        plot_progression(intermediates, prism, date_str, out_dir)

        if prism is not None:
            ens_mean = ensemble.mean(axis=0)
            rmse = np.sqrt(((ens_mean - prism) ** 2).mean())
            log.info(f"  RMSE vs PRISM: {rmse:.3f} mm/day")

    log.info(f"Total: {time.time()-t_start:.0f}s")
    log.info(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()