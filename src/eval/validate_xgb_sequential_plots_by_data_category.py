#!/usr/bin/env python3
"""
XGBoost baseline validation with same rain categorization, progression plots,
and results.json format as the CFM validation script.

Three categories:
  - "both_rain":  ERA5 > threshold AND PRISM > threshold
  - "era5_only":  ERA5 rain but PRISM dry
  - "prism_only": PRISM rain but ERA5 dry

Usage:
    python -u slurm_validate_xgb_sequential_plots_by_data_category.py \
        --xgb-model results/baselines/xgb_with_aef.json \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_tif_2025 \
        --output-dir results/sequential_plots_xgb_categorized \
        --n-ensemble 40 --n-days-per-cat 10
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
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AEF_YEAR_OFFSET = 2017
M_TO_MM = 1000.0
D_AEF = 64
RAIN_THRESHOLD_MM = 0.1

TRAIN_YEARS = set(range(2017, 2025))

DOWNSCALE_STEPS = [
    (25.0,    12.5,    25.0,    12.5),
    (12.5,     6.25,   12.5,     6.25),
    ( 6.25,    3.125,   6.25,    3.125),
    ( 3.125,   1.5625,  3.125,   1.5625),
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


def resize_aef_to_grid(aef, target_h, target_w):
    D = aef.shape[2]
    return np.stack([
        zoom(aef[:, :, d], (target_h / aef.shape[0], target_w / aef.shape[1]), order=1)
        for d in range(D)
    ], axis=-1).astype(np.float32)



def build_pixel_features(field_2d_mm, aef_coarse, aef_fine):
    H, W = field_2d_mm.shape
    D = aef_coarse.shape[2]
    n_pixels = H * W
    coarse_log = np.log1p(np.maximum(field_2d_mm, 0)).ravel()
    X = np.empty((n_pixels, 1 + D + D), dtype=np.float32)
    col = 0
    X[:, col] = coarse_log; col += 1
    X[:, col:col + D] = aef_coarse.reshape(-1, D); col += D
    X[:, col:col + D] = aef_fine.reshape(-1, D)
    return X


def inverse_log_residual(coarse_up, log_residual):
    log_coarse = np.log1p(np.maximum(coarse_up, 0))
    return np.maximum(np.expm1(log_coarse + log_residual), 0)



def load_prism_day(prism_dir, date, target_lats, target_lons):
    import rasterio

    date_str = str(date)[:10].replace("-", "")
    year = date_str[:4]
    tif_path = Path(prism_dir) / year / f"prism_ppt_us_30s_{date_str}.tif"
    if not tif_path.exists():
        return None

    with rasterio.open(tif_path) as src:
        tr = src.transform
        data = src.read(1).astype(np.float32)
        H_p, W_p = data.shape
        prism_lons = np.array([tr[2] + tr[0] * (j + 0.5) for j in range(W_p)])
        prism_lats = np.array([tr[5] + tr[4] * (i + 0.5) for i in range(H_p)])

    data[data < -900] = 0

    target_lons_neg = -np.abs(np.where(target_lons > 180,
                                       target_lons - 360, target_lons))
    lat_min, lat_max = target_lats.min() - 0.1, target_lats.max() + 0.1
    lon_min, lon_max = target_lons_neg.min() - 0.1, target_lons_neg.max() + 0.1

    lat_mask = (prism_lats >= lat_min) & (prism_lats <= lat_max)
    lon_mask = (prism_lons >= lon_min) & (prism_lons <= lon_max)
    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        return None

    li = np.where(lat_mask)[0]
    wi = np.where(lon_mask)[0]
    cropped = data[li[0]:li[-1] + 1, wi[0]:wi[-1] + 1]

    H_t, W_t = len(target_lats), len(target_lons)
    regrid = zoom(cropped, (H_t / cropped.shape[0], W_t / cropped.shape[1]),
                  order=1).astype(np.float32)
    return np.maximum(regrid, 0)



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
    era5_25km_mm, year, booster, aef_dir, noise_pools,
    n_ensemble=40, seed=None,
):
    import xgboost as xgb

    rng = np.random.RandomState(seed)
    t_idx = year - AEF_YEAR_OFFSET

    current_fields = np.stack([era5_25km_mm.astype(np.float32)] * n_ensemble)

    intermediates = [{
        "step": -1, "res_km": 25.0,
        "mean": era5_25km_mm.copy(),
        "std": np.zeros_like(era5_25km_mm),
        "shape": era5_25km_mm.shape,
    }]

    for step_idx, (in_res, out_res, aef_c_res, aef_f_res) in enumerate(DOWNSCALE_STEPS):
        H_in, W_in = current_fields.shape[1], current_fields.shape[2]
        H_out, W_out = H_in * 2, W_in * 2

        try:
            aef_c = load_aef_nc(find_aef_file(Path(aef_dir), t_idx, aef_c_res))
            aef_f = load_aef_nc(find_aef_file(Path(aef_dir), t_idx, aef_f_res))
        except FileNotFoundError:
            aef_c = np.zeros((H_out, W_out, D_AEF), dtype=np.float32)
            aef_f = np.zeros((H_out, W_out, D_AEF), dtype=np.float32)

        aef_c_resized = resize_aef_to_grid(aef_c, H_out, W_out)
        aef_f_resized = resize_aef_to_grid(aef_f, H_out, W_out)

        noise_pool = noise_pools.get(step_idx, np.zeros(1000, dtype=np.float32))

        new_fields = np.empty((n_ensemble, H_out, W_out), dtype=np.float32)
        for i in range(n_ensemble):
            coarse_up = zoom(current_fields[i],
                             (H_out / H_in, W_out / W_in),
                             order=3).astype(np.float32)
            coarse_up = np.maximum(coarse_up, 0)

            X = build_pixel_features(coarse_up, aef_c_resized, aef_f_resized)
            r_log_det = booster.predict(xgb.DMatrix(X)).reshape(H_out, W_out)
            noise = rng.choice(noise_pool, size=(H_out, W_out), replace=True)
            new_fields[i] = inverse_log_residual(coarse_up, r_log_det + noise)

        current_fields = new_fields

        intermediates.append({
            "step": step_idx, "res_km": out_res,
            "mean": current_fields.mean(axis=0),
            "std": current_fields.std(axis=0),
            "shape": (H_out, W_out),
        })

    return current_fields, intermediates



def categorize_days(A_fine, A_dates, A_test_idx, prism_dir, era5_lats, era5_lons):
    categories = {"both_rain": [], "era5_only": [], "prism_only": [], "both_dry": []}

    for t_idx in A_test_idx:
        date = A_dates[t_idx]
        date_str = str(date)[:10]
        era5_mean_mm = float(A_fine[t_idx].mean() * M_TO_MM)
        era5_rainy = era5_mean_mm > RAIN_THRESHOLD_MM

        prism = None
        try:
            prism = load_prism_day(prism_dir, date, era5_lats, era5_lons)
        except Exception:
            pass

        if prism is None:
            continue

        prism_mean_mm = float(prism.mean())
        prism_rainy = prism_mean_mm > RAIN_THRESHOLD_MM

        if era5_rainy and prism_rainy:
            cat = "both_rain"
        elif era5_rainy and not prism_rainy:
            cat = "era5_only"
        elif not era5_rainy and prism_rainy:
            cat = "prism_only"
        else:
            cat = "both_dry"

        categories[cat].append({
            "t_idx": int(t_idx),
            "date": date_str,
            "era5_mean_mm": era5_mean_mm,
            "prism_mean_mm": prism_mean_mm,
        })

    for cat, days in categories.items():
        log.info(f"  Category '{cat}': {len(days)} days")

    return categories


def plot_progression(intermediates, prism, date_str, category, era5_lats, era5_lons, out_dir):
    n_steps = len(intermediates)
    has_prism = prism is not None
    n_cols = n_steps + (1 if has_prism else 0)

    # Increased width slightly to accommodate colorbars
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols + 1.5, 7),
                             constrained_layout=True)

    # Ensure axes is always 2D even if n_cols == 1
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # --- 1. Calculate Shared Scales ---
    # Top Row: Means and PRISM
    all_means = [inter["mean"] for inter in intermediates]
    vmin = min(m.min() for m in all_means)
    vmax = max(m.max() for m in all_means)
    if has_prism:
        vmin = min(vmin, prism.min())
        vmax = max(vmax, prism.max())
    if vmax <= vmin:
        vmax = vmin + 0.1

    # Bottom Row: Spread (ensure all spread plots share the same scale)
    all_stds = [inter["std"] for inter in intermediates]
    std_vmin = 0
    std_vmax = max(s.max() for s in all_stds) if all_stds else 0.1
    if std_vmax <= 0:
        std_vmax = 0.1

    # --- 2. Plot Iterations ---
    im_mean = None
    im_std = None

    for col, inter in enumerate(intermediates):
        H, W = inter["shape"]
        lats = np.linspace(era5_lats[0], era5_lats[-1], H)
        lons = np.linspace(era5_lons[0], era5_lons[-1], W)
        extent = [lons[0], lons[-1], lats[-1], lats[0]]
        res = inter["res_km"]
        step = inter["step"]

        label = f"ERA5 Input\n{res}km ({H}×{W})" if step == -1 else f"Step {step}\n{res}km ({H}×{W})"

        # Top row: Mean
        im_mean = axes[0, col].imshow(inter["mean"], cmap="YlGnBu",
                                      vmin=vmin, vmax=vmax,
                                      extent=extent, aspect="auto")
        axes[0, col].set_title(label, fontsize=8)
        axes[0, col].tick_params(labelsize=6)

        # Bottom row: Spread
        std = inter["std"]
        im_std = axes[1, col].imshow(std, cmap="Oranges", 
                                     vmin=std_vmin, vmax=std_vmax, 
                                     extent=extent, aspect="auto")
        if std.max() > 0:
            axes[1, col].set_title(f"Spread\nμ={std.mean():.2f} mm", fontsize=8)
        else:
            axes[1, col].set_title("(no spread)", fontsize=8)
        axes[1, col].tick_params(labelsize=6)

    # --- 3. Plot PRISM Truth and Error ---
    if has_prism:
        col = n_steps
        final = intermediates[-1]
        H, W = final["shape"]
        lats = np.linspace(era5_lats[0], era5_lats[-1], H)
        lons = np.linspace(era5_lons[0], era5_lons[-1], W)
        extent = [lons[0], lons[-1], lats[-1], lats[0]]

        # PRISM Truth (Top right)
        axes[0, col].imshow(prism, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                            extent=extent, aspect="auto")
        axes[0, col].set_title(f"PRISM Truth\n({prism.shape[0]}×{prism.shape[1]})",
                               fontsize=8, fontweight="bold", color="darkgreen")
        axes[0, col].tick_params(labelsize=6)

        # Error vs PRISM (Bottom right)
        diff = final["mean"] - prism
        rmse = np.sqrt((diff ** 2).mean())
        abs_max = max(abs(diff.min()), abs(diff.max()), 0.01)
        im_err = axes[1, col].imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                                     extent=extent, aspect="auto")
        axes[1, col].set_title(f"Error vs PRISM\nRMSE={rmse:.2f} mm",
                               fontsize=8, color="red")
        axes[1, col].tick_params(labelsize=6)

        # Dedicated colorbar for the error plot
        cbar_err = fig.colorbar(im_err, ax=axes[1, col], shrink=0.8)
        cbar_err.set_label("Error (mm)", fontsize=8)

    # --- 4. Add Shared Colorbars ---
    # Shared colorbar for the Mean/Truth row
    cbar_mean = fig.colorbar(im_mean, ax=axes[0, :], shrink=0.8, location='right')
    cbar_mean.set_label("Precipitation (mm/day)", fontsize=9, fontweight="bold")

    # Shared colorbar for the Spread plots
    spread_axes = axes[1, :n_steps] if has_prism else axes[1, :]
    cbar_std = fig.colorbar(im_std, ax=spread_axes, shrink=0.8, location='right')
    cbar_std.set_label("Spread (mm/day)", fontsize=9, fontweight="bold")

    axes[0, 0].set_ylabel("Ensemble Mean\n(mm/day)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Ensemble Spread\n(mm/day)", fontsize=10, fontweight="bold")

    cat_label = {"both_rain": "Both Rain", "era5_only": "ERA5 Only",
                 "prism_only": "PRISM Only"}.get(category, category)
    fig.suptitle(f"XGBoost Recursive Downscaling — {date_str} [{cat_label}]",
                 fontsize=13, fontweight="bold")

    cat_dir = out_dir / category
    cat_dir.mkdir(parents=True, exist_ok=True)
    save_path = cat_dir / f"progression_{date_str}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {category}/{save_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xgb-model", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--prism-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-ensemble", type=int, default=40)
    parser.add_argument("--n-days-per-cat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import xgboost as xgb

    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load XGBoost model
    log.info(f"Loading XGBoost: {args.xgb_model}")
    booster = xgb.Booster()
    booster.load_model(args.xgb_model)

    # Load ERA5
    era5_dir = Path(args.era5_dir)
    era5_lats = np.load(era5_dir / "era5_lats.npy")
    era5_lons = np.load(era5_dir / "era5_lons.npy")
    A_fine = np.load(era5_dir / "pair_A_fine.npy")  # meters/day
    A_dates = np.load(era5_dir / "valid_times_A.npy")
    A_test_idx = np.load(era5_dir / "test_indices_A.npy")

    # Build noise pools from training years
    B_coarse_up = np.load(era5_dir / "pair_B_coarse_up.npy") * M_TO_MM
    B_fine = np.load(era5_dir / "pair_B_fine.npy") * M_TO_MM
    B_dates = np.load(era5_dir / "valid_times_B.npy")

    yrs_B = np.array([int(str(d)[:4]) for d in B_dates])
    B_train_mask = np.isin(yrs_B, list(TRAIN_YEARS))
    B_train_idx = np.where(B_train_mask)[0]

    log.info(f"Building noise pools from {len(B_train_idx)} training days...")
    log_resid = []
    for t in B_train_idx:
        r = np.log1p(np.maximum(B_fine[t], 0)) - np.log1p(np.maximum(B_coarse_up[t], 0))
        log_resid.append(r.ravel())
    noise_pool_0 = np.concatenate(log_resid).astype(np.float32)
    log.info(f"  Noise pool: {noise_pool_0.size:,} samples, std={noise_pool_0.std():.4f}")

    noise_pools = {0: noise_pool_0}
    for step in range(1, 4):
        scale = 0.5 ** (step * 0.5)
        noise_pools[step] = (noise_pool_0 * scale).astype(np.float32)

    del B_coarse_up, B_fine, log_resid

    log.info(f"Test days: {len(A_test_idx)}")

    # Categorize
    log.info("Categorizing test days...")
    categories = categorize_days(A_fine, A_dates, A_test_idx,
                                 args.prism_dir, era5_lats, era5_lons)

    # Sample from each category
    rng = np.random.RandomState(args.seed)
    eval_plan = []

    for cat in ["both_rain", "era5_only", "prism_only"]:
        days = categories[cat]
        n_sample = min(args.n_days_per_cat, len(days))
        if n_sample == 0:
            log.warning(f"  No days for '{cat}'")
            continue
        sampled = rng.choice(len(days), size=n_sample, replace=False)
        for i in sampled:
            d = days[i]
            eval_plan.append((d["t_idx"], d["date"], cat,
                              d["era5_mean_mm"], d["prism_mean_mm"]))

    eval_plan.sort(key=lambda x: x[1])
    log.info(f"Total evaluation days: {len(eval_plan)}")

    # Final grid coords
    H_final = era5_lats.shape[0] * 16
    W_final = era5_lons.shape[0] * 16
    final_lats = np.linspace(era5_lats[0], era5_lats[-1], H_final)
    final_lons = np.linspace(era5_lons[0], era5_lons[-1], W_final)

    # Run evaluation
    per_day_results = []

    for day_i, (t_idx, date_str, category, era5_mm, prism_mm) in enumerate(eval_plan):
        year = int(date_str[:4])
        log.info(f"Day {day_i+1}/{len(eval_plan)}: {date_str} [{category}] "
                 f"(ERA5={era5_mm:.2f}, PRISM={prism_mm:.2f} mm/day)")

        era5_field_mm = A_fine[t_idx] * M_TO_MM

        t0 = time.time()
        ensemble, intermediates = recursive_downscale(
            era5_field_mm, year, booster, args.aef_dir, noise_pools,
            n_ensemble=args.n_ensemble, seed=args.seed + day_i,
        )
        log.info(f"  Downscaled in {time.time()-t0:.1f}s, shape={ensemble.shape}")

        H_ens, W_ens = ensemble.shape[1], ensemble.shape[2]

        # Load PRISM
        prism = None
        try:
            prism = load_prism_day(args.prism_dir, date_str, final_lats, final_lons)
        except Exception as e:
            log.warning(f"  PRISM failed: {e}")

        if prism is not None and prism.shape != (H_ens, W_ens):
            prism = zoom(prism, (H_ens / prism.shape[0], W_ens / prism.shape[1]),
                         order=1).astype(np.float32)
            prism = np.maximum(prism, 0)

        # Plot progression
        plot_progression(intermediates, prism, date_str, category,
                         era5_lats, era5_lons, out_dir)

        if prism is None:
            per_day_results.append({
                "date": date_str, "category": category,
                "era5_domain_mean_mm": era5_mm,
                "prism_domain_mean_mm": prism_mm,
                "metrics_available": False,
            })
            continue

        ens_mean = ensemble.mean(axis=0)
        ens_std = ensemble.std(axis=0)

        rmse_mean = float(np.sqrt(((ens_mean - prism) ** 2).mean()))

        bicubic = zoom(A_fine[t_idx],
                       (H_ens / A_fine[t_idx].shape[0], W_ens / A_fine[t_idx].shape[1]),
                       order=3).astype(np.float32)
        bicubic = np.maximum(bicubic, 0) * M_TO_MM
        rmse_bicubic = float(np.sqrt(((bicubic - prism) ** 2).mean()))

        mean_crps, _ = crps_grid(ensemble, prism)
        cov = ensemble_coverage(ensemble, prism)

        spread = float(ens_std.mean())
        skill = float(np.abs(ens_mean - prism).mean())
        ss = spread / (skill + 1e-10)

        day_metrics = {
            "date": date_str, "category": category,
            "era5_domain_mean_mm": era5_mm,
            "prism_domain_mean_mm": prism_mm,
            "metrics_available": True,
            "rmse_mean": rmse_mean,
            "rmse_bicubic": rmse_bicubic,
            "crps": float(mean_crps),
            "spread_skill": ss,
            **cov,
        }

        log.info(f"  RMSE(ens)={rmse_mean:.3f}, RMSE(bic)={rmse_bicubic:.3f}, "
                 f"CRPS={mean_crps:.3f}, S/S={ss:.3f}")
        log.info(f"  Coverage: " + ", ".join(f"{k}={v:.3f}" for k, v in cov.items()))

        per_day_results.append(day_metrics)

    # Aggregate
    aggregate = {}
    for cat in ["both_rain", "era5_only", "prism_only", "all"]:
        if cat == "all":
            cat_days = [d for d in per_day_results if d.get("metrics_available")]
        else:
            cat_days = [d for d in per_day_results
                        if d.get("metrics_available") and d["category"] == cat]

        if not cat_days:
            aggregate[cat] = {"n_days": 0}
            continue

        agg = {"n_days": len(cat_days)}
        for metric in ["rmse_mean", "rmse_bicubic", "crps", "spread_skill",
                        "coverage_50", "coverage_80", "coverage_90"]:
            vals = [d[metric] for d in cat_days if metric in d]
            if vals:
                agg[metric] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        aggregate[cat] = agg

    results = {
        "model": "XGBoost + AEF",
        "checkpoint": str(args.xgb_model),
        "n_ensemble": args.n_ensemble,
        "rain_threshold_mm": RAIN_THRESHOLD_MM,
        "aggregate": aggregate,
        "per_day": per_day_results,
    }

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Saved results to {results_path}")

    # Print summary
    log.info("=" * 60)
    log.info("AGGREGATE RESULTS (mm/day) — XGBoost")
    for cat in ["both_rain", "era5_only", "prism_only", "all"]:
        agg = aggregate[cat]
        if agg["n_days"] == 0:
            log.info(f"  [{cat}]: no days")
            continue
        log.info(f"  [{cat}] ({agg['n_days']} days):")
        for metric in ["rmse_mean", "rmse_bicubic", "crps", "spread_skill",
                        "coverage_50", "coverage_80", "coverage_90"]:
            if metric in agg:
                m = agg[metric]
                log.info(f"    {metric}: {m['mean']:.4f} ± {m['std']:.4f} "
                         f"[{m['min']:.4f}, {m['max']:.4f}]")

    log.info(f"Total time: {time.time()-t_start:.0f}s")
    log.info(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()