#!/usr/bin/env python3
"""
Recursive ensemble downscaling with XGBoost + AEF baseline,
validated against PRISM 800m observations.

CHANGES (vs. original):
  - Reduced feature set in build_pixel_features: now [coarse_precip(1) +
    aef_coarse(64) + aef_fine(64)] = 129 features. No lat/lon/doy.
  - Train/test split is strictly by calendar year:
      * Test days  = 2025 (PRISM 2025 must be available).
      * Noise pools / training residuals come from years 2017-2024.
    The script overrides any A_test_idx / B_train_idx in `era5-dir` by
    filtering on the date arrays.
  - Optionally accepts a second model (`--xgb-prism-head`) — the
    PRISM-supervised head trained by train_baselines_and_results.py.
    When provided, the rollout's final step (3.125→1.5625km) uses the
    PRISM head's prediction as the deterministic residual; ensemble
    noise is still injected.

Pipeline:
  1. For each 2025 test day, take ERA5 25km field.
  2. Recursively downscale 25→12.5→6.25→3.125→1.5625 km.
  3. At each step, generate N ensemble members via empirical noise
     sampled from training-year residuals.
  4. Compare ensemble vs PRISM 2025: CRPS, spread/skill, coverage, RMSE.

Usage:
    python -u validate_baseline_ensemble_on_prism.py \
        --xgb-model results/baselines/xgb_with_aef.json \
        --features-dir data/xgb_features \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_tif \
        --output-dir results/baselines_ensemble \
        [--xgb-prism-head results/baselines/xgb_prism_head.json]
"""

import argparse
import json
import logging
import time
from datetime import datetime
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


# ── Constants ────────────────────────────────────────────────────────────────

AEF_YEAR_OFFSET = 2017
N_ENSEMBLE = 100
D_AEF = 64
N_FEATURES = 1 + D_AEF + D_AEF  # 129

TRAIN_YEARS = set(range(2017, 2025))   # 2017..2024
TEST_YEARS  = {2025}

M_TO_MM = 1000.0  # ERA5 m/day → mm/day

# Recursive downscaling steps:
# (in_res_km, out_res_km, aef_coarse_km, aef_fine_km)
DOWNSCALE_STEPS = [
    (25.0,    12.5,    25.0,    12.5),
    (12.5,     6.25,   12.5,     6.25),
    ( 6.25,    3.125,   6.25,    3.125),
    ( 3.125,   1.5625,  3.125,   1.5625),
]


# ── Generic helpers ──────────────────────────────────────────────────────────

def load_aef_nc(nc_path):
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32).transpose(1, 2, 0)
    ds.close()
    return arr


def find_aef_file(aef_dir, t_idx, res_km):
    candidates = [
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{res_km}km.nc",
        aef_dir / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"aef_illinois_{res_km}km.nc",
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
    """
    Build XGBoost feature matrix for all pixels in a grid.
    Features per pixel: [log1p(coarse_precip)(1), aef_coarse(64), aef_fine(64)]

    Args:
        field_2d_mm: (H, W) coarse-upsampled precipitation in mm/day
        aef_coarse:  (H, W, 64) AEF at coarse resolution (already on this grid)
        aef_fine:    (H, W, 64) AEF at fine resolution (already on this grid)

    Returns:
        X: (H*W, 129) feature matrix
    """
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


# ── PRISM loading ────────────────────────────────────────────────────────────

def load_prism_day(prism_dir, date, target_lats, target_lons):
    """
    PRISM is mm/day; we keep mm/day throughout to match the model's units.
    (The original script converted PRISM mm → m to match an old m/day pipeline.
    Here, models are trained in mm/day, so we leave PRISM in mm/day.)

    Returns (H, W) float32 mm/day or None.
    """
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

    data[data < -900] = 0  # nodata

    # Coerce target lons to negative (PRISM is in negative-west convention)
    target_lons_neg = -np.abs(np.where(target_lons > 180,
                                       target_lons - 360, target_lons))

    lat_min = target_lats.min() - 0.1
    lat_max = target_lats.max() + 0.1
    lon_min = target_lons_neg.min() - 0.1
    lon_max = target_lons_neg.max() + 0.1

    lat_mask = (prism_lats >= lat_min) & (prism_lats <= lat_max)
    lon_mask = (prism_lons >= lon_min) & (prism_lons <= lon_max)
    if lat_mask.sum() == 0 or lon_mask.sum() == 0:
        log.warning(f"  No PRISM pixels in IL extent for {date_str}")
        return None

    li = np.where(lat_mask)[0]
    wi = np.where(lon_mask)[0]
    cropped = data[li[0]:li[-1] + 1, wi[0]:wi[-1] + 1]

    H_t, W_t = len(target_lats), len(target_lons)
    regrid = zoom(cropped, (H_t / cropped.shape[0], W_t / cropped.shape[1]),
                  order=1).astype(np.float32)
    return np.maximum(regrid, 0)  # mm/day


# ── CRPS / coverage ──────────────────────────────────────────────────────────

def crps_ensemble(ensemble, observation):
    N = len(ensemble)
    mae = np.abs(ensemble - observation).mean()
    sorted_ens = np.sort(ensemble)
    diff_sum = 0.0
    for i in range(N):
        diff_sum += (2 * i - N) * sorted_ens[i]
    spread = 2 * diff_sum / (N * N)
    return mae - 0.5 * spread


def crps_grid(ensemble_grids, observation_grid):
    H, W = observation_grid.shape
    crps_values = np.empty((H, W))
    for i in range(H):
        for j in range(W):
            crps_values[i, j] = crps_ensemble(
                ensemble_grids[:, i, j],
                observation_grid[i, j],
            )
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


# ── Recursive ensemble downscaling ───────────────────────────────────────────

def recursive_downscale(
    era5_25km_mm,        # (H25, W25) mm/day
    year,
    booster_main,        # XGBoost AEF model (used for steps 0..2)
    aef_dir,
    training_residuals,  # dict: step_idx → flat residual pool (log-space)
    booster_prism_head=None,   # optional: PRISM-supervised head for step 3
    n_ensemble=N_ENSEMBLE,
    seed=None,
):
    """
    Returns:
        ensemble_final: (n_ensemble, H_final, W_final) at 1.5625km, mm/day
        intermediate:   dict of per-step ensemble stats
    """
    import xgboost as xgb

    rng = np.random.RandomState(seed)
    t_idx = year - AEF_YEAR_OFFSET

    current_fields = np.stack([era5_25km_mm.astype(np.float32)] * n_ensemble)
    intermediate = {}

    for step_idx, (in_res, out_res, aef_c_res, aef_f_res) in enumerate(DOWNSCALE_STEPS):
        H_in, W_in = current_fields.shape[1], current_fields.shape[2]
        H_out, W_out = H_in * 2, W_in * 2

        # Load AEF at output grid
        try:
            aef_c = load_aef_nc(find_aef_file(Path(aef_dir), t_idx, aef_c_res))
            aef_f = load_aef_nc(find_aef_file(Path(aef_dir), t_idx, aef_f_res))
        except FileNotFoundError:
            log.warning(f"  AEF not found for step {step_idx}, using zeros")
            aef_c = np.zeros((H_out, W_out, D_AEF), dtype=np.float32)
            aef_f = np.zeros((H_out, W_out, D_AEF), dtype=np.float32)

        aef_c_resized = resize_aef_to_grid(aef_c, H_out, W_out)
        aef_f_resized = resize_aef_to_grid(aef_f, H_out, W_out)

        noise_pool = training_residuals.get(step_idx, np.zeros(1000, dtype=np.float32))

        # Pick which booster to use at this step
        booster = booster_prism_head if (step_idx == 3 and booster_prism_head is not None) \
            else booster_main

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

        intermediate[f"step_{step_idx}_{out_res}km"] = {
            "mean": current_fields.mean(axis=0),
            "std":  current_fields.std(axis=0),
            "shape": current_fields.shape,
        }

    return current_fields, intermediate


# ── Year filter ──────────────────────────────────────────────────────────────

def filter_indices_by_year(dates, indices_or_none, years):
    """
    Return indices into `dates` whose calendar year is in `years`.
    If `indices_or_none` is given, restrict to that subset first.
    """
    yrs_arr = np.array([int(str(d)[:4]) for d in dates])
    if indices_or_none is None:
        return np.where(np.isin(yrs_arr, list(years)))[0]
    sub = np.array(indices_or_none, dtype=int)
    return sub[np.isin(yrs_arr[sub], list(years))]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Recursive ensemble downscaling + PRISM 2025 validation."
    )
    parser.add_argument("--xgb-model", required=True,
                        help="Path to XGBoost AEF model (.json)")
    parser.add_argument("--xgb-prism-head", default=None,
                        help="Optional path to PRISM-supervised head (.json)."
                             " When given, used at the final 3.125→1.5625km step.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--prism-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-ensemble", type=int, default=100)
    parser.add_argument("--n-days", type=int, default=10,
                        help="How many 2025 days to evaluate.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import xgboost as xgb

    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────
    log.info(f"Loading XGBoost AEF model: {args.xgb_model}")
    model_main = xgb.Booster()
    model_main.load_model(args.xgb_model)

    model_prism = None
    if args.xgb_prism_head is not None:
        log.info(f"Loading PRISM-supervised head: {args.xgb_prism_head}")
        model_prism = xgb.Booster()
        model_prism.load_model(args.xgb_prism_head)

    # ── Load ERA5 data ───────────────────────────────────────────────────
    era5_dir = Path(args.era5_dir)
    era5_lats = np.load(era5_dir / "era5_lats.npy")
    era5_lons = np.load(era5_dir / "era5_lons.npy")

    # ERA5 fields (m/day → mm/day)
    A_fine = np.load(era5_dir / "pair_A_fine.npy") * M_TO_MM   # 25km native
    A_dates = np.load(era5_dir / "valid_times_A.npy")

    B_coarse_up = np.load(era5_dir / "pair_B_coarse_up.npy") * M_TO_MM
    B_fine      = np.load(era5_dir / "pair_B_fine.npy") * M_TO_MM
    B_dates     = np.load(era5_dir / "valid_times_B.npy")

    # ── Year-based splits (override any *_idx files on disk) ─────────────
    A_test_idx_2025 = filter_indices_by_year(A_dates, None, TEST_YEARS)
    B_train_idx_lt2025 = filter_indices_by_year(B_dates, None, TRAIN_YEARS)

    log.info(f"Pair A 2025 test days available: {len(A_test_idx_2025)}")
    log.info(f"Pair B 2017-2024 training days for noise pool: {len(B_train_idx_lt2025)}")

    if len(A_test_idx_2025) == 0:
        raise RuntimeError("No 2025 days found in valid_times_A.npy — "
                           "cannot evaluate on PRISM 2025.")

    # ── Build noise pools from 2017-2024 training residuals ──────────────
    log.info("Building noise pools from 2017-2024 training residuals...")
    log_resid_B = []
    for t in B_train_idx_lt2025:
        r = (np.log1p(np.maximum(B_fine[t], 0))
             - np.log1p(np.maximum(B_coarse_up[t], 0)))
        log_resid_B.append(r.ravel())
    if not log_resid_B:
        raise RuntimeError("No 2017-2024 Pair B training residuals available "
                           "to build noise pool.")
    noise_pool_0 = np.concatenate(log_resid_B).astype(np.float32)
    log.info(f"  Step 0 noise pool: {noise_pool_0.size:,} samples, "
             f"std={noise_pool_0.std():.6f}")

    # Steps 1..3: scale step-0 pool as a heuristic for finer resolutions
    training_residuals = {0: noise_pool_0}
    for step in range(1, 4):
        scale = 0.5 ** (step * 0.5)   # ~0.71, 0.50, 0.35
        training_residuals[step] = (noise_pool_0 * scale).astype(np.float32)
        log.info(f"  Step {step} noise: scaled by {scale:.2f}, "
                 f"std={training_residuals[step].std():.6f}")

    # Free unneeded big arrays
    del B_coarse_up, B_fine, log_resid_B

    # ── Choose 2025 test days ────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)
    n_eval = min(args.n_days, len(A_test_idx_2025))
    eval_indices = rng.choice(A_test_idx_2025, size=n_eval, replace=False)
    eval_indices.sort()

    # ── Final-grid coords (1.5625 km) ────────────────────────────────────
    H_final = era5_lats.shape[0] * 16
    W_final = era5_lons.shape[0] * 16
    final_lats = np.linspace(era5_lats[0], era5_lats[-1], H_final)
    final_lons = np.linspace(era5_lons[0], era5_lons[-1], W_final)
    log.info(f"Final grid: ({H_final}, {W_final}) at ~1.5625km")

    # ── Inference + PRISM 2025 validation ────────────────────────────────
    all_crps = []
    all_rmse_mean = []
    all_rmse_bicubic = []
    all_coverage = {f"coverage_{l}": [] for l in [50, 80, 90]}
    all_spread_skill = []
    per_day_records = []

    for day_i, t_idx in enumerate(eval_indices):
        date = A_dates[t_idx]
        date_str = str(date)[:10]
        year = int(date_str[:4])
        month = int(date_str[5:7])
        day = int(date_str[8:10])
        doy = (datetime(year, month, day) - datetime(year, 1, 1)).days + 1

        log.info(f"Day {day_i + 1}/{n_eval}: {date_str} (doy={doy})")
        era5_field_mm = A_fine[t_idx]   # (H25, W25) mm/day

        # ── Recursive ensemble ──────────────────────────────────────────
        t0 = time.time()
        ensemble, intermediate = recursive_downscale(
            era5_field_mm, year,
            model_main, args.aef_dir,
            training_residuals,
            booster_prism_head=model_prism,
            n_ensemble=args.n_ensemble,
            seed=args.seed + day_i,
        )
        H_ens, W_ens = ensemble.shape[1], ensemble.shape[2]
        log.info(f"  Downscaled in {time.time() - t0:.1f}s, "
                 f"ensemble shape: {ensemble.shape}")

        # ── PRISM truth (mm/day) ─────────────────────────────────────────
        try:
            prism = load_prism_day(args.prism_dir, date, final_lats, final_lons)
        except Exception as e:
            log.warning(f"  PRISM load failed: {e}")
            prism = None

        if prism is None:
            log.warning(f"  No PRISM data for {date_str}, skipping.")
            continue

        if prism.shape != (H_ens, W_ens):
            prism = zoom(prism, (H_ens / prism.shape[0], W_ens / prism.shape[1]),
                         order=1).astype(np.float32)
            prism = np.maximum(prism, 0)

        log.info(f"  PRISM shape: {prism.shape}, "
                 f"range=[{prism.min():.4f}, {prism.max():.4f}] mm/day")

        # ── Metrics ──────────────────────────────────────────────────────
        ens_mean = ensemble.mean(axis=0)
        ens_std  = ensemble.std(axis=0)

        rmse_mean = float(np.sqrt(((ens_mean - prism) ** 2).mean()))
        all_rmse_mean.append(rmse_mean)

        bicubic_final = zoom(era5_field_mm,
                             (H_ens / era5_field_mm.shape[0],
                              W_ens / era5_field_mm.shape[1]),
                             order=3).astype(np.float32)
        bicubic_final = np.maximum(bicubic_final, 0)
        rmse_bicubic = float(np.sqrt(((bicubic_final - prism) ** 2).mean()))
        all_rmse_bicubic.append(rmse_bicubic)

        mean_crps, crps_map = crps_grid(ensemble, prism)
        all_crps.append(float(mean_crps))

        cov = ensemble_coverage(ensemble, prism)
        for k, v in cov.items():
            all_coverage[k].append(v)

        spread = float(ens_std.mean())
        skill  = float(np.abs(ens_mean - prism).mean())
        ssr    = spread / (skill + 1e-10)
        all_spread_skill.append(ssr)

        log.info(f"  RMSE(ens mean)={rmse_mean:.4f} mm, "
                 f"RMSE(bicubic)={rmse_bicubic:.4f} mm")
        log.info(f"  CRPS={mean_crps:.4f}, Spread/Skill={ssr:.3f}")
        log.info(f"  Coverage: " + ", ".join(f"{k}={v:.3f}" for k, v in cov.items()))

        per_day_records.append({
            "date": date_str,
            "rmse_mean": rmse_mean,
            "rmse_bicubic": rmse_bicubic,
            "crps": float(mean_crps),
            "spread_skill_ratio": ssr,
            **{k: float(v) for k, v in cov.items()},
        })

        # ── Per-day plot for first 5 days ────────────────────────────────
        if day_i < 5:
            fig, axes = plt.subplots(1, 5, figsize=(22, 4), constrained_layout=True)
            vmin = float(min(prism.min(), ens_mean.min(), bicubic_final.min()))
            vmax = float(max(prism.max(), ens_mean.max(), bicubic_final.max()))
            extent = [final_lons[0], final_lons[-1], final_lats[-1], final_lats[0]]

            axes[0].imshow(prism, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                           extent=extent, aspect="auto")
            axes[0].set_title("PRISM 2025 (truth)", fontsize=9, fontweight="bold")

            axes[1].imshow(ens_mean, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                           extent=extent, aspect="auto")
            axes[1].set_title(f"Ensemble Mean\nRMSE={rmse_mean:.4f}", fontsize=9)

            axes[2].imshow(bicubic_final, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                           extent=extent, aspect="auto")
            axes[2].set_title(f"Bicubic\nRMSE={rmse_bicubic:.4f}", fontsize=9)

            axes[3].imshow(ens_std, cmap="Oranges", extent=extent, aspect="auto")
            axes[3].set_title(f"Ensemble Spread\nmean={spread:.4f}", fontsize=9)

            axes[4].imshow(crps_map, cmap="Reds", extent=extent, aspect="auto")
            axes[4].set_title(f"CRPS Map\nmean={mean_crps:.4f}", fontsize=9)

            for ax in axes:
                ax.set_xlabel("Lon")
            axes[0].set_ylabel("Lat")
            fig.suptitle(f"Ensemble Downscaling — {date_str} (mm/day)", fontsize=12)
            fig.savefig(out_dir / f"ensemble_{date_str}.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info(f"  Saved ensemble_{date_str}.png")

    # ── Aggregate ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("AGGREGATE RESULTS (PRISM 2025)")
    if not all_crps:
        log.error("No 2025 days yielded valid metrics.")
        return

    results = {
        "n_days_evaluated": len(all_crps),
        "test_years": sorted(TEST_YEARS),
        "train_years_for_noise": sorted(TRAIN_YEARS),
        "used_prism_head": model_prism is not None,
        "rmse_ensemble_mean": {
            "mean": float(np.mean(all_rmse_mean)),
            "std":  float(np.std(all_rmse_mean)),
            "units": "mm/day",
        },
        "rmse_bicubic": {
            "mean": float(np.mean(all_rmse_bicubic)),
            "std":  float(np.std(all_rmse_bicubic)),
            "units": "mm/day",
        },
        "crps": {
            "mean": float(np.mean(all_crps)),
            "std":  float(np.std(all_crps)),
            "units": "mm/day",
        },
        "spread_skill_ratio": {
            "mean": float(np.mean(all_spread_skill)),
            "std":  float(np.std(all_spread_skill)),
            "note": "ideal is ~1.0",
        },
        "coverage": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in all_coverage.items() if v
        },
        "per_day": per_day_records,
    }

    log.info(f"  RMSE (ens mean): {results['rmse_ensemble_mean']['mean']:.4f} "
             f"± {results['rmse_ensemble_mean']['std']:.4f} mm/day")
    log.info(f"  RMSE (bicubic):  {results['rmse_bicubic']['mean']:.4f} "
             f"± {results['rmse_bicubic']['std']:.4f} mm/day")
    log.info(f"  CRPS:            {results['crps']['mean']:.4f} "
             f"± {results['crps']['std']:.4f} mm/day")
    log.info(f"  Spread/Skill:    {results['spread_skill_ratio']['mean']:.3f} "
             f"± {results['spread_skill_ratio']['std']:.3f}")
    for k, v in results["coverage"].items():
        log.info(f"  {k}: {v['mean']:.3f} ± {v['std']:.3f}")

    with open(out_dir / "ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nTotal time: {time.time() - t_start:.0f}s")
    log.info(f"Results: {out_dir}")


if __name__ == "__main__":
    main()