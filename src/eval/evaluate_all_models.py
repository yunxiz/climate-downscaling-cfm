#!/usr/bin/env python3
"""
Combined evaluation: all models on the same test examples.

Aligned with:
  - The OT-CFM model (model_ode.py / train_phase3.py): Phase 3 is supervised
    against PRISM at 1.5625km via 4-step recursive rollout.
  - The new XGBoost pipeline (build_xgb_data.py / train_baselines_and_results.py):
      * 129-feature layout: [coarse_precip, aef_coarse(64), aef_fine(64)]
      * mm/day units throughout
      * Year-based split: train 2017-2024, test 2025
      * Optional PRISM-supervised "Pair C" head (xgb_prism_head.json)

Two evaluation modes:
  1. Pair A & Pair B (50→25 km, 25→12.5 km) — short-range residual quality.
     The CFM model evaluated here is its single-step Phase 1/2 behavior.
     For Phase 3 (rollout) checkpoints this is still informative as the
     finest-step capability hasn't been compromised by Phase 3 fine-tuning,
     but the headline number is the PRISM rollout below.
  2. PRISM 2025 rollout (25 km → 1.5625 km, 4 recursive steps) — final
     downscaling skill compared against PRISM 800m coarsened to 1.5625km.

Produces side-by-side comparisons:
  Ground Truth | Bicubic | XGBoost + AEF | (XGB + PRISM head) | OT-CFM Ensemble Mean

Usage:
    python -u evaluate_all_models.py \
        --checkpoint checkpoints/cfm_ode_v3/best_model_phase3.pt \
        --xgb-model results/baselines/xgb_with_aef.json \
        --xgb-prism-head results/baselines/xgb_prism_head.json \
        --features-dir data/xgb_features \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_tif_2025 \
        --output-dir results/combined
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
import torch
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
N_FEATURES = 1 + D_AEF + D_AEF  # 129

# Recursive downscaling steps for PRISM 2025 rollout
DOWNSCALE_STEPS = [
    (25.0,    12.5,    25.0,    12.5),
    (12.5,     6.25,   12.5,     6.25),
    ( 6.25,    3.125,   6.25,    3.125),
    ( 3.125,   1.5625,  3.125,   1.5625),
]


# ── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt((residuals ** 2).mean())
    mae = np.abs(residuals).mean()
    ss_res = (residuals ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    bias = residuals.mean()
    return {"rmse": float(rmse), "mae": float(mae),
            "r2": float(r2), "bias": float(bias)}


def crps_per_pixel(ensemble, observation):
    N = len(ensemble)
    mae = np.abs(ensemble - observation).mean()
    sorted_ens = np.sort(ensemble)
    diff_sum = sum((2 * i - N) * sorted_ens[i] for i in range(N))
    spread = 2 * diff_sum / (N * N)
    return mae - 0.5 * spread


def compute_probabilistic_metrics(ensemble_grids, gt_grids):
    """ensemble_grids: (T, K, H, W); gt_grids: (T, H, W)."""
    T, K, H, W = ensemble_grids.shape
    all_crps = []
    for t in range(T):
        for i in range(H):
            for j in range(W):
                all_crps.append(crps_per_pixel(
                    ensemble_grids[t, :, i, j], gt_grids[t, i, j]))
    mean_crps = float(np.mean(all_crps))

    coverage = {}
    for level in [0.5, 0.8, 0.9]:
        alpha = (1 - level) / 2
        lo = np.quantile(ensemble_grids, alpha, axis=1)
        hi = np.quantile(ensemble_grids, 1 - alpha, axis=1)
        covered = ((gt_grids >= lo) & (gt_grids <= hi)).mean()
        coverage[f"coverage_{int(level*100)}"] = float(covered)

    ens_mean = ensemble_grids.mean(axis=1)
    ens_std = ensemble_grids.std(axis=1)
    spread = float(ens_std.mean())
    skill = float(np.abs(ens_mean - gt_grids).mean())
    ss_ratio = spread / (skill + 1e-10)

    return {"crps": mean_crps, "spread_skill": float(ss_ratio), **coverage}


# ── XGBoost helpers ──────────────────────────────────────────────────────────

def log_transform_residuals(coarse_up_mm, fine_mm):
    return np.log1p(np.maximum(fine_mm, 0)) - np.log1p(np.maximum(coarse_up_mm, 0))


def inverse_log_residual(coarse_up_mm, log_residual):
    """coarse_up and output both in mm/day."""
    log_coarse = np.log1p(np.maximum(coarse_up_mm, 0))
    return np.maximum(np.expm1(log_coarse + log_residual), 0)


def run_xgboost_predictions(xgb_model, X_test, coarse_up_test_mm, n_test, H, W):
    """
    X_test: (n_test*H*W, 129) — already in mm/day in the precip column.
    coarse_up_test_mm: (n_test, H, W) in mm/day.
    Returns: predicted fine field in mm/day, shape (n_test, H, W).
    """
    import xgboost as xgb

    # Apply log1p to precipitation column (col 0). Build features mirror this
    # transform at training time.
    X_test_log = X_test.copy()
    X_test_log[:, 0] = np.log1p(np.maximum(X_test_log[:, 0], 0))

    pred_log = xgb_model.predict(xgb.DMatrix(X_test_log))
    pred_grids = pred_log.reshape(n_test, H, W)

    fine_grids_mm = np.array([
        inverse_log_residual(coarse_up_test_mm[i], pred_grids[i])
        for i in range(n_test)
    ], dtype=np.float32)
    return fine_grids_mm


# ── AEF helpers ──────────────────────────────────────────────────────────────

def load_aef_nc(nc_path):
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32).transpose(1, 2, 0)
    ds.close()
    return arr


def find_aef_file(aef_dir, t_idx, res_km):
    aef_dir = Path(aef_dir)   # ADD THIS LINE

    res_str = str(int(res_km)) if res_km == int(res_km) else str(res_km)
    candidates = [
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{res_str}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{res_str}km.nc",
        aef_dir / f"aef_illinois_t{t_idx}_{res_str}km.nc",
        aef_dir / f"aef_illinois_{res_str}km.nc",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def resize_aef_grid_np(aef_hwd, target_h, target_w):
    D = aef_hwd.shape[2]
    return np.stack([
        zoom(aef_hwd[:, :, d],
             (target_h / aef_hwd.shape[0], target_w / aef_hwd.shape[1]),
             order=1)
        for d in range(D)
    ], axis=-1).astype(np.float32)


def load_aef_as_tensor(aef_dir, t_idx, res_km, target_h, target_w, device):
    p = find_aef_file(aef_dir, t_idx, res_km)
    if p is None:
        log.warning(f"  AEF not found: t{t_idx}, {res_km}km — using zeros")
        return torch.zeros(1, D_AEF, target_h, target_w, device=device)
    aef = load_aef_nc(p)
    aef_resized = resize_aef_grid_np(aef, target_h, target_w)  # (H, W, D)
    return torch.from_numpy(aef_resized.transpose(2, 0, 1)).unsqueeze(0).to(device)


def load_aef_as_np(aef_dir, t_idx, res_km, target_h, target_w):
    p = find_aef_file(aef_dir, t_idx, res_km)
    if p is None:
        return np.zeros((target_h, target_w, D_AEF), dtype=np.float32)
    return resize_aef_grid_np(load_aef_nc(p), target_h, target_w)


def build_pixel_features_xgb(field_2d_mm, aef_coarse, aef_fine):
    """129-feature row matrix used for XGBoost rollout."""
    H, W = field_2d_mm.shape
    D = aef_coarse.shape[2]
    n = H * W
    X = np.empty((n, 1 + D + D), dtype=np.float32)
    col = 0
    X[:, col] = np.log1p(np.maximum(field_2d_mm, 0)).ravel(); col += 1
    X[:, col:col + D] = aef_coarse.reshape(-1, D); col += D
    X[:, col:col + D] = aef_fine.reshape(-1, D)
    return X


# ── PRISM loader ─────────────────────────────────────────────────────────────

def load_prism_day(prism_dir, date, target_lats, target_lons):
    """Returns mm/day or None."""
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


# ── ODE integration ──────────────────────────────────────────────────────────

@torch.no_grad()
def ode_integrate(model, coarse_up, alpha_c, alpha_f,
                  n_ensemble, n_steps=50, device="cuda"):
    """Pure deterministic OT-CFM Euler integration. Ensemble = different x0."""
    B = n_ensemble
    _, _, H, W = coarse_up.shape

    coarse_up_b = coarse_up.expand(B, -1, -1, -1)
    alpha_c_b   = alpha_c.expand(B, -1, -1, -1)
    alpha_f_b   = alpha_f.expand(B, -1, -1, -1)

    x = torch.randn(B, 1, H, W, device=device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_val = step * dt
        t_batch = torch.full((B,), t_val, device=device)
        x_input = torch.cat([coarse_up_b, x], dim=1)
        v = model(t_batch, x_input, alpha_c_b, alpha_f_b)
        x = x + v * dt
    return x  # residual in mm/day (model trained on mm/day)


@torch.no_grad()
def predict_cfm_single_step(
    model, coarse_up_mm, alpha_c, alpha_f,
    n_ensemble, n_steps, device, ensemble_batch=None,
):
    """
    Single-step CFM downscaling. coarse_up_mm is np.ndarray in mm/day.
    Returns ensemble (K, H, W) and ensemble mean (H, W), both mm/day.
    """
    H, W = coarse_up_mm.shape
    coarse_t = torch.from_numpy(coarse_up_mm).float().reshape(1, 1, H, W).to(device)

    if ensemble_batch is None:
        ensemble_batch = n_ensemble

    members = []
    for batch_start in range(0, n_ensemble, ensemble_batch):
        b = min(ensemble_batch, n_ensemble - batch_start)
        residuals = ode_integrate(model, coarse_t, alpha_c, alpha_f,
                                  n_ensemble=b, n_steps=n_steps, device=device)
        fine = (coarse_t + residuals).squeeze(1).cpu().numpy()
        fine = np.maximum(fine, 0)
        members.append(fine)
    ensemble = np.concatenate(members, axis=0)
    return ensemble, ensemble.mean(axis=0)


# ── CFM recursive rollout (25 → 1.5625 km) ──────────────────────────────────

@torch.no_grad()
def cfm_recursive_rollout(
    model, era5_25km_mm, year, aef_dir, device,
    n_ensemble=1, n_steps=50,
):
    aef_dir = Path(aef_dir)
    t_idx = year - AEF_YEAR_OFFSET

    H_in, W_in = era5_25km_mm.shape
    current = np.broadcast_to(
        era5_25km_mm.astype(np.float32),
        (n_ensemble, H_in, W_in)
    ).copy()

    MAX_CFM_PIXELS = 184 * 136  # avoid final 368x272 attention OOM

    for step_idx, (in_res, out_res, aef_c_res, aef_f_res) in enumerate(DOWNSCALE_STEPS):
        H_cur, W_cur = current.shape[1], current.shape[2]
        H_out, W_out = H_cur * 2, W_cur * 2

        if H_out * W_out > MAX_CFM_PIXELS:
            log.warning(
                f"Skipping CFM step {step_idx}: {in_res}->{out_res} km "
                f"grid {H_out}x{W_out} too large for global attention. "
                "Using bicubic upsample for this step."
            )

            new = np.empty((n_ensemble, H_out, W_out), dtype=np.float32)
            for k in range(n_ensemble):
                coarse_up = zoom(
                    current[k],
                    (H_out / H_cur, W_out / W_cur),
                    order=3
                ).astype(np.float32)
                new[k] = np.maximum(coarse_up, 0)

            current = new
            continue

        alpha_c = load_aef_as_tensor(aef_dir, t_idx, aef_c_res, H_out, W_out, device)
        alpha_f = load_aef_as_tensor(aef_dir, t_idx, aef_f_res, H_out, W_out, device)

        new = np.empty((n_ensemble, H_out, W_out), dtype=np.float32)

        for k in range(n_ensemble):
            coarse_up = zoom(
                current[k],
                (H_out / H_cur, W_out / W_cur),
                order=3
            ).astype(np.float32)
            coarse_up = np.maximum(coarse_up, 0)

            _, mean = predict_cfm_single_step(
                model,
                coarse_up,
                alpha_c,
                alpha_f,
                n_ensemble=1,
                n_steps=n_steps,
                device=device,
            )
            new[k] = mean

        current = new

    return current, current.mean(axis=0)

# ── XGBoost recursive rollout (25 → 1.5625 km) ──────────────────────────────

def xgb_recursive_rollout(
    booster_main, era5_25km_mm, year, aef_dir,
    booster_prism_head=None,
):
    """
    Mirror of validate_baseline_ensemble_on_prism's recursive_downscale, but
    deterministic (no noise injection). Returns final (H_final, W_final)
    in mm/day.
    """
    import xgboost as xgb

    aef_dir = Path(aef_dir)
    t_idx = year - AEF_YEAR_OFFSET
    current = era5_25km_mm.astype(np.float32)

    for step_idx, (in_res, out_res, aef_c_res, aef_f_res) in enumerate(DOWNSCALE_STEPS):
        H_in, W_in = current.shape
        H_out, W_out = H_in * 2, W_in * 2

        coarse_up = zoom(current, (H_out / H_in, W_out / W_in),
                         order=3).astype(np.float32)
        coarse_up = np.maximum(coarse_up, 0)

        aef_c = load_aef_as_np(aef_dir, t_idx, aef_c_res, H_out, W_out)
        aef_f = load_aef_as_np(aef_dir, t_idx, aef_f_res, H_out, W_out)

        booster = booster_prism_head if (step_idx == 3 and booster_prism_head is not None) \
            else booster_main

        X = build_pixel_features_xgb(coarse_up, aef_c, aef_f)
        log_resid = booster.predict(xgb.DMatrix(X)).reshape(H_out, W_out)
        current = inverse_log_residual(coarse_up, log_resid)

    return current


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(
    fine_true_grids_mm, predictions_dict_mm, lats, lons,
    pair_name, output_dir, n_examples=5, seed=42,
):
    """All inputs already in mm/day."""
    rng = np.random.RandomState(seed)
    n_times = fine_true_grids_mm.shape[0]
    if n_times == 0:
        log.warning(f"  {pair_name}: no test days to plot")
        return

    domain_means = np.array([fine_true_grids_mm[t].mean() for t in range(n_times)])
    rainy_mask = domain_means > np.percentile(domain_means, 80)
    rainy_indices = np.where(rainy_mask)[0]

    if len(rainy_indices) >= n_examples:
        example_indices = rng.choice(rainy_indices, size=n_examples, replace=False)
    else:
        example_indices = rng.choice(n_times, size=min(n_examples, n_times), replace=False)
    example_indices.sort()

    model_names = list(predictions_dict_mm.keys())
    n_models = len(model_names)
    extent = [lons[0], lons[-1], lats[-1], lats[0]]

    for ex_i, t_idx in enumerate(example_indices):
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4),
                                 constrained_layout=True)
        gt = fine_true_grids_mm[t_idx]
        all_vals = [gt] + [predictions_dict_mm[m][t_idx] for m in model_names]
        vmin = min(v.min() for v in all_vals)
        vmax = max(v.max() for v in all_vals)

        im = axes[0].imshow(gt, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                            extent=extent, aspect="auto")
        axes[0].set_title("Ground Truth", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        for j, m in enumerate(model_names):
            pred = predictions_dict_mm[m][t_idx]
            axes[j + 1].imshow(pred, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                               extent=extent, aspect="auto")
            rmse = np.sqrt(((gt - pred) ** 2).mean())
            axes[j + 1].set_title(f"{m}\nRMSE={rmse:.4f}", fontsize=9)
            axes[j + 1].set_xlabel("Longitude")

        fig.colorbar(im, ax=axes, shrink=0.8, label="Precipitation (mm/day)")
        fig.suptitle(f"{pair_name} — Test Example {ex_i + 1}", fontsize=12)
        fig.savefig(output_dir / f"{pair_name}_example_{ex_i + 1}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved {pair_name}_example_{ex_i + 1}.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined evaluation: Bicubic vs XGBoost(+AEF/+PRISM head) "
                    "vs OT-CFM on Pair A/B and on PRISM 2025 rollout."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="OT-CFM checkpoint (.pt). Phase 3 recommended.")
    parser.add_argument("--n-ensemble", type=int, default=20)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--ensemble-batch", type=int, default=None)
    parser.add_argument("--use-ema", action="store_true", default=True)

    parser.add_argument("--xgb-model", required=True,
                        help="XGBoost AEF model (.json)")
    parser.add_argument("--xgb-prism-head", default=None,
                        help="Optional: XGBoost PRISM-supervised head (.json). "
                             "Used at step 3 of the rollout.")

    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--prism-dir", default=None,
                        help="PRISM dir for the 2025 rollout evaluation. "
                             "If omitted, only Pair A/B metrics are reported.")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--prism-n-days", type=int, default=20,
                        help="Number of 2025 days for PRISM rollout eval.")
    parser.add_argument("--cfm-rollout-ensemble", type=int, default=1,
                        help="CFM ensemble size for the 2025 PRISM rollout. "
                             "1 = deterministic mean only (fast).")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--base-channels", type=int, default=48)
    parser.add_argument("--channel-mult", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--aef-dim", type=int, default=64)
    parser.add_argument("--time-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import xgboost as xgb

    t_start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load OT-CFM model ────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from model_ode import DownscalingUNet

    log.info(f"Loading OT-CFM: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    cfm_model = DownscalingUNet(
        in_channels=2, out_channels=1,
        base_channels=args.base_channels,
        channel_mult=tuple(args.channel_mult),
        aef_dim=args.aef_dim,
        time_dim=args.time_dim,
        num_heads=args.num_heads,
    ).to(device)

    if args.use_ema and "ema_state_dict" in ckpt:
        cfm_model.load_state_dict(ckpt["ema_state_dict"])
        log.info("  Loaded EMA weights")
    else:
        cfm_model.load_state_dict(ckpt["model_state_dict"])
        log.info("  Loaded model weights")
    cfm_model.eval()
    n_params = sum(p.numel() for p in cfm_model.parameters())
    log.info(f"  Parameters: {n_params:,}")

    # ── Load XGBoost models ──────────────────────────────────────────────
    log.info(f"Loading XGBoost AEF model: {args.xgb_model}")
    xgb_aef = xgb.Booster()
    xgb_aef.load_model(args.xgb_model)

    xgb_prism = None
    if args.xgb_prism_head is not None:
        log.info(f"Loading XGBoost PRISM head: {args.xgb_prism_head}")
        xgb_prism = xgb.Booster()
        xgb_prism.load_model(args.xgb_prism_head)

    # ── Load data ────────────────────────────────────────────────────────
    log.info("Loading data...")
    era5_dir = Path(args.era5_dir)
    feat_dir = Path(args.features_dir)
    aef_dir = Path(args.aef_dir)

    # ERA5 fields: stored as m/day on disk → convert to mm/day for everything
    A_coarse_up = np.load(era5_dir / "pair_A_coarse_up.npy") * M_TO_MM
    A_fine      = np.load(era5_dir / "pair_A_fine.npy") * M_TO_MM
    A_dates     = np.load(era5_dir / "valid_times_A.npy")

    B_coarse_up = np.load(era5_dir / "pair_B_coarse_up.npy") * M_TO_MM
    B_fine      = np.load(era5_dir / "pair_B_fine.npy") * M_TO_MM
    B_dates     = np.load(era5_dir / "valid_times_B.npy")

    # Year-filtered indices (written by build_xgb_data.py)
    if (feat_dir / "A_test_idx.npy").exists():
        A_test_idx = np.load(feat_dir / "A_test_idx.npy")
        B_test_idx = np.load(feat_dir / "B_test_idx.npy")
        log.info("  Using year-filtered test indices from features dir")
    else:
        log.warning("  No A_test_idx.npy in features dir — falling back to "
                    "era5_dir/test_indices_*.npy (may not be 2025-only)")
        A_test_idx = np.load(era5_dir / "test_indices_A.npy")
        B_test_idx = np.load(era5_dir / "test_indices_B.npy")

    era5_lats = np.load(era5_dir / "era5_lats.npy")
    era5_lons = np.load(era5_dir / "era5_lons.npy")
    if (era5_dir / "era5_12km_lats.npy").exists():
        lats_12km = np.load(era5_dir / "era5_12km_lats.npy")
        lons_12km = np.load(era5_dir / "era5_12km_lons.npy")
    else:
        H_B, W_B = B_fine.shape[1], B_fine.shape[2]
        lats_12km = np.linspace(era5_lats[0], era5_lats[-1], H_B)
        lons_12km = np.linspace(era5_lons[0], era5_lons[-1], W_B)

    X_test_A = np.load(feat_dir / "A_test_X.npy")
    X_test_B = np.load(feat_dir / "B_test_X.npy")

    # Defensive shape check
    H_A, W_A = A_fine.shape[1], A_fine.shape[2]
    H_B, W_B = B_fine.shape[1], B_fine.shape[2]
    n_test_A = len(A_test_idx)
    n_test_B = len(B_test_idx)
    expected_rows_A = n_test_A * H_A * W_A
    expected_rows_B = n_test_B * H_B * W_B
    if X_test_A.shape[0] != expected_rows_A:
        raise RuntimeError(
            f"X_test_A has {X_test_A.shape[0]} rows but expected "
            f"{expected_rows_A} = {n_test_A} days × {H_A}×{W_A}. "
            f"Are A_test_idx.npy and A_test_X.npy from the same build?"
        )
    if X_test_B.shape[0] != expected_rows_B:
        raise RuntimeError(
            f"X_test_B has {X_test_B.shape[0]} rows but expected "
            f"{expected_rows_B} = {n_test_B} days × {H_B}×{W_B}."
        )
    if X_test_A.shape[1] != N_FEATURES:
        raise RuntimeError(
            f"X_test_A has {X_test_A.shape[1]} cols, expected {N_FEATURES}. "
            f"Did you regenerate features with the updated build_xgb_data.py?"
        )

    # Test arrays in mm/day (pair_A/B already scaled above)
    gt_fine_A_mm = np.array([A_fine[t] for t in A_test_idx])
    gt_fine_B_mm = np.array([B_fine[t] for t in B_test_idx])
    coarse_up_A_test_mm = np.array([A_coarse_up[t] for t in A_test_idx])
    coarse_up_B_test_mm = np.array([B_coarse_up[t] for t in B_test_idx])

    log.info(f"  Pair A: {n_test_A} test days, grid ({H_A}, {W_A})")
    log.info(f"  Pair B: {n_test_B} test days, grid ({H_B}, {W_B})")

    all_results = {}

    # ════════════════════════════════════════════════════════════════════
    # PART 1 — Pair A & Pair B evaluation (single-step downscaling)
    # ════════════════════════════════════════════════════════════════════
    log.info("=" * 60)
    log.info("PART 1: Pair A & Pair B (single-step downscaling)")

    # Bicubic
    log.info("Bicubic:")
    gt_all = np.concatenate([gt_fine_A_mm.ravel(), gt_fine_B_mm.ravel()]) \
        if n_test_A and n_test_B else \
        (gt_fine_A_mm.ravel() if n_test_A else gt_fine_B_mm.ravel())
    bicubic_all = np.concatenate([coarse_up_A_test_mm.ravel(),
                                  coarse_up_B_test_mm.ravel()]) \
        if n_test_A and n_test_B else \
        (coarse_up_A_test_mm.ravel() if n_test_A else coarse_up_B_test_mm.ravel())
    all_results["pairAB_bicubic"] = compute_metrics(gt_all, bicubic_all)
    log.info(f"  RMSE={all_results['pairAB_bicubic']['rmse']:.4f} mm/day, "
             f"R²={all_results['pairAB_bicubic']['r2']:.4f}")

    # XGBoost + AEF
    log.info("XGBoost + AEF:")
    xgb_fine_A_mm = run_xgboost_predictions(xgb_aef, X_test_A,
                                            coarse_up_A_test_mm, n_test_A, H_A, W_A)
    xgb_fine_B_mm = run_xgboost_predictions(xgb_aef, X_test_B,
                                            coarse_up_B_test_mm, n_test_B, H_B, W_B)
    xgb_all = np.concatenate([xgb_fine_A_mm.ravel(), xgb_fine_B_mm.ravel()])
    all_results["pairAB_xgb_aef"] = compute_metrics(gt_all, xgb_all)
    log.info(f"  RMSE={all_results['pairAB_xgb_aef']['rmse']:.4f} mm/day, "
             f"R²={all_results['pairAB_xgb_aef']['r2']:.4f}")

    # OT-CFM single-step (Pair A: 50→25, Pair B: 25→12.5)
    log.info("OT-CFM single-step:")
    log.info(f"  Pair A (50→25 km), {n_test_A} days × {args.n_ensemble} ensemble...")
    cfm_fine_A_mm = np.empty((n_test_A, H_A, W_A), dtype=np.float32)
    cfm_ensemble_A_mm = np.empty((n_test_A, args.n_ensemble, H_A, W_A), dtype=np.float32)
    for i, t_idx in enumerate(A_test_idx):
        year = int(str(A_dates[t_idx])[:4])
        aef_t = year - AEF_YEAR_OFFSET
        alpha_c = load_aef_as_tensor(aef_dir, aef_t, 50.0, H_A, W_A, device)
        alpha_f = load_aef_as_tensor(aef_dir, aef_t, 25.0, H_A, W_A, device)
        ens, mean = predict_cfm_single_step(
            cfm_model, coarse_up_A_test_mm[i], alpha_c, alpha_f,
            n_ensemble=args.n_ensemble, n_steps=args.n_steps, device=device,
            ensemble_batch=args.ensemble_batch,
        )
        cfm_fine_A_mm[i] = mean
        cfm_ensemble_A_mm[i] = ens
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"    Pair A: {i + 1}/{n_test_A}")

    log.info(f"  Pair B (25→12.5 km), {n_test_B} days × {args.n_ensemble} ensemble...")
    cfm_fine_B_mm = np.empty((n_test_B, H_B, W_B), dtype=np.float32)
    cfm_ensemble_B_mm = np.empty((n_test_B, args.n_ensemble, H_B, W_B), dtype=np.float32)
    for i, t_idx in enumerate(B_test_idx):
        year = int(str(B_dates[t_idx])[:4])
        aef_t = year - AEF_YEAR_OFFSET
        alpha_c = load_aef_as_tensor(aef_dir, aef_t, 25.0, H_B, W_B, device)
        alpha_f = load_aef_as_tensor(aef_dir, aef_t, 12.5, H_B, W_B, device)
        ens, mean = predict_cfm_single_step(
            cfm_model, coarse_up_B_test_mm[i], alpha_c, alpha_f,
            n_ensemble=args.n_ensemble, n_steps=args.n_steps, device=device,
            ensemble_batch=args.ensemble_batch,
        )
        cfm_fine_B_mm[i] = mean
        cfm_ensemble_B_mm[i] = ens
        if (i + 1) % 50 == 0 or i == 0:
            log.info(f"    Pair B: {i + 1}/{n_test_B}")

    cfm_all = np.concatenate([cfm_fine_A_mm.ravel(), cfm_fine_B_mm.ravel()])
    all_results["pairAB_cfm"] = compute_metrics(gt_all, cfm_all)
    all_results["pairA_cfm"] = compute_metrics(gt_fine_A_mm.ravel(),
                                               cfm_fine_A_mm.ravel())
    all_results["pairB_cfm"] = compute_metrics(gt_fine_B_mm.ravel(),
                                               cfm_fine_B_mm.ravel())

    if n_test_A and args.n_ensemble > 1:
        cfm_prob_A = compute_probabilistic_metrics(cfm_ensemble_A_mm, gt_fine_A_mm)
        all_results["pairA_cfm"].update(cfm_prob_A)
        log.info(f"  Pair A — CRPS={cfm_prob_A['crps']:.4f}, "
                 f"S/S={cfm_prob_A['spread_skill']:.3f}, "
                 f"Cov90={cfm_prob_A['coverage_90']:.3f}")
    if n_test_B and args.n_ensemble > 1:
        cfm_prob_B = compute_probabilistic_metrics(cfm_ensemble_B_mm, gt_fine_B_mm)
        all_results["pairB_cfm"].update(cfm_prob_B)
        log.info(f"  Pair B — CRPS={cfm_prob_B['crps']:.4f}, "
                 f"S/S={cfm_prob_B['spread_skill']:.3f}, "
                 f"Cov90={cfm_prob_B['coverage_90']:.3f}")
    log.info(f"  Combined RMSE={all_results['pairAB_cfm']['rmse']:.4f} mm/day, "
             f"R²={all_results['pairAB_cfm']['r2']:.4f}")

    # Plot Pair A / Pair B
    log.info("Plotting Pair A / Pair B...")
    if n_test_A:
        plot_comparison(
            gt_fine_A_mm,
            {"Bicubic": coarse_up_A_test_mm,
             "XGBoost + AEF": xgb_fine_A_mm,
             "OT-CFM": cfm_fine_A_mm},
            era5_lats, era5_lons, "Pair_A_50to25km", out_dir,
            n_examples=args.n_examples, seed=args.seed,
        )
    if n_test_B:
        plot_comparison(
            gt_fine_B_mm,
            {"Bicubic": coarse_up_B_test_mm,
             "XGBoost + AEF": xgb_fine_B_mm,
             "OT-CFM": cfm_fine_B_mm},
            lats_12km, lons_12km, "Pair_B_25to12km", out_dir,
            n_examples=args.n_examples, seed=args.seed,
        )

    # ════════════════════════════════════════════════════════════════════
    # PART 2 — PRISM 2025 rollout (25 → 1.5625 km)
    # ════════════════════════════════════════════════════════════════════
    if args.prism_dir is None:
        log.info("=" * 60)
        log.info("Skipping PRISM 2025 rollout (no --prism-dir given)")
    else:
        log.info("=" * 60)
        log.info("PART 2: PRISM 2025 rollout (25 → 1.5625 km)")

        # Sample 2025 days from A_test_idx (which is 2025-only when using
        # the year-filtered indices from build_xgb_data.py)
        rng = np.random.RandomState(args.seed)
        n_days = min(args.prism_n_days, n_test_A)
        sampled_t = rng.choice(A_test_idx, size=n_days, replace=False)
        sampled_t.sort()
        log.info(f"  Sampling {n_days} days from {n_test_A} test days")

        # Final-grid coords
        H_final = era5_lats.shape[0] * 16
        W_final = era5_lons.shape[0] * 16
        final_lats = np.linspace(era5_lats[0], era5_lats[-1], H_final)
        final_lons = np.linspace(era5_lons[0], era5_lons[-1], W_final)

        bicubic_25_to_15625 = []
        xgb_aef_rollout = []
        xgb_prism_rollout = []   # only filled if xgb_prism is not None
        cfm_rollout_mean = []
        cfm_rollout_ensemble = []
        prism_truth = []
        kept_dates = []

        for k, t_idx in enumerate(sampled_t):
            date = A_dates[t_idx]
            date_str = str(date)[:10]
            year = int(date_str[:4])

            try:
                truth = load_prism_day(args.prism_dir, date, final_lats, final_lons)
            except Exception as e:
                log.warning(f"  PRISM load failed for {date_str}: {e}")
                truth = None
            if truth is None:
                log.info(f"  [{k+1}/{n_days}] {date_str}: PRISM missing, skipping")
                continue

            era5_25_mm = A_fine[t_idx]   # mm/day

            # 1. Naive 16× bicubic from 25km
            bic = zoom(era5_25_mm,
                       (H_final / era5_25_mm.shape[0],
                        W_final / era5_25_mm.shape[1]),
                       order=3).astype(np.float32)
            bic = np.maximum(bic, 0)
            if truth.shape != bic.shape:
                truth = zoom(truth, (bic.shape[0] / truth.shape[0],
                                     bic.shape[1] / truth.shape[1]),
                             order=1).astype(np.float32)
                truth = np.maximum(truth, 0)

            # 2. XGB AEF recursive rollout
            xgb_aef_pred = xgb_recursive_rollout(
                xgb_aef, era5_25_mm, year, args.aef_dir,
                booster_prism_head=None,
            )
            # crop in case of off-by-one
            xgb_aef_pred = xgb_aef_pred[:H_final, :W_final]

            # 3. XGB AEF + PRISM head (if available)
            if xgb_prism is not None:
                xgb_prism_pred = xgb_recursive_rollout(
                    xgb_aef, era5_25_mm, year, args.aef_dir,
                    booster_prism_head=xgb_prism,
                )
                xgb_prism_pred = xgb_prism_pred[:H_final, :W_final]
            else:
                xgb_prism_pred = None

            # 4. OT-CFM rollout
            cfm_ens, cfm_mean = cfm_recursive_rollout(
                cfm_model, era5_25_mm, year, args.aef_dir, device,
                n_ensemble=args.cfm_rollout_ensemble, n_steps=args.n_steps,
            )
            cfm_mean = cfm_mean[:H_final, :W_final]
            cfm_ens = cfm_ens[:, :H_final, :W_final]

            bicubic_25_to_15625.append(bic[:H_final, :W_final])
            xgb_aef_rollout.append(xgb_aef_pred)
            if xgb_prism_pred is not None:
                xgb_prism_rollout.append(xgb_prism_pred)
            cfm_rollout_mean.append(cfm_mean)
            cfm_rollout_ensemble.append(cfm_ens)
            prism_truth.append(truth[:H_final, :W_final])
            kept_dates.append(date_str)

            log.info(f"  [{k+1}/{n_days}] {date_str}: rolled out OK")

        if not prism_truth:
            log.error("  No valid PRISM 2025 days. Skipping aggregate.")
        else:
            bic_arr   = np.array(bicubic_25_to_15625)
            xgb_arr   = np.array(xgb_aef_rollout)
            cfm_arr   = np.array(cfm_rollout_mean)
            cfm_ens_arr = np.array(cfm_rollout_ensemble)
            truth_arr = np.array(prism_truth)

            log.info(f"  Aggregate over {len(prism_truth)} 2025 days:")
            all_results["prism2025_bicubic"] = compute_metrics(
                truth_arr.ravel(), bic_arr.ravel())
            all_results["prism2025_xgb_aef"] = compute_metrics(
                truth_arr.ravel(), xgb_arr.ravel())
            log.info(f"    Bicubic RMSE={all_results['prism2025_bicubic']['rmse']:.4f}")
            log.info(f"    XGB AEF RMSE={all_results['prism2025_xgb_aef']['rmse']:.4f}")

            if xgb_prism_rollout:
                xgb_prism_arr = np.array(xgb_prism_rollout)
                all_results["prism2025_xgb_aef_plus_head"] = compute_metrics(
                    truth_arr.ravel(), xgb_prism_arr.ravel())
                log.info(f"    XGB AEF + PRISM head RMSE="
                         f"{all_results['prism2025_xgb_aef_plus_head']['rmse']:.4f}")

            all_results["prism2025_cfm"] = compute_metrics(
                truth_arr.ravel(), cfm_arr.ravel())
            log.info(f"    OT-CFM RMSE={all_results['prism2025_cfm']['rmse']:.4f}")

            if args.cfm_rollout_ensemble > 1:
                # cfm_ens_arr: (T, K, H, W)
                cfm_prob = compute_probabilistic_metrics(cfm_ens_arr, truth_arr)
                all_results["prism2025_cfm"].update(cfm_prob)
                log.info(f"    OT-CFM CRPS={cfm_prob['crps']:.4f}, "
                         f"S/S={cfm_prob['spread_skill']:.3f}")

            # Plots — pick a few rainy 2025 days
            log.info("  Plotting PRISM 2025 examples...")
            preds_dict = {
                "Bicubic": bic_arr,
                "XGBoost + AEF": xgb_arr,
                "OT-CFM": cfm_arr,
            }
            if xgb_prism_rollout:
                preds_dict["XGB + AEF + PRISM head"] = np.array(xgb_prism_rollout)
            plot_comparison(
                truth_arr, preds_dict,
                final_lats, final_lons, "PRISM2025_25to1p5625km", out_dir,
                n_examples=min(args.n_examples, len(prism_truth)),
                seed=args.seed,
            )

    # ── Summary tables ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SUMMARY (all metrics in mm/day)")
    log.info(f"{'Metric':<40} {'RMSE':>10} {'MAE':>10} {'R²':>10} {'Bias':>12}")
    log.info("-" * 84)
    for name, m in all_results.items():
        log.info(f"{name:<40} {m['rmse']:>10.4f} {m['mae']:>10.4f} "
                 f"{m['r2']:>10.4f} {m['bias']:>12.6f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    log.info(f"\nResults saved to: {out_dir}")
    log.info(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()