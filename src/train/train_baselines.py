"""
Baseline comparison for precipitation downscaling with a PRISM-supervised
fine-tuning stage.

Models:
  1. Bicubic interpolation (residual = 0)
  2. XGBoost WITHOUT AEF embeddings  (1 feature: coarse_precip only)
  3. XGBoost WITH AEF embeddings     (129 features)
  4. XGBoost WITH AEF + PRISM-refined head ("Pair C")
       Recursively rolls the AEF model out from 25km → 1.5625km on
       training-year days, then trains a *second* booster whose target
       is the log-space residual between PRISM-coarsened-to-1.5625km
       and the AEF model's 3.125km-bicubic-upsampled prediction.
       This injects PRISM information into the baseline pipeline.

Notes:
  - Log-transform: residuals computed in log1p space to compress the
    heavy-tailed precipitation distribution.
  - Residuals are stored as mm/day by build_xgb_data.py. The log1p is
    applied here on top of mm/day values.
  - Coarse-precip feature is also log1p-transformed at predict time to
    match training.
  - Test split is years in TEST_YEARS (default {2025}); training uses
    years in TRAIN_YEARS (default 2017..2024). The script reads the
    pre-split indices written by build_xgb_data.py.

Usage:
    python -u train_baselines_and_results.py \
        --features-dir data/xgb_features \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_tif_2018_2024 \
        --output-dir results/baselines
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
M_TO_MM = 1000.0
D_AEF = 64
N_NON_AEF_FEATURES = 1                          # just coarse_precip
N_FEATURES_WITH_AEF = 1 + D_AEF + D_AEF         # 129

# Recursive downscaling steps used for the PRISM-supervised stage:
# (in_res_km, out_res_km, aef_coarse_km, aef_fine_km)
DOWNSCALE_STEPS = [
    (25.0,    12.5,    25.0,    12.5),
    (12.5,     6.25,   12.5,     6.25),
    ( 6.25,    3.125,   6.25,    3.125),
    ( 3.125,   1.5625,  3.125,   1.5625),
]


# ── Generic helpers ──────────────────────────────────────────────────────────

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


def log_transform_residuals(coarse_up, fine):
    """log1p(fine) - log1p(coarse_up), all clipped >= 0."""
    return np.log1p(np.maximum(fine, 0)) - np.log1p(np.maximum(coarse_up, 0))


def inverse_log_residual(coarse_up, log_residual):
    """Reconstruct: max(expm1(log1p(coarse_up) + r), 0)."""
    log_coarse = np.log1p(np.maximum(coarse_up, 0))
    return np.maximum(np.expm1(log_coarse + log_residual), 0)


def log_transform_X(X):
    """Apply log1p to the precipitation column (col 0). Returns a copy."""
    out = X.copy()
    out[:, 0] = np.log1p(np.maximum(out[:, 0], 0))
    return out


# ── AEF helpers (mirrors build_xgb_data.py) ──────────────────────────────────

def load_aef_nc(nc_path: Path) -> np.ndarray:
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32).transpose(1, 2, 0)
    ds.close()
    return arr


def find_aef_file(aef_dir: Path, t_idx: int, res_km):
    candidates = [
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{res_km}km.nc",
        aef_dir / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"aef_illinois_{res_km}km.nc",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def resize_aef_grid(aef_hwd: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Bilinear-resize a (H, W, D) AEF grid to (target_h, target_w, D)."""
    D = aef_hwd.shape[2]
    return np.stack([
        zoom(aef_hwd[:, :, d],
             (target_h / aef_hwd.shape[0], target_w / aef_hwd.shape[1]),
             order=1)
        for d in range(D)
    ], axis=-1).astype(np.float32)


def get_aef_for_year_resized(aef_dir: Path, year: int, res_km,
                             target_h: int, target_w: int) -> np.ndarray:
    """Load (or zero-fill) AEF at `res_km` for `year`, resized to target grid."""
    t_idx = year - AEF_YEAR_OFFSET
    p = find_aef_file(aef_dir, t_idx, res_km)
    if p is None:
        return np.zeros((target_h, target_w, D_AEF), dtype=np.float32)
    raw = load_aef_nc(p)  # (H, W, D)
    return resize_aef_grid(raw, target_h, target_w)


# ── PRISM loader ─────────────────────────────────────────────────────────────

def load_prism_day(prism_dir, date, target_lats, target_lons):
    """Load PRISM TIF for `date` (np.datetime64 or 'YYYY-MM-DD'), regrid to
    (target_lats, target_lons). Returns mm/day or None."""
    import rasterio  # lazy import — only needed in PRISM stage

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

    # Allow caller to pass lon either positive or negative; PRISM is negative.
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
    return np.maximum(regrid, 0)  # mm/day


# ── Recursive rollout of Stage-2 (AEF) model from 25km to 1.5625km ──────────

def rollout_one_day(
    booster, era5_25km_mm, year, aef_dir,
    return_intermediate=False,
):
    """
    Take a single ERA5 25km field (already in mm/day), recursively
    downscale through DOWNSCALE_STEPS using the trained AEF booster,
    and return the predicted field at 1.5625km PLUS, optionally, the
    coarse_up at the final step (needed when training the PRISM head).

    Returns
    -------
    dict with keys:
        "final"       : (H_final, W_final) mm/day prediction at 1.5625km
        "coarse_up_3" : (H_final, W_final) bicubic upsample fed into step 3
                        (i.e. 3.125km field bicubic-upsampled to 1.5625km grid)
        "step3_X"     : (H_final*W_final, 129) feature matrix used at step 3,
                        or None if return_intermediate=False
    """
    import xgboost as xgb

    aef_dir = Path(aef_dir)
    current = era5_25km_mm.astype(np.float32)
    step3_coarse_up = None
    step3_X = None

    for step_idx, (in_res, out_res, aef_c_res, aef_f_res) in enumerate(DOWNSCALE_STEPS):
        H_in, W_in = current.shape
        H_out, W_out = H_in * 2, W_in * 2

        # Bicubic upsample current field
        coarse_up = zoom(current, (H_out / H_in, W_out / W_in),
                         order=3).astype(np.float32)
        coarse_up = np.maximum(coarse_up, 0)

        # AEF (resized to output grid)
        aef_c = get_aef_for_year_resized(aef_dir, year, aef_c_res, H_out, W_out)
        aef_f = get_aef_for_year_resized(aef_dir, year, aef_f_res, H_out, W_out)

        # Build features: [log1p(coarse), aef_coarse, aef_fine]
        n_pix = H_out * W_out
        X = np.empty((n_pix, N_FEATURES_WITH_AEF), dtype=np.float32)
        col = 0
        X[:, col] = np.log1p(coarse_up).ravel(); col += 1
        X[:, col:col + D_AEF] = aef_c.reshape(-1, D_AEF); col += D_AEF
        X[:, col:col + D_AEF] = aef_f.reshape(-1, D_AEF)

        if step_idx == 3 and return_intermediate:
            step3_coarse_up = coarse_up.copy()
            step3_X = X.copy()

        # Predict log-space residual, reconstruct
        log_resid = booster.predict(xgb.DMatrix(X)).reshape(H_out, W_out)
        current = inverse_log_residual(coarse_up, log_resid)

    return {
        "final": current,
        "coarse_up_3": step3_coarse_up,
        "step3_X": step3_X,
    }


# ── Visualization ────────────────────────────────────────────────────────────

def plot_comparison(fine_true_grids, predictions_dict, lats, lons,
                    pair_name, output_dir, n_examples=5, seed=42):
    rng = np.random.RandomState(seed)
    n_times = fine_true_grids.shape[0]
    if n_times == 0:
        return
    example_indices = rng.choice(n_times, size=min(n_examples, n_times), replace=False)
    example_indices.sort()

    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    for ex_i, t_idx in enumerate(example_indices):
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4),
                                 constrained_layout=True)
        gt = fine_true_grids[t_idx]
        all_vals = [gt] + [predictions_dict[m][t_idx] for m in model_names]
        vmin = min(v.min() for v in all_vals)
        vmax = max(v.max() for v in all_vals)

        im = axes[0].imshow(gt, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                            extent=[lons[0], lons[-1], lats[-1], lats[0]],
                            aspect="auto")
        axes[0].set_title("Ground Truth", fontsize=10, fontweight="bold")
        axes[0].set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        for j, name in enumerate(model_names):
            pred = predictions_dict[name][t_idx]
            axes[j + 1].imshow(pred, cmap="YlGnBu", vmin=vmin, vmax=vmax,
                               extent=[lons[0], lons[-1], lats[-1], lats[0]],
                               aspect="auto")
            rmse = np.sqrt(((gt - pred) ** 2).mean())
            axes[j + 1].set_title(f"{name}\nRMSE={rmse:.6f}", fontsize=9)
            axes[j + 1].set_xlabel("Longitude")

        fig.colorbar(im, ax=axes, shrink=0.8, label="Precipitation (mm/day)")
        fig.suptitle(f"{pair_name} — Test Example {ex_i + 1}", fontsize=12)
        fig.savefig(output_dir / f"{pair_name}_example_{ex_i + 1}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved {pair_name}_example_{ex_i + 1}.png")

    # Error maps for last example
    t_idx = example_indices[-1]
    gt = fine_true_grids[t_idx]
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4),
                             constrained_layout=True)
    if n_models == 1:
        axes = [axes]
    for j, name in enumerate(model_names):
        diff = predictions_dict[name][t_idx] - gt
        abs_max = max(abs(diff.min()), abs(diff.max()), 1e-8)
        axes[j].imshow(diff, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                       extent=[lons[0], lons[-1], lats[-1], lats[0]],
                       aspect="auto")
        rmse = np.sqrt((diff ** 2).mean())
        axes[j].set_title(f"{name}\nError (RMSE={rmse:.6f})", fontsize=9)
        axes[j].set_xlabel("Longitude")
        if j == 0:
            axes[j].set_ylabel("Latitude")
    fig.suptitle(f"{pair_name} — Prediction Errors (Example {len(example_indices)})")
    fig.savefig(output_dir / f"{pair_name}_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Baselines + PRISM-supervised refinement (Pair C)."
    )
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True,
                        help="Required for the PRISM-supervised stage.")
    parser.add_argument("--prism-train-dir", required=True,
                        help="PRISM daily TIFs root, with YYYY/ subdirs.")
    parser.add_argument("--prism-test-dir",
                        default=None,
                        help="PRISM data directory for 2025 test year (defaults to prism-train-dir if not set)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-child-weight", type=int, default=50)
    parser.add_argument("--n-rounds", type=int, default=500)
    parser.add_argument("--early-stopping", type=int, default=20)
    parser.add_argument("--prism-train-days", type=int, default=300,
                        help="Number of training-year days to roll out for the PRISM stage.")
    parser.add_argument("--prism-n-rounds", type=int, default=300,
                        help="Boosting rounds for the PRISM head.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import xgboost as xgb

    t_start = time.time()
    feat_dir = Path(args.features_dir)
    era5_dir = Path(args.era5_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prism_train_dir = Path(args.prism_train_dir)
    prism_test_dir = Path(args.prism_test_dir) if args.prism_test_dir else prism_train_dir
    log.info(f"PRISM train dir: {prism_train_dir}")
    log.info(f"PRISM test dir:  {prism_test_dir}")


    # ── Load feature matrices and indices written by build_xgb_data.py ───
    log.info("Loading feature matrices...")
    X_train_A = np.load(feat_dir / "A_train_X.npy")
    y_train_A = np.load(feat_dir / "A_train_y.npy")          # mm/day residuals
    X_test_A  = np.load(feat_dir / "A_test_X.npy")
    y_test_A  = np.load(feat_dir / "A_test_y.npy")
    X_train_B = np.load(feat_dir / "B_train_X.npy")
    y_train_B = np.load(feat_dir / "B_train_y.npy")
    X_test_B  = np.load(feat_dir / "B_test_X.npy")
    y_test_B  = np.load(feat_dir / "B_test_y.npy")

    A_train_idx = np.load(feat_dir / "A_train_idx.npy")
    A_test_idx  = np.load(feat_dir / "A_test_idx.npy")
    B_train_idx = np.load(feat_dir / "B_train_idx.npy")
    B_test_idx  = np.load(feat_dir / "B_test_idx.npy")

    # ERA5 fields (m/day) — convert to mm/day for consistency with features
    A_coarse_up = np.load(era5_dir / "pair_A_coarse_up.npy") * M_TO_MM
    A_fine      = np.load(era5_dir / "pair_A_fine.npy") * M_TO_MM
    B_coarse_up = np.load(era5_dir / "pair_B_coarse_up.npy") * M_TO_MM
    B_fine      = np.load(era5_dir / "pair_B_fine.npy") * M_TO_MM
    A_dates     = np.load(era5_dir / "valid_times_A.npy")

    era5_lats = np.load(era5_dir / "era5_lats.npy")
    era5_lons = np.load(era5_dir / "era5_lons.npy")
    if (era5_dir / "era5_12km_lats.npy").exists():
        lats_12km = np.load(era5_dir / "era5_12km_lats.npy")
        lons_12km = np.load(era5_dir / "era5_12km_lons.npy")
    else:
        lats_12km = np.linspace(era5_lats[0], era5_lats[-1], B_fine.shape[1])
        lons_12km = np.linspace(era5_lons[0], era5_lons[-1], B_fine.shape[2])

    H_A, W_A = A_fine.shape[1], A_fine.shape[2]
    H_B, W_B = B_fine.shape[1], B_fine.shape[2]

    log.info(f"Feature shapes: train_A={X_train_A.shape}, test_A={X_test_A.shape}, "
             f"train_B={X_train_B.shape}, test_B={X_test_B.shape}")
    log.info(f"Test days: A={len(A_test_idx)}, B={len(B_test_idx)} "
             f"(should be 2025 only)")

    # ── Build log-space targets and log-transformed feature matrices ─────
    # Note: y_*_X are mm/day residuals; we convert to log-space residuals using
    # the residual-log identity.  But residual targets in mm/day differ from
    # log1p(fine) - log1p(coarse_up) directly, so we recompute from fields.

    log.info("Computing log-space residuals from fields...")

    def stack_log_resid(coarse_up, fine, indices):
        out = []
        for t in indices:
            r = log_transform_residuals(coarse_up[t], fine[t])  # both mm/day
            out.append(r.ravel())
        return np.concatenate(out) if out else np.array([], dtype=np.float32)

    # IMPORTANT: build_xgb_data.py emits AUGMENTED training rows (heavy days
    # × 4 flips). The naive log-from-fields above only matches if we flip
    # likewise. We do the augmented log targets by flipping the residual grid
    # in the same flip_code order produced by build_xgb_data.
    # We re-derive by reading split_info.json and replicating the order.
    split_info = json.load(open(feat_dir / "split_info.json"))
    augment_train = bool(split_info.get("augment_train", True))
    heavy_pct = float(split_info.get("heavy_precip_percentile", 60.0))

    def build_index_pairs(fine, indices, augment, heavy_pct=heavy_pct):
        if not augment or len(indices) == 0:
            return [(int(t), 0) for t in indices]
        means = np.array([fine[int(t)].mean() for t in indices])
        thr = np.percentile(means, heavy_pct)
        heavy = indices[means >= thr]
        light = indices[means <  thr]
        pairs = [(int(t), 0) for t in light]
        for t in heavy:
            for fc in (0, 1, 2, 3):
                pairs.append((int(t), fc))
        return pairs

    def flip2d(arr, fc):
        out = arr
        if fc in (1, 3): out = np.flip(out, axis=-1)
        if fc in (2, 3): out = np.flip(out, axis=-2)
        return np.ascontiguousarray(out)

    def stack_log_resid_aug(coarse_up, fine, index_pairs):
        out = []
        for t, fc in index_pairs:
            cu = flip2d(coarse_up[t], fc)
            fn = flip2d(fine[t], fc)
            r = log_transform_residuals(cu, fn)
            out.append(r.ravel())
        return np.concatenate(out) if out else np.array([], dtype=np.float32)

    pairs_A_train = build_index_pairs(A_fine, A_train_idx, augment_train)
    pairs_B_train = build_index_pairs(B_fine, B_train_idx, augment_train)
    pairs_A_test  = build_index_pairs(A_fine, A_test_idx,  augment=False)
    pairs_B_test  = build_index_pairs(B_fine, B_test_idx,  augment=False)

    y_train_A_log = stack_log_resid_aug(A_coarse_up, A_fine, pairs_A_train)
    y_train_B_log = stack_log_resid_aug(B_coarse_up, B_fine, pairs_B_train)
    y_test_A_log  = stack_log_resid_aug(A_coarse_up, A_fine, pairs_A_test)
    y_test_B_log  = stack_log_resid_aug(B_coarse_up, B_fine, pairs_B_test)

    log.info(f"  log-resid sizes: trainA={y_train_A_log.size}, "
             f"trainB={y_train_B_log.size}, testA={y_test_A_log.size}, "
             f"testB={y_test_B_log.size}")
    log.info(f"  Pair A log-resid:  mean={y_train_A_log.mean():.6f}, "
             f"std={y_train_A_log.std():.6f}")
    log.info(f"  Pair B log-resid:  mean={y_train_B_log.mean():.6f}, "
             f"std={y_train_B_log.std():.6f}")

    # Defensive shape check
    if y_train_A_log.size != X_train_A.shape[0]:
        raise RuntimeError(
            f"Mismatch: y_train_A_log ({y_train_A_log.size}) vs "
            f"X_train_A rows ({X_train_A.shape[0]}). "
            f"Did you regenerate features with the new build_xgb_data.py?"
        )

    # log-transform the precipitation column (col 0) of each X
    X_train_A_log = log_transform_X(X_train_A)
    X_test_A_log  = log_transform_X(X_test_A)
    X_train_B_log = log_transform_X(X_train_B)
    X_test_B_log  = log_transform_X(X_test_B)

    # column index of "no AEF" features = just coarse_precip
    no_aef_cols = [0]

    # ── XGBoost params (shared) ──────────────────────────────────────────
    xgb_params = {
        "objective": "reg:squarederror",
        "max_depth": args.max_depth,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": args.min_child_weight,
        "tree_method": "hist",
        "seed": args.seed,
    }
    log.info(f"XGBoost: max_depth={args.max_depth}, "
             f"min_child_weight={args.min_child_weight}, rounds={args.n_rounds}")

    all_results = {}

    # Ground truth (test-set, mm/day)
    gt_fine_A = np.array([A_fine[t] for t in A_test_idx])
    gt_fine_B = np.array([B_fine[t] for t in B_test_idx])
    coarse_up_A_test = np.array([A_coarse_up[t] for t in A_test_idx])
    coarse_up_B_test = np.array([B_coarse_up[t] for t in B_test_idx])

    if len(A_test_idx) == 0 or len(B_test_idx) == 0:
        log.warning("Empty 2025 test set for at least one pair — "
                    "Pair A/B test metrics will be NaN.")

    gt_all = np.concatenate([gt_fine_A.ravel(), gt_fine_B.ravel()]) \
        if len(A_test_idx) and len(B_test_idx) else \
        (gt_fine_A.ravel() if len(A_test_idx) else gt_fine_B.ravel())

    # ── Model 1: Bicubic ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Model 1: Bicubic Interpolation")
    bicubic_all = np.concatenate([coarse_up_A_test.ravel(), coarse_up_B_test.ravel()]) \
        if len(A_test_idx) and len(B_test_idx) else \
        (coarse_up_A_test.ravel() if len(A_test_idx) else coarse_up_B_test.ravel())
    all_results["bicubic"] = compute_metrics(gt_all, bicubic_all)
    log.info(f"  RMSE={all_results['bicubic']['rmse']:.6f}, "
             f"R²={all_results['bicubic']['r2']:.4f}")

    # ── Model 2: XGBoost no AEF (log-space) ──────────────────────────────
    log.info("=" * 60)
    log.info("Model 2: XGBoost without AEF (log-space, 1 feature)")

    X_tr = np.concatenate([X_train_A_log[:, no_aef_cols],
                           X_train_B_log[:, no_aef_cols]])
    y_tr = np.concatenate([y_train_A_log, y_train_B_log])
    X_te = np.concatenate([X_test_A_log[:, no_aef_cols],
                           X_test_B_log[:, no_aef_cols]])
    y_te = np.concatenate([y_test_A_log, y_test_B_log])

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest  = xgb.DMatrix(X_te, label=y_te)
    model_no_aef = xgb.train(
        xgb_params, dtrain, num_boost_round=args.n_rounds,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=args.early_stopping, verbose_eval=50,
    )
    model_no_aef.save_model(str(out_dir / "xgb_no_aef.json"))

    pred_log = model_no_aef.predict(dtest)
    n_A = len(A_test_idx) * H_A * W_A

    if len(A_test_idx):
        no_aef_fine_A = np.array([
            inverse_log_residual(coarse_up_A_test[i],
                                 pred_log[:n_A].reshape(len(A_test_idx), H_A, W_A)[i])
            for i in range(len(A_test_idx))
        ])
    else:
        no_aef_fine_A = np.empty((0, H_A, W_A), dtype=np.float32)

    if len(B_test_idx):
        no_aef_fine_B = np.array([
            inverse_log_residual(coarse_up_B_test[i],
                                 pred_log[n_A:].reshape(len(B_test_idx), H_B, W_B)[i])
            for i in range(len(B_test_idx))
        ])
    else:
        no_aef_fine_B = np.empty((0, H_B, W_B), dtype=np.float32)

    no_aef_all = np.concatenate([no_aef_fine_A.ravel(), no_aef_fine_B.ravel()])
    all_results["xgb_no_aef"] = compute_metrics(gt_all, no_aef_all)
    log.info(f"  RMSE={all_results['xgb_no_aef']['rmse']:.6f}, "
             f"R²={all_results['xgb_no_aef']['r2']:.4f}")
    del X_tr, dtrain

    # ── Model 3: XGBoost with AEF (log-space) ────────────────────────────
    log.info("=" * 60)
    log.info("Model 3: XGBoost with AEF (log-space, 129 features)")

    X_tr = np.concatenate([X_train_A_log, X_train_B_log])
    X_te = np.concatenate([X_test_A_log,  X_test_B_log])
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dtest  = xgb.DMatrix(X_te, label=y_te)

    model_aef = xgb.train(
        xgb_params, dtrain, num_boost_round=args.n_rounds,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=args.early_stopping, verbose_eval=50,
    )
    model_aef.save_model(str(out_dir / "xgb_with_aef.json"))

    pred_log = model_aef.predict(dtest)

    if len(A_test_idx):
        aef_fine_A = np.array([
            inverse_log_residual(coarse_up_A_test[i],
                                 pred_log[:n_A].reshape(len(A_test_idx), H_A, W_A)[i])
            for i in range(len(A_test_idx))
        ])
    else:
        aef_fine_A = np.empty((0, H_A, W_A), dtype=np.float32)
    if len(B_test_idx):
        aef_fine_B = np.array([
            inverse_log_residual(coarse_up_B_test[i],
                                 pred_log[n_A:].reshape(len(B_test_idx), H_B, W_B)[i])
            for i in range(len(B_test_idx))
        ])
    else:
        aef_fine_B = np.empty((0, H_B, W_B), dtype=np.float32)

    aef_all = np.concatenate([aef_fine_A.ravel(), aef_fine_B.ravel()])
    all_results["xgb_with_aef"] = compute_metrics(gt_all, aef_all)
    log.info(f"  RMSE={all_results['xgb_with_aef']['rmse']:.6f}, "
             f"R²={all_results['xgb_with_aef']['r2']:.4f}")

    importance = model_aef.get_score(importance_type="gain")
    top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
    log.info("  Top 10 features:")
    for fname, gain in top_features:
        log.info(f"    {fname}: {gain:.4f}")

    del X_tr, dtrain, dtest

    # ── Model 4: PRISM-supervised refinement ("Pair C") ──────────────────
    # Workflow:
    #   For each of `prism_train_days` randomly-sampled training-year days
    #   (years 2017..2024 from valid_times_A):
    #     1) Take ERA5 25km field for that day.
    #     2) Recursively roll out with model_aef through 4 steps; record:
    #          - coarse_up_3 (3.125km field bicubic-upsampled to 1.5625km grid)
    #          - step3_X     (the 129-feature matrix used at step 3)
    #     3) Load PRISM 800m for that day, regrid to the 1.5625km grid.
    #     4) Target = log1p(PRISM) - log1p(coarse_up_3).
    #   Train an XGBoost head on (step3_X, target) — i.e. a residual head
    #   on top of the AEF model's final-step bicubic upsample.
    log.info("=" * 60)
    log.info("Model 4: XGBoost with AEF + PRISM-supervised head ('Pair C')")

    # 25km native field is Pair A's fine field (in mm/day already)
    A_25km_mm = A_fine

    # Sample training-year days
    rng = np.random.RandomState(args.seed)
    n_avail = len(A_train_idx)
    n_use = min(args.prism_train_days, n_avail)
    sampled_train_t = rng.choice(A_train_idx, size=n_use, replace=False)
    sampled_train_t.sort()

    log.info(f"  Sampling {n_use}/{n_avail} training-year days for PRISM rollout")

    H_final = H_A * 16   # after 4× doublings
    W_final = W_A * 16
    final_lats = np.linspace(era5_lats[0], era5_lats[-1], H_final)
    final_lons = np.linspace(era5_lons[0], era5_lons[-1], W_final)

    X_prism_rows = []
    y_prism_rows = []
    n_used = 0
    n_skipped = 0
    t0 = time.time()
    for k, t_idx in enumerate(sampled_train_t):
        date = A_dates[t_idx]
        date_str = str(date)[:10]
        year = int(date_str[:4])

        try:
            prism_mm = load_prism_day(args.prism_train_dir, date, final_lats, final_lons)
        except Exception as e:
            log.warning(f"    PRISM load failed for {date_str}: {e}")
            prism_mm = None
        if prism_mm is None:
            n_skipped += 1
            continue

        out = rollout_one_day(
            model_aef, A_25km_mm[t_idx], year, args.aef_dir,
            return_intermediate=True,
        )
        coarse_up_3 = out["coarse_up_3"]   # (H_final, W_final)
        step3_X     = out["step3_X"]       # (H_final*W_final, 129)

        # Match shapes if PRISM regrid landed on a slightly off grid
        if prism_mm.shape != coarse_up_3.shape:
            prism_mm = zoom(prism_mm,
                            (coarse_up_3.shape[0] / prism_mm.shape[0],
                             coarse_up_3.shape[1] / prism_mm.shape[1]),
                            order=1).astype(np.float32)
            prism_mm = np.maximum(prism_mm, 0)

        # Target: log-space residual against PRISM
        log_resid_target = log_transform_residuals(coarse_up_3, prism_mm).ravel()

        X_prism_rows.append(step3_X)
        y_prism_rows.append(log_resid_target)
        n_used += 1

        if (k + 1) % 25 == 0:
            log.info(f"    Rolled out {k+1}/{n_use} days "
                     f"(used={n_used}, skipped={n_skipped}, "
                     f"elapsed={time.time()-t0:.0f}s)")

    if n_used == 0:
        log.error("  No PRISM-supervised samples could be built. Skipping Model 4.")
        prism_fine_final = None
    else:
        X_prism_train = np.concatenate(X_prism_rows, axis=0)
        y_prism_train = np.concatenate(y_prism_rows, axis=0)
        log.info(f"  PRISM training set: X={X_prism_train.shape}, "
                 f"y mean={y_prism_train.mean():.6f}, std={y_prism_train.std():.6f}")

        # Free intermediate lists
        del X_prism_rows, y_prism_rows

        # Optional small validation split: hold out last 10% rolled-out days
        n_rows = X_prism_train.shape[0]
        n_val = max(1, int(0.1 * n_rows))
        idx_perm = rng.permutation(n_rows)
        val_idx, tr_idx = idx_perm[:n_val], idx_perm[n_val:]

        d_pr_train = xgb.DMatrix(X_prism_train[tr_idx], label=y_prism_train[tr_idx])
        d_pr_val   = xgb.DMatrix(X_prism_train[val_idx], label=y_prism_train[val_idx])

        prism_head = xgb.train(
            xgb_params, d_pr_train, num_boost_round=args.prism_n_rounds,
            evals=[(d_pr_train, "train"), (d_pr_val, "val")],
            early_stopping_rounds=args.early_stopping, verbose_eval=50,
        )
        prism_head.save_model(str(out_dir / "xgb_prism_head.json"))
        del d_pr_train, d_pr_val, X_prism_train, y_prism_train

        # ── Apply on 2025 test days ──────────────────────────────────────
        log.info("  Applying combined AEF+PRISM model to 2025 test days...")
        n_test = len(A_test_idx)
        prism_fine_final = np.empty((n_test, H_final, W_final), dtype=np.float32)
        prism_truth_2025 = np.full((n_test, H_final, W_final), np.nan, dtype=np.float32)

        for i, t_idx in enumerate(A_test_idx):
            date = A_dates[t_idx]
            date_str = str(date)[:10]
            year = int(date_str[:4])

            out = rollout_one_day(
                model_aef, A_25km_mm[t_idx], year, args.aef_dir,
                return_intermediate=True,
            )
            coarse_up_3 = out["coarse_up_3"]
            step3_X     = out["step3_X"]

            log_resid = prism_head.predict(xgb.DMatrix(step3_X)).reshape(H_final, W_final)
            prism_fine_final[i] = inverse_log_residual(coarse_up_3, log_resid)

            # Cache PRISM truth for 2025
            try:
                prism_truth = load_prism_day(args.prism_test_dir, date, final_lats, final_lons)
            except Exception:
                prism_truth = None
            if prism_truth is not None:
                if prism_truth.shape != (H_final, W_final):
                    prism_truth = zoom(prism_truth,
                                       (H_final / prism_truth.shape[0],
                                        W_final / prism_truth.shape[1]),
                                       order=1).astype(np.float32)
                    prism_truth = np.maximum(prism_truth, 0)
                prism_truth_2025[i] = prism_truth

            if (i + 1) % 25 == 0:
                log.info(f"    Test rollout {i+1}/{n_test}")

        # ── Metrics on 2025 PRISM ────────────────────────────────────────
        valid_mask = ~np.isnan(prism_truth_2025[:, 0, 0])
        n_valid_test = int(valid_mask.sum())
        if n_valid_test > 0:
            preds_v = prism_fine_final[valid_mask]
            truth_v = prism_truth_2025[valid_mask]
            # bicubic-only baseline at 1.5625km from 25km: just zoom 16×
            bicubic_v = np.array([
                np.maximum(zoom(A_25km_mm[t], (16, 16), order=3), 0)
                for t in A_test_idx[valid_mask]
            ]).astype(np.float32)
            # crop bicubic to (H_final, W_final) in case of off-by-one
            bicubic_v = bicubic_v[:, :H_final, :W_final]

            all_results["xgb_aef_prism_vs_prism2025"] = \
                compute_metrics(truth_v.ravel(), preds_v.ravel())
            all_results["bicubic_vs_prism2025"] = \
                compute_metrics(truth_v.ravel(), bicubic_v.ravel())
            all_results["xgb_with_aef_only_vs_prism2025"] = compute_metrics(
                truth_v.ravel(),
                # AEF-only output at 1.5625km is the rollout WITHOUT a PRISM head
                # = inverse_log_residual(coarse_up_3, model_aef_step3_pred), but
                # we conveniently already have coarse_up_3 in `out`. Cheaper:
                # re-run rollout returning the final field.
                np.array([
                    rollout_one_day(model_aef, A_25km_mm[t], int(str(A_dates[t])[:4]),
                                    args.aef_dir, return_intermediate=False)["final"]
                    for t in A_test_idx[valid_mask]
                ]).ravel(),
            )
            log.info(f"  PRISM-2025 metrics over {n_valid_test} days:")
            for k in ("bicubic_vs_prism2025", "xgb_with_aef_only_vs_prism2025",
                      "xgb_aef_prism_vs_prism2025"):
                m = all_results[k]
                log.info(f"    {k:40s}  RMSE={m['rmse']:.6f}, R²={m['r2']:.4f}, "
                         f"bias={m['bias']:.6f}")
        else:
            log.warning("  No PRISM 2025 truth available — skipping PRISM-2025 metrics.")

    # ── Visualize Pair A and Pair B test predictions ─────────────────────
    log.info("=" * 60)
    log.info("Plotting Pair A / Pair B example panels...")

    if len(A_test_idx):
        plot_comparison(
            gt_fine_A,
            {"Bicubic": coarse_up_A_test,
             "XGB (no AEF)": no_aef_fine_A,
             "XGB + AEF": aef_fine_A},
            era5_lats, era5_lons,
            "Pair_A_50to25km", out_dir, n_examples=args.n_examples,
        )
    if len(B_test_idx):
        plot_comparison(
            gt_fine_B,
            {"Bicubic": coarse_up_B_test,
             "XGB (no AEF)": no_aef_fine_B,
             "XGB + AEF": aef_fine_B},
            lats_12km, lons_12km,
            "Pair_B_25to12km", out_dir, n_examples=args.n_examples,
        )

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("RESULTS")
    for name, m in all_results.items():
        log.info(f"  {name:42s}  RMSE={m['rmse']:.6f}  "
                 f"MAE={m['mae']:.6f}  R²={m['r2']:.4f}  bias={m['bias']:.6f}")

    with open(out_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\nResults: {out_dir}")
    log.info(f"Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()