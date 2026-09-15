"""
Standalone replot script — load already-trained XGBoost baselines and
re-generate the Pair A / Pair B example panels with a corrected colorbar
that anchors on Ground Truth instead of the global panel max.

Usage:
    python -u replot_baselines.py \
        --baselines-dir results/baselines_phase3_newEra5 \
        --features-dir data/xgb_features \
        --era5-dir data/era5_processed \
        --output-dir results/baselines_phase3_newEra5 \
        --n-examples 5
"""

import argparse
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


M_TO_MM = 1000.0


# ── Helpers ──────────────────────────────────────────────────────────────────

def log_transform_X(X):
    """Apply log1p to the precipitation column (col 0). Returns a copy."""
    out = X.copy()
    out[:, 0] = np.log1p(np.maximum(out[:, 0], 0))
    return out


def inverse_log_residual(coarse_up, log_residual):
    log_coarse = np.log1p(np.maximum(coarse_up, 0))
    return np.maximum(np.expm1(log_coarse + log_residual), 0)


def predict_pair(booster, X_test_log, coarse_up_test, n_test, H, W):
    """Run booster on one pair's test features. Returns (n_test, H, W) mm/day."""
    import xgboost as xgb
    if n_test == 0:
        return np.empty((0, H, W), dtype=np.float32)
    pred_log = booster.predict(xgb.DMatrix(X_test_log)).reshape(n_test, H, W)
    out = np.empty((n_test, H, W), dtype=np.float32)
    for i in range(n_test):
        out[i] = inverse_log_residual(coarse_up_test[i], pred_log[i])
    return out


# ── Plotting (anchored on Ground Truth) ──────────────────────────────────────

def plot_comparison(fine_true_grids, predictions_dict, lats, lons,
                    pair_name, output_dir, n_examples=5, seed=42):
    """
    Color scaling is anchored on the GROUND TRUTH for each example, with a
    small headroom factor:
        vmax = max(p99(gt) * 1.2, max(gt) * 1.05, 0.1)
    Predictions exceeding vmax are visibly clipped to dark blue, and the
    panel title is annotated with (pred max=X.XX, clipped).
    """
    rng = np.random.RandomState(seed)
    n_times = fine_true_grids.shape[0]
    if n_times == 0:
        log.warning(f"  {pair_name}: no test days to plot")
        return

    example_indices = rng.choice(n_times, size=min(n_examples, n_times), replace=False)
    example_indices.sort()

    model_names = list(predictions_dict.keys())
    n_models = len(model_names)

    for ex_i, t_idx in enumerate(example_indices):
        fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4),
                                 constrained_layout=True)
        gt = fine_true_grids[t_idx]

        vmin = 0.0
        gt_p99 = float(np.percentile(gt, 99)) if gt.max() > 0 else 0.0
        gt_max = float(gt.max())
        vmax = max(gt_p99 * 1.2, gt_max * 1.05, 0.1)

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
            pmax = float(pred.max())
            extra = f"\n(pred max={pmax:.2f}, clipped)" if pmax > vmax * 1.01 else ""
            axes[j + 1].set_title(f"{name}\nRMSE={rmse:.4f}{extra}", fontsize=9)
            axes[j + 1].set_xlabel("Longitude")

        fig.colorbar(im, ax=axes, shrink=0.8, label="Precipitation (mm/day)")
        fig.suptitle(f"{pair_name} — Test Example {ex_i + 1}", fontsize=12)
        out_path = output_dir / f"{pair_name}_example_{ex_i + 1}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved {out_path.name}  (gt_max={gt_max:.3f}, vmax={vmax:.3f})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reload trained baselines and re-plot Pair A/B test "
                    "examples with a Ground-Truth-anchored colorbar."
    )
    parser.add_argument("--baselines-dir", required=True,
                        help="Directory containing xgb_no_aef.json, xgb_with_aef.json")
    parser.add_argument("--features-dir", required=True,
                        help="Directory containing {A,B}_test_X.npy and {A,B}_test_idx.npy")
    parser.add_argument("--era5-dir", required=True,
                        help="Directory containing pair_{A,B}_{coarse_up,fine}.npy "
                             "and era5_lats.npy / era5_lons.npy")
    parser.add_argument("--output-dir", required=True,
                        help="Where to save the new PNGs")
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import xgboost as xgb

    t_start = time.time()
    base_dir = Path(args.baselines_dir)
    feat_dir = Path(args.features_dir)
    era5_dir = Path(args.era5_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load XGBoost models ──────────────────────────────────────────────
    log.info(f"Loading XGBoost models from {base_dir}")
    no_aef_path = base_dir / "xgb_no_aef.json"
    aef_path    = base_dir / "xgb_with_aef.json"
    if not no_aef_path.exists():
        raise FileNotFoundError(f"Missing: {no_aef_path}")
    if not aef_path.exists():
        raise FileNotFoundError(f"Missing: {aef_path}")

    booster_no_aef = xgb.Booster()
    booster_no_aef.load_model(str(no_aef_path))
    booster_aef = xgb.Booster()
    booster_aef.load_model(str(aef_path))
    log.info(f"  Loaded xgb_no_aef.json and xgb_with_aef.json")

    # ── Load features & indices ──────────────────────────────────────────
    log.info(f"Loading features from {feat_dir}")
    X_test_A = np.load(feat_dir / "A_test_X.npy")
    X_test_B = np.load(feat_dir / "B_test_X.npy")
    A_test_idx = np.load(feat_dir / "A_test_idx.npy")
    B_test_idx = np.load(feat_dir / "B_test_idx.npy")
    log.info(f"  X_test_A: {X_test_A.shape},  A_test_idx: {len(A_test_idx)}")
    log.info(f"  X_test_B: {X_test_B.shape},  B_test_idx: {len(B_test_idx)}")

    # ── Load ERA5 fields (m/day → mm/day) ────────────────────────────────
    log.info(f"Loading ERA5 fields from {era5_dir}")
    A_coarse_up = np.load(era5_dir / "pair_A_coarse_up.npy") * M_TO_MM
    A_fine      = np.load(era5_dir / "pair_A_fine.npy") * M_TO_MM
    B_coarse_up = np.load(era5_dir / "pair_B_coarse_up.npy") * M_TO_MM
    B_fine      = np.load(era5_dir / "pair_B_fine.npy") * M_TO_MM

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
    n_test_A = len(A_test_idx)
    n_test_B = len(B_test_idx)

    # Defensive shape check
    if X_test_A.shape[0] != n_test_A * H_A * W_A:
        raise RuntimeError(
            f"X_test_A has {X_test_A.shape[0]} rows but expected "
            f"{n_test_A * H_A * W_A} = {n_test_A} days x {H_A}x{W_A}."
        )
    if X_test_B.shape[0] != n_test_B * H_B * W_B:
        raise RuntimeError(
            f"X_test_B has {X_test_B.shape[0]} rows but expected "
            f"{n_test_B * H_B * W_B} = {n_test_B} days x {H_B}x{W_B}."
        )

    # Test fields (mm/day, ordered by the test indices)
    gt_fine_A     = np.array([A_fine[t]      for t in A_test_idx])
    gt_fine_B     = np.array([B_fine[t]      for t in B_test_idx])
    coarse_up_A_t = np.array([A_coarse_up[t] for t in A_test_idx])
    coarse_up_B_t = np.array([B_coarse_up[t] for t in B_test_idx])

    # log-transform the precip column (col 0) of test features
    X_test_A_log = log_transform_X(X_test_A)
    X_test_B_log = log_transform_X(X_test_B)

    # ── Run predictions ──────────────────────────────────────────────────
    no_aef_cols = [0]   # only the precipitation column for "no AEF"

    log.info("Predicting Pair A...")
    no_aef_fine_A = predict_pair(
        booster_no_aef, X_test_A_log[:, no_aef_cols],
        coarse_up_A_t, n_test_A, H_A, W_A,
    )
    aef_fine_A = predict_pair(
        booster_aef, X_test_A_log,
        coarse_up_A_t, n_test_A, H_A, W_A,
    )
    log.info(f"  no_aef: shape={no_aef_fine_A.shape}, range={no_aef_fine_A.min():.4f}..{no_aef_fine_A.max():.4f}")
    log.info(f"  +aef:   shape={aef_fine_A.shape}, range={aef_fine_A.min():.4f}..{aef_fine_A.max():.4f}")
    log.info(f"  gt_A:   range={gt_fine_A.min():.4f}..{gt_fine_A.max():.4f}")

    log.info("Predicting Pair B...")
    no_aef_fine_B = predict_pair(
        booster_no_aef, X_test_B_log[:, no_aef_cols],
        coarse_up_B_t, n_test_B, H_B, W_B,
    )
    aef_fine_B = predict_pair(
        booster_aef, X_test_B_log,
        coarse_up_B_t, n_test_B, H_B, W_B,
    )
    log.info(f"  no_aef: shape={no_aef_fine_B.shape}, range={no_aef_fine_B.min():.4f}..{no_aef_fine_B.max():.4f}")
    log.info(f"  +aef:   shape={aef_fine_B.shape}, range={aef_fine_B.min():.4f}..{aef_fine_B.max():.4f}")
    log.info(f"  gt_B:   range={gt_fine_B.min():.4f}..{gt_fine_B.max():.4f}")

    # ── Re-plot with anchored colorbar ───────────────────────────────────
    log.info("Plotting Pair A (50 -> 25 km)...")
    if n_test_A:
        plot_comparison(
            gt_fine_A,
            {"Bicubic":      coarse_up_A_t,
             "XGB (no AEF)": no_aef_fine_A,
             "XGB + AEF":    aef_fine_A},
            era5_lats, era5_lons, "Pair_A_50to25km", out_dir,
            n_examples=args.n_examples, seed=args.seed,
        )

    log.info("Plotting Pair B (25 -> 12.5 km)...")
    if n_test_B:
        plot_comparison(
            gt_fine_B,
            {"Bicubic":      coarse_up_B_t,
             "XGB (no AEF)": no_aef_fine_B,
             "XGB + AEF":    aef_fine_B},
            lats_12km, lons_12km, "Pair_B_25to12km", out_dir,
            n_examples=args.n_examples, seed=args.seed,
        )

    log.info(f"Done in {time.time() - t_start:.0f}s")
    log.info(f"PNGs written to {out_dir}")


if __name__ == "__main__":
    main()