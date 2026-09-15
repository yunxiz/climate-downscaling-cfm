"""
Build tabular feature matrices for XGBoost baseline from preprocessed
ERA5 .npy files and pre-pooled AEF NetCDFs.

CHANGES (vs. original):
  - Removed lat, lon, sin_doy, cos_doy features. Now keeps only:
      * coarse_precip (1)
      * aef_coarse    (64)
      * aef_fine      (64)
    Total: 129 features (or 1 for "no AEF" variant).
  - Train/test split is now strictly by calendar year:
      train years = 2017..2024,  test years = 2025.
    Index files from disk are ignored — we rebuild train/test indices
    from the date arrays.
  - Heavy-precipitation augmentation matching dataset.py:
      * domain-mean threshold = 60th percentile (training only)
      * heavy days emit 4 oriented copies (orig + h-flip + v-flip + both)
      * light days emit 1 copy (orig)
    Augmentation flips both the precipitation field AND the AEF
    embeddings together so the spatial correspondence is preserved.

For each (timestep, flip_code) sample, every pixel becomes a feature row:
  - coarse_precip (1):  bicubic-upsampled coarse value at this pixel (mm/day)
  - aef_coarse    (64): AEF embedding at coarse resolution (oriented)
  - aef_fine      (64): AEF embedding at fine resolution (oriented)
  Total: 129 features

Target: residual (fine - coarse_up) at this pixel, in mm/day.
NOTE: We keep raw mm/day residuals here. The downstream training
script (train_baselines_and_results.py) is responsible for the
log1p transform.

Outputs per pair:
  {pair}_train_X.npy, {pair}_train_y.npy
  {pair}_test_X.npy,  {pair}_test_y.npy
  feature_names.json
  split_info.json   (records which years went into train/test)

Usage:
    python -u build_xgb_data.py \
        --era5-dir data/era5_processed \
        --aef-dir data/aef_downsampled_by_year \
        --output-dir data/xgb_features
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AEF_YEAR_OFFSET = 2017            # aef time index = calendar_year - 2017
TRAIN_YEARS = set(range(2017, 2025))   # 2017..2024 inclusive
TEST_YEARS  = {2025}

# m/day -> mm/day, matches dataset.py
M_TO_MM = 1000.0

# 60th-percentile heavy-precip threshold per dataset.py
HEAVY_PRECIP_PERCENTILE = 60.0

PAIR_AEF_SCALES = {
    "A": {"coarse": 50, "fine": 25},
    "B": {"coarse": 25, "fine": 12.5},
}


# ── AEF I/O ──────────────────────────────────────────────────────────────────

def load_aef_nc(nc_path: Path) -> np.ndarray:
    """Load pre-pooled AEF NetCDF → (H, W, D) float32."""
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32)  # (D, H, W)
    ds.close()
    return arr.transpose(1, 2, 0)  # → (H, W, D)


def find_aef_file(aef_dir: Path, t_idx: int, res_km: float) -> Path:
    """Find AEF file with either naming convention."""
    candidates = [
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{res_km}km.nc",
        aef_dir / f"aef_illinois_t{t_idx}_{res_km}km.nc",
        aef_dir / f"aef_illinois_{res_km}km.nc",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"AEF not found for t{t_idx} at {res_km}km. Tried: {[str(c) for c in candidates]}"
    )


def load_aef_by_year(aef_dir: Path, years: list, coarse_res: float,
                     fine_res: float, target_h: int, target_w: int):
    """
    Load AEF for multiple years, resized to match the target grid.
    Returns dict: year → (coarse_grid, fine_grid)
      coarse_grid: (H, W, D)   — kept as 2D grid so we can flip spatially
      fine_grid:   (H, W, D)
    """
    from scipy.ndimage import zoom

    aef_by_year = {}
    for year in years:
        t_idx = year - AEF_YEAR_OFFSET

        try:
            coarse_path = find_aef_file(aef_dir, t_idx, coarse_res)
            fine_path = find_aef_file(aef_dir, t_idx, fine_res)
        except FileNotFoundError as e:
            log.warning(f"  {e}")
            continue

        ac = load_aef_nc(coarse_path)  # (H_c, W_c, D)
        af = load_aef_nc(fine_path)    # (H_f, W_f, D)

        D = ac.shape[2]

        # Bilinear-resize each channel to (target_h, target_w)
        ac_resized = np.stack([
            zoom(ac[:, :, d], (target_h / ac.shape[0], target_w / ac.shape[1]), order=1)
            for d in range(D)
        ], axis=-1).astype(np.float32)

        af_resized = np.stack([
            zoom(af[:, :, d], (target_h / af.shape[0], target_w / af.shape[1]), order=1)
            for d in range(D)
        ], axis=-1).astype(np.float32)

        # Keep as (H, W, D) so flips can be applied in __getitem-style fashion
        aef_by_year[year] = (ac_resized, af_resized)
        log.info(f"  Year {year}: coarse {ac.shape} → {ac_resized.shape}, "
                 f"fine {af.shape} → {af_resized.shape}")

    return aef_by_year


# ── Augmentation helpers ─────────────────────────────────────────────────────

def apply_flip_2d(field_2d: np.ndarray, flip_code: int) -> np.ndarray:
    """Apply flip_code in {0,1,2,3} to a (H, W) field."""
    out = field_2d
    if flip_code in (1, 3):                       # h-flip (last axis)
        out = np.flip(out, axis=-1)
    if flip_code in (2, 3):                       # v-flip (rows)
        out = np.flip(out, axis=-2)
    return np.ascontiguousarray(out)


def apply_flip_3d_hw(field_3d_hwd: np.ndarray, flip_code: int) -> np.ndarray:
    """Apply flip_code in {0,1,2,3} to a (H, W, D) AEF tensor — flip H/W only."""
    out = field_3d_hwd
    if flip_code in (1, 3):                       # flip W
        out = np.flip(out, axis=1)
    if flip_code in (2, 3):                       # flip H
        out = np.flip(out, axis=0)
    return np.ascontiguousarray(out)


def build_index_pairs(
    fine: np.ndarray,
    indices: np.ndarray,
    augment: bool,
    heavy_pct: float = HEAVY_PRECIP_PERCENTILE,
) -> list:
    """
    Build (time_idx, flip_code) pairs. Heavy days (above `heavy_pct`-th
    percentile of domain-mean precip) get 4 flips; light days get 1.
    """
    if not augment or len(indices) == 0:
        return [(int(t), 0) for t in indices]

    domain_means = np.array([fine[int(t)].mean() for t in indices])
    threshold = np.percentile(domain_means, heavy_pct)
    heavy_mask = domain_means >= threshold

    heavy_idx = indices[heavy_mask]
    light_idx = indices[~heavy_mask]

    pairs = [(int(t), 0) for t in light_idx]
    for t in heavy_idx:
        pairs.append((int(t), 0))
        pairs.append((int(t), 1))
        pairs.append((int(t), 2))
        pairs.append((int(t), 3))

    log.info(f"    Aug: {len(heavy_idx)} heavy × 4 + {len(light_idx)} light × 1 "
             f"= {len(pairs)} samples (was {len(indices)})")
    return pairs


# ── Year-based splitting ─────────────────────────────────────────────────────

def split_indices_by_year(dates: np.ndarray, train_years: set, test_years: set):
    """Return (train_idx, test_idx) selecting calendar years strictly."""
    years = np.array([int(str(d)[:4]) for d in dates])
    train_idx = np.where(np.isin(years, list(train_years)))[0]
    test_idx  = np.where(np.isin(years, list(test_years)))[0]
    return train_idx, test_idx


# ── Feature builder ──────────────────────────────────────────────────────────

def build_features_for_pair(
    pair_name: str,
    coarse_up: np.ndarray,                  # (T, H, W) m/day
    residual: np.ndarray,                   # (T, H, W) m/day  (fine - coarse_up)
    dates: np.ndarray,                      # (T,) datetime64
    indices: np.ndarray,                    # which time indices this split uses
    fine: np.ndarray,                       # (T, H, W) m/day — for heavy-rain ranking
    aef_by_year: dict,                      # year → (coarse_grid_HWD, fine_grid_HWD)
    augment: bool,
) -> tuple:
    """
    Build flat feature matrix and target vector for one pair + split.

    Features (per pixel): [coarse_precip(1), aef_coarse(D), aef_fine(D)]
    Total = 129.

    Both precip and residual are scaled m/day -> mm/day to match the rest
    of the pipeline (dataset.py, training loop).
    """
    H, W = coarse_up.shape[1], coarse_up.shape[2]
    n_pixels = H * W

    if not aef_by_year:
        raise RuntimeError(f"{pair_name}: no AEF years available; cannot build features")

    fallback_year = sorted(aef_by_year.keys())[0]
    D = aef_by_year[fallback_year][0].shape[2]
    n_features = 1 + D + D  # 129

    index_pairs = build_index_pairs(fine, indices, augment=augment)
    n_samples = len(index_pairs) * n_pixels

    log.info(f"  {pair_name}: {len(index_pairs)} (time, flip) pairs × {n_pixels} px "
             f"= {n_samples:,} rows, {n_features} features")

    X = np.empty((n_samples, n_features), dtype=np.float32)
    y = np.empty(n_samples, dtype=np.float32)

    for i, (t, flip_code) in enumerate(index_pairs):
        row_start = i * n_pixels
        row_end = row_start + n_pixels

        # Pull and orient precip + residual (mm/day)
        coarse_grid = (coarse_up[t] * M_TO_MM).astype(np.float32)
        resid_grid  = (residual[t]  * M_TO_MM).astype(np.float32)
        if flip_code != 0:
            coarse_grid = apply_flip_2d(coarse_grid, flip_code)
            resid_grid  = apply_flip_2d(resid_grid,  flip_code)

        # Pull and orient AEF for this sample's year
        year = int(str(dates[t])[:4])
        ac_grid, af_grid = aef_by_year.get(year, aef_by_year[fallback_year])
        if flip_code != 0:
            ac_grid = apply_flip_3d_hw(ac_grid, flip_code)
            af_grid = apply_flip_3d_hw(af_grid, flip_code)

        # Flatten in row-major (H, W) order — matching ravel() used by predictors
        coarse_flat = coarse_grid.ravel()                  # (H*W,)
        ac_flat     = ac_grid.reshape(-1, D)               # (H*W, D)
        af_flat     = af_grid.reshape(-1, D)               # (H*W, D)
        resid_flat  = resid_grid.ravel()

        col = 0
        X[row_start:row_end, col] = coarse_flat;            col += 1
        X[row_start:row_end, col:col+D] = ac_flat;          col += D
        X[row_start:row_end, col:col+D] = af_flat;          col += D

        y[row_start:row_end] = resid_flat

        if (i + 1) % 500 == 0:
            log.info(f"    {i+1}/{len(index_pairs)} (time, flip) processed")

    return X, y


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build XGBoost feature matrices from ERA5 + AEF data "
                    "(year-based split: 2017-2024 train, 2025 test; with "
                    "heavy-rain augmentation; reduced feature set)."
    )
    parser.add_argument("--era5-dir", required=True,
                        help="Directory with preprocessed ERA5 .npy files")
    parser.add_argument("--aef-dir", required=True,
                        help="Base directory with AEF files (t0/, t1/, ...)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for feature matrices")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable heavy-rain augmentation on training set")
    args = parser.parse_args()

    t_start = time.time()
    era5_dir = Path(args.era5_dir)
    aef_dir = Path(args.aef_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    augment_train = not args.no_augment

    # ── Pair A: 50km → 25km ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Pair A: 50km → 25km")

    A_coarse_up = np.load(era5_dir / "pair_A_coarse_up.npy")
    A_residual  = np.load(era5_dir / "pair_A_residual.npy")
    A_fine      = np.load(era5_dir / "pair_A_fine.npy")
    A_dates     = np.load(era5_dir / "valid_times_A.npy")

    A_train_idx, A_test_idx = split_indices_by_year(A_dates, TRAIN_YEARS, TEST_YEARS)
    log.info(f"  Pair A split  → train={len(A_train_idx)} (years {sorted(TRAIN_YEARS)}), "
             f"test={len(A_test_idx)} (years {sorted(TEST_YEARS)})")
    if len(A_test_idx) == 0:
        log.warning("  Pair A: no 2025 samples in valid_times_A.npy — test set empty")

    H_A, W_A = A_coarse_up.shape[1], A_coarse_up.shape[2]
    years_present_A = sorted(set(int(str(d)[:4]) for d in A_dates))
    log.info(f"  Years present in valid_times_A: {years_present_A}")

    log.info("Loading AEF for Pair A...")
    aef_A = load_aef_by_year(
        aef_dir, years_present_A,
        PAIR_AEF_SCALES["A"]["coarse"],
        PAIR_AEF_SCALES["A"]["fine"],
        H_A, W_A,
    )

    log.info("Building Pair A train features (with augmentation={})".format(augment_train))
    X_train_A, y_train_A = build_features_for_pair(
        "A_train", A_coarse_up, A_residual, A_dates, A_train_idx,
        A_fine, aef_A, augment=augment_train,
    )
    log.info("Building Pair A test features (no augmentation)")
    X_test_A, y_test_A = build_features_for_pair(
        "A_test", A_coarse_up, A_residual, A_dates, A_test_idx,
        A_fine, aef_A, augment=False,
    )

    np.save(out_dir / "A_train_X.npy", X_train_A)
    np.save(out_dir / "A_train_y.npy", y_train_A)
    np.save(out_dir / "A_test_X.npy",  X_test_A)
    np.save(out_dir / "A_test_y.npy",  y_test_A)
    log.info(f"  Saved Pair A: train={X_train_A.shape}, test={X_test_A.shape}")

    del X_train_A, y_train_A, X_test_A, y_test_A
    del A_coarse_up, A_residual, A_fine

    # ── Pair B: 25km → 12.5km ────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Pair B: 25km → 12.5km")

    B_coarse_up = np.load(era5_dir / "pair_B_coarse_up.npy")
    B_residual  = np.load(era5_dir / "pair_B_residual.npy")
    B_fine      = np.load(era5_dir / "pair_B_fine.npy")
    B_dates     = np.load(era5_dir / "valid_times_B.npy")

    B_train_idx, B_test_idx = split_indices_by_year(B_dates, TRAIN_YEARS, TEST_YEARS)
    log.info(f"  Pair B split  → train={len(B_train_idx)}, test={len(B_test_idx)}")

    H_B, W_B = B_coarse_up.shape[1], B_coarse_up.shape[2]
    years_present_B = sorted(set(int(str(d)[:4]) for d in B_dates))
    log.info(f"  Years present in valid_times_B: {years_present_B}")

    log.info("Loading AEF for Pair B...")
    aef_B = load_aef_by_year(
        aef_dir, years_present_B,
        PAIR_AEF_SCALES["B"]["coarse"],
        PAIR_AEF_SCALES["B"]["fine"],
        H_B, W_B,
    )

    log.info("Building Pair B train features (with augmentation={})".format(augment_train))
    X_train_B, y_train_B = build_features_for_pair(
        "B_train", B_coarse_up, B_residual, B_dates, B_train_idx,
        B_fine, aef_B, augment=augment_train,
    )
    log.info("Building Pair B test features (no augmentation)")
    X_test_B, y_test_B = build_features_for_pair(
        "B_test", B_coarse_up, B_residual, B_dates, B_test_idx,
        B_fine, aef_B, augment=False,
    )

    np.save(out_dir / "B_train_X.npy", X_train_B)
    np.save(out_dir / "B_train_y.npy", y_train_B)
    np.save(out_dir / "B_test_X.npy",  X_test_B)
    np.save(out_dir / "B_test_y.npy",  y_test_B)
    log.info(f"  Saved Pair B: train={X_train_B.shape}, test={X_test_B.shape}")

    # Save indices used (these may differ from the .npy files on disk, since
    # we resplit by year)
    np.save(out_dir / "A_train_idx.npy", A_train_idx)
    np.save(out_dir / "A_test_idx.npy",  A_test_idx)
    np.save(out_dir / "B_train_idx.npy", B_train_idx)
    np.save(out_dir / "B_test_idx.npy",  B_test_idx)

    # ── Feature names ────────────────────────────────────────────────────
    D = 64
    feature_names = (
        ["coarse_precip"] +
        [f"aef_coarse_{i}" for i in range(D)] +
        [f"aef_fine_{i}"   for i in range(D)]
    )
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # ── Split info ───────────────────────────────────────────────────────
    split_info = {
        "train_years": sorted(TRAIN_YEARS),
        "test_years":  sorted(TEST_YEARS),
        "augment_train": augment_train,
        "heavy_precip_percentile": HEAVY_PRECIP_PERCENTILE,
        "feature_count": len(feature_names),
        "feature_layout": "[coarse_precip(1), aef_coarse(64), aef_fine(64)]",
        "units": "mm/day (precip and residual scaled by 1000 from m/day)",
        "pair_A": {"train_n_days": int(len(A_train_idx)),
                   "test_n_days":  int(len(A_test_idx))},
        "pair_B": {"train_n_days": int(len(B_train_idx)),
                   "test_n_days":  int(len(B_test_idx))},
    }
    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    log.info("=" * 60)
    log.info(f"Done in {time.time() - t_start:.0f}s")
    log.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()