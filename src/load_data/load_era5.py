"""
Loads native ERA5 25 km grib data and constructs the training pair (based on the spec):
  Pair A: 50 km (coarsened ERA5)  to  25 km (native ERA5)
  Pair B: 25 km (native ERA5) to 12.5 km (upscaled ERA5-Land)

So this script does:
  1. Load era5 grib file to xr.Dataset with (time, latitude, longitude)
  2. Build Pair A:
       a. Coarsen native 25 km field to 50 km via 2x mean pooling.
          This gives us a 50 km field on a half-resolution grid.
       b. Bicubic upsample the 50 km field back up to the 25 km grid.
          This is x_coarse_up: a blurry version of the fine field that
          captures large-scale structure already available at 50 km.
          The model sees this as its input and predicts the residual.
       c. Residual = x_fine - x_coarse_up.
          This is the training target: only the high-frequency detail
          missing from the blurry bicubic upsample.  Modelling the
          residual rather than the full field is much easier because
          its dynamic range is far smaller.
  3. Build Pair B: Bicubic upsample 25 km ERA5 to the 12.5 km grid, 
     resample ERA5-Land to 12.5 km, compute residual.
  4. Stratified 80/20 train/val split by month (no seasonal leakage).
  5. Normalisation stats computed on training split ONLY and saved to
     norm_stats.json.  Applying normalisation to val/test using training
     stats is correct; computing stats on the full dataset would leak
     information from held-out timesteps into training.
  6. Save arrays as .npy files for fast loading during training.

why normalization:
  Raw era5 precipitation values are in kg/m² and extremely right-skewed:
  most values are near zero, rare extremes are very large. Without
  normalisation the model's loss is dominated by scale effects and
  gradients are noisy.  Z-score normalisation (subtract mean, divide by
  std) centres the distribution and makes learning stable.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import cfgrib
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

log = logging.getLogger(__name__)

VARIABLE: str = "tp"


def load_era5_grib(
    grib_path: str | Path,
    year_filter: Optional[int] = None,
) -> xr.Dataset:
    """
    Open an ERA5 GRIB2 file and return an xarray Dataset for total
    precipitation only.

    The GRIB shortname for total precipitation is 'tp'.  We filter
    typeOfLevel='surface' to avoid ambiguity with pressure-level messages
    if the GRIB contains mixed level types.

    Args:
        grib_path:    Path to the .grib file.
        year_filter:  If given (e.g. 2017), only timesteps from that calendar
                      year are returned.  Use this when the GRIB spans multiple
                      years but your auxiliary data (e.g. AEF embeddings) only
                      covers one year.  AEF embeddings are static terrain
                      features, so this filter is purely for scoping the
                      training experiment, not for correctness.

    Returns:
        xr.Dataset with dims (time, latitude, longitude) and data variable
        'tp' in units kg m⁻².  Latitude is descending (north-first).
    """
    grib_path = Path(grib_path)
    if not grib_path.exists():
        raise FileNotFoundError(f"GRIB file not found: {grib_path}")

    log.info("Opening GRIB: %s", grib_path)
    try:
        ds = cfgrib.open_dataset(
            str(grib_path),
            backend_kwargs={
                "filter_by_keys": {"shortName": VARIABLE, "typeOfLevel": "surface"}
            },
            indexpath=None,   # avoid writing .idx files next to read-only data
        )
    except Exception as e:
        raise RuntimeError(
            f"cfgrib failed to open {grib_path}.\n"
            f"Check that the file contains shortName='{VARIABLE}' at typeOfLevel='surface'.\n"
            f"Original error: {e}"
        )

    # Rename cfgrib's internal variable name (often 'tp' already, but may
    # appear as 'unknown' or 'param228.128' depending on the GRIB edition).
    data_vars = list(ds.data_vars)
    if VARIABLE not in data_vars:
        if len(data_vars) == 1:
            ds = ds.rename({data_vars[0]: VARIABLE})
            log.info("  Renamed '%s' → '%s'", data_vars[0], VARIABLE)
        else:
            raise RuntimeError(
                f"Expected a single data variable for '{VARIABLE}', "
                f"got: {data_vars}"
            )

    # Ensure latitude is descending (north-first) for consistent array layout
    if ds.latitude.values[0] < ds.latitude.values[-1]:
        ds = ds.isel(latitude=slice(None, None, -1))

    # Optional year filter — subset to a single calendar year.
    # AEF embeddings are static (terrain / land cover do not change year to
    # year), so filtering ERA5 to 2017 does not create any mismatch with the
    # 2017 AEF snapshot.  This simply scopes the training experiment.
    if year_filter is not None:
        ds = ds.sel(time=ds.time.dt.year == year_filter)
        if ds.dims.get("time", 0) == 0:
            raise RuntimeError(
                f"year_filter={year_filter} matched no timestamps in {grib_path}. "
                f"Check that the GRIB contains data for that year."
            )
        log.info("  Filtered to year %d", year_filter)

    T = ds.dims.get("time", 1)
    H = ds.dims["latitude"]
    W = ds.dims["longitude"]
    log.info("  Loaded tp: T=%d H=%d W=%d", T, H, W)
    return ds[[VARIABLE]]


# Pair A construction: 50 km to 25 km

def build_pair_A(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct Pair A from native 25 km ERA5.

    Steps:
      1. fine_arr  = native 25 km ERA5 field, shape (T, H, W).
      2. Coarsen to 50 km by 2x spatial mean-pooling:
           coarse shape is (T, H//2, W//2).
         This is the simulated 50 km input.
      3. Bicubic upsample the 50 km field back to (T, H, W):
           x_coarse_up is the blurry coarse field on the fine grid.
         This is what the model receives as input.  It contains the
         large-scale structure already present at 50 km, but is smooth
         and missing fine-scale precipitation detail.
      4. residual = fine_arr - x_coarse_up.
         This is the training target: only the high-frequency precipitation
         detail that bicubic upsampling cannot recover from the coarse input.

    Why mean-pool for coarsening?
      Mean-pooling preserves total precipitation mass over the area, which
      is physically consistent and is also the inverse operation used in
      the cycle-consistency loss.

    Why bicubic upsample (not bilinear)?
      Bicubic produces smoother gradients at the boundary of each coarse
      cell and is the standard choice for super-resolution residual models
      (used in both GenBCSR and R2-D2, the spec's cited baselines).

    Returns:
        x_coarse_up  (T, H, W): bicubic-upsampled 50 km field on 25 km grid
        x_fine       (T, H, W): native 25 km ERA5 (ground truth)
        residual     (T, H, W): x_fine - x_coarse_up  (training target)
    """
    fine_arr = ds[VARIABLE].values.astype(np.float32)  # (T, H, W)
    T, H, W  = fine_arr.shape
    log.info("Building Pair A from shape (T=%d, H=%d, W=%d)", T, H, W)

    # Add a channel dimension so PyTorch pooling / interpolation functions work.
    # Shape: (T, H, W) → (T, 1, H, W)
    fine_t = torch.from_numpy(fine_arr).unsqueeze(1)

    # Step 1: 2× mean pool → (T, 1, H//2, W//2)
    # Each output cell is the spatial average of the 2×2 block above it.
    coarse_t = F.avg_pool2d(fine_t, kernel_size=2, stride=2)

    # Step 2: Bicubic upsample back to the original (H, W) fine grid.
    # We pass size=(H, W) explicitly so that odd input dimensions
    # (which can arise when H or W is not divisible by 2) are handled
    # cleanly without an extra crop or resize step.
    x_coarse_up_t = F.interpolate(
        coarse_t,
        size=(H, W),
        mode="bicubic",
        align_corners=False,
    )  # (T, 1, H, W)

    x_coarse_up = x_coarse_up_t.squeeze(1).numpy()  # (T, H, W)

    # Step 3: Residual = fine - coarse_up
    residual = fine_arr - x_coarse_up  # (T, H, W)

    log.info(
        "  tp fine:     min=%.4f  max=%.4f  mean=%.4f",
        fine_arr.min(), fine_arr.max(), fine_arr.mean(),
    )
    log.info(
        "  tp residual: min=%.4f  max=%.4f  mean=%.4f",
        residual.min(), residual.max(), residual.mean(),
    )
    return x_coarse_up, fine_arr, residual


# Train/val split

def stratified_split_by_month(
    times: np.ndarray,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time indices 80/20 (train/val), stratified by calendar month.

    Stratification ensures every month is represented in both splits,
    preventing the model from training on summer-only and validating on
    winter-only (or vice versa), which would make val loss misleading.

    Args:
        times:          Array of np.datetime64 timestamps.
        train_fraction: Fraction to use for training (default 0.8).
        seed:           Random seed for reproducibility.

    Returns:
        (train_indices, val_indices): integer index arrays into *times*
    """
    # Extract month as integer 1–12
    months = np.array([int(str(np.datetime64(t, "M")).split("-")[1]) for t in times])
    rng    = np.random.default_rng(seed)

    train_idx, val_idx = [], []
    for m in range(1, 13):
        m_idx = np.where(months == m)[0]
        if len(m_idx) == 0:
            log.warning("No timestamps found for month %d", m)
            continue
        rng.shuffle(m_idx)
        split = max(1, int(len(m_idx) * train_fraction))
        train_idx.extend(m_idx[:split].tolist())
        val_idx.extend(m_idx[split:].tolist())

    return np.array(sorted(train_idx)), np.array(sorted(val_idx))



# Normalisation

def compute_norm_stats(
    residual_train: np.ndarray,
    fine_train: np.ndarray,
    save_path: Optional[str | Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and std for the residual and the fine field, using only
    the training split.

    We keep two separate sets of stats:
      - "residual": used to normalise the model's training target (r).
      - "fine":     used when we need to work with the full fine field
                    in physical units (e.g. cycle-consistency loss).

    Using ONLY the training split is critical.  If stats were computed
    over the full dataset, the normalisation constants would encode
    information from held-out timesteps, constituting data leakage.

    Args:
        residual_train:  (T_train, H, W)  residuals for training timesteps
        fine_train:      (T_train, H, W)  fine fields for training timesteps
        save_path:       If given, write stats as JSON to this path.

    Returns:
        {"residual": {"mean": float, "std": float},
         "fine":     {"mean": float, "std": float}}
    """
    def _stats(arr: np.ndarray) -> Dict[str, float]:
        flat = arr[np.isfinite(arr)].ravel()
        return {"mean": float(flat.mean()), "std": float(float(flat.std()))}

    stats = {
        "residual": _stats(residual_train),
        "fine":     _stats(fine_train),
    }

    log.info("Normalisation stats (training split only):")
    for key, s in stats.items():
        log.info("  %-10s  mean=%+.6f  std=%.6f", key, s["mean"], s["std"])

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)
        log.info("  Saved → %s", save_path)

    return stats


def load_norm_stats(path: str | Path) -> Dict[str, Dict[str, float]]:
    """Load normalisation stats previously saved by compute_norm_stats."""
    with open(path) as f:
        return json.load(f)


def normalise(arr: np.ndarray, stats: Dict[str, float], eps: float = 1e-8) -> np.ndarray:
    """Apply z-score normalisation:  (x - mean) / (std + eps)."""
    return (arr - stats["mean"]) / (stats["std"] + eps)


def denormalise(arr: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Invert z-score normalisation:  x * std + mean."""
    return arr * stats["std"] + stats["mean"]



def preprocess_and_save(
    era5_grib: str,
    output_dir: str,
    train_fraction: float = 0.8,
    seed: int = 42,
    year_filter: Optional[int] = None,
) -> None:
    """
    Full preprocessing pipeline for Pair A (50 km → 25 km).

    Output layout under output_dir/:
      norm_stats.json         : mean/std for residual and fine field
      pair_A_coarse_up.npy    : (T, H, W)  bicubic-upsampled 50 km on 25 km grid
      pair_A_fine.npy         : (T, H, W)  native 25 km ERA5
      pair_A_residual.npy     : (T, H, W)  = fine - coarse_up  (training target)
      train_indices.npy       : (T_train,) integer time indices
      val_indices.npy         : (T_val,)   integer time indices

    Args:
        era5_grib:       Path to the native 25 km ERA5 GRIB file.
        output_dir:      Where to write all output files.
        train_fraction:  Fraction of timesteps used for training (default 0.8).
        seed:            Random seed for the split.
        year_filter:     If given (e.g. 2017), only that year's data is used.
                         Pass this when your AEF embeddings only cover one year.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load (with optional year filter) ----
    era5  = load_era5_grib(era5_grib, year_filter=year_filter)
    times = era5.time.values  # (T,) np.datetime64

    # ---- Pair A ----
    x_coarse_up, x_fine, residual = build_pair_A(era5)

    # ---- Split ----
    log.info("Splitting %d timesteps …", len(times))
    train_idx, val_idx = stratified_split_by_month(times, train_fraction, seed)
    log.info("  Train=%d  Val=%d", len(train_idx), len(val_idx))

    # ---- Norm stats (training split only) ----
    stats = compute_norm_stats(
        residual_train = residual[train_idx],
        fine_train     = x_fine[train_idx],
        save_path      = out / "norm_stats.json",
    )

    # ---- Save arrays ----
    log.info("Saving arrays to %s …", out)

    # Save the ERA5 lat/lon coordinate arrays so that load_aef.py can build
    # AEF pooling grids that are exactly aligned with this ERA5 grid — rather
    # than approximating from hardcoded degree steps.
    era5_lats = era5.latitude.values.astype(np.float64)   # (H,) descending
    era5_lons = era5.longitude.values.astype(np.float64)  # (W,) ascending

    for name, arr in [
        ("pair_A_coarse_up", x_coarse_up),
        ("pair_A_fine",      x_fine),
        ("pair_A_residual",  residual),
        ("train_indices",    train_idx),
        ("val_indices",      val_idx),
        ("era5_lats",        era5_lats),
        ("era5_lons",        era5_lons),
    ]:
        np.save(out / f"{name}.npy", arr)
        log.info("  %-22s  shape=%-20s  dtype=%s", name, str(arr.shape), arr.dtype)

    log.info("Done → %s", out)

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Preprocess ERA5 total precipitation for U-Net downscaling training."
    )
    parser.add_argument(
        "--era5", required=True,
        help="Path to the native 25 km ERA5 GRIB file (must contain shortName='tp')."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory for preprocessed .npy files and norm_stats.json."
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.8,
        help="Fraction of time steps for training (default: 0.8)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/val split (default: 42)."
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Filter ERA5 to a single year (e.g. 2017). Use this when your "
             "AEF embeddings only cover one year.  Omit to use all years in the GRIB."
    )
    args = parser.parse_args()

    preprocess_and_save(
        era5_grib      = args.era5,
        output_dir     = args.output,
        train_fraction = args.train_fraction,
        seed           = args.seed,
        year_filter    = args.year,
    )

# run this script with:
# python load_era5.py \
#     --era5 /projects/bgua/CS598_G7_project/data/era5.grib \
#     --output /projects/bgua/CS598_G7_project/data/processed \
#     --year 2017