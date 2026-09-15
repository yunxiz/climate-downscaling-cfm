"""
Loads pre-pooled AEF (AlphaEarth Foundation) embeddings and produces
dual-resolution conditioning tensors at every ERA5 grid scale.

This reads the pre-computed GeM-pooled NetCDF files (aef_illinois_t{T}_{R}km.nc) 
and aligns them to the ERA5 grid.

The spec (Section 2.2) requires two AEF tensors at each 2x step:
  alpha_coarse: AEF pooled to the INPUT  (coarse) resolution
  alpha_fine:   AEF pooled to the OUTPUT (fine)   resolution

Both enter the model simultaneously via dual-resolution cross-attention.

Output files (under the preprocessing output dir):
  aef_alpha_50km.npy    (H50,  W50,  D)
  aef_alpha_25km.npy    (H25,  W25,  D)
  aef_alpha_12km.npy    (H12,  W12,  D)
  aef_alpha_6km.npy     (H6,   W6,   D)
  aef_alpha_3km.npy     (H3,   W3,   D)
  aef_alpha_1.5km.npy   (H1.5, W1.5, D)
  
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr

log = logging.getLogger(__name__)

# The embedding dimension of AEF
AEF_EMBED_DIM: int = 64

# q exponent for GeM pooling (used only for minor grid alignment)
GEM_Q: float = 3.0

# Mapping from output scale names to pre-pooled file resolution suffixes
SCALE_TO_FILE_RES = {
    "50km":  50.0,
    "25km":  25.0,
    "12km":  12.5,
    "6km":   6.25,
    "3km":   3.125,
    "1.5km": 1.5625,
}


def _load_prepooled_nc(nc_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a pre-pooled AEF NetCDF file.

    Expected structure:
      - embeddings: (band, y, x) int8
      - y: latitude coordinate
      - x: longitude coordinate

    Returns:
        embeddings: (H, W, D) float32
        lats: (H,) latitude array
        lons: (W,) longitude array
    """
    ds = xr.open_dataset(nc_path, engine="netcdf4")

    arr = ds["embeddings"].values  # (band, y, x) int8
    arr = arr.astype(np.float32)

    # Transpose from (D, H, W) to (H, W, D)
    arr = arr.transpose(1, 2, 0)

    lats = ds["y"].values.astype(np.float64)
    lons = ds["x"].values.astype(np.float64)

    # Ensure north-first (descending latitude)
    if len(lats) > 1 and lats[0] < lats[-1]:
        arr = arr[::-1, :, :].copy()
        lats = lats[::-1]

    ds.close()
    return arr, lats, lons


def gem_pool_numpy(
    embeddings: np.ndarray,
    target_h: int,
    target_w: int,
    q: float = GEM_Q,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Apply Generalised Mean Pooling to resize an embedding grid.

    Used only for minor grid alignment (pre-pooled resolution may not
    exactly match ERA5 grid dimensions).

    Args:
        embeddings: (H_src, W_src, D) float32 array
        target_h:   Height of the output grid
        target_w:   Width of the output grid
        q:          GeM exponent
        eps:        Numerical stability

    Returns:
        (target_h, target_w, D) float32 array
    """
    H, W, D = embeddings.shape

    if H == target_h and W == target_w:
        return embeddings

    # Convert to (1, D, H, W) for torch pooling
    t = torch.from_numpy(embeddings.transpose(2, 0, 1)).unsqueeze(0).float()

    # Handle NoData: replace -128 with 0 before pooling
    nodata_mask = (t == -128)
    t = torch.where(nodata_mask, torch.zeros_like(t), t)

    # Clamp to avoid numerical issues
    t = t.clamp(min=eps)

    # GeM: power → pool → un-power
    t_q = t.pow(q)
    t_pool = F.adaptive_avg_pool2d(t_q, output_size=(target_h, target_w))
    t_gem = t_pool.pow(1.0 / q)

    return t_gem.squeeze(0).permute(1, 2, 0).numpy()


def _derive_grid_at_scale(
    base_lats: np.ndarray,
    base_lons: np.ndarray,
    scale_factor: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive a lat/lon grid at a different resolution by resampling the
    base 25 km ERA5 grid by scale_factor.
    """
    H_base = len(base_lats)
    W_base = len(base_lons)

    H_new = max(1, round(H_base / scale_factor))
    W_new = max(1, round(W_base / scale_factor))

    lats = np.linspace(base_lats[0], base_lats[-1], H_new)
    lons = np.linspace(base_lons[0], base_lons[-1], W_new)
    return lats, lons


# ── Main public API (same signatures as original) ───────────────────────────

def load_aef_nc(
    nc_path: str | Path,
    embed_var: Optional[str] = None,
    lat_var: str = "y",
    lon_var: str = "x",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Open a pre-pooled AEF NetCDF file and return embeddings + coordinates.

    API-compatible with the original load_aef.load_aef_nc().

    Returns:
        embeddings: (H, W, D) float32 array  (lat x lon x embed_dim)
        lats:       (H,) descending latitude array
        lons:       (W,) ascending longitude array
    """
    return _load_prepooled_nc(Path(nc_path))


def pool_aef_multiscale(
    aef_prepooled_dir: str,
    era5_preprocessed_dir: str,
    output_dir: str,
    q: float = GEM_Q,
    time_index: int = 0,
    embed_var: Optional[str] = None,
) -> None:
    """
    Load pre-pooled AEF embeddings at each resolution and align them to
    the ERA5 grid, saving .npy files in the same format as the original.

    API-compatible with the original load_aef.pool_aef_multiscale().

    Args:
        aef_prepooled_dir:     Directory containing pre-pooled files, e.g.
                               aef_illinois_t0_50km.nc, aef_illinois_t0_25km.nc, etc.
                               Or: aef_illinois_50km.nc (without time prefix).
        era5_preprocessed_dir: Directory with era5_lats.npy and era5_lons.npy.
        output_dir:            Where to write pooled AEF .npy files.
        q:                     GeM exponent for grid alignment (default 3.0).
        time_index:            Time index for file naming (default 0).
        embed_var:             Unused, kept for API compatibility.
    """
    out = Path(output_dir)
    prep_dir = Path(era5_preprocessed_dir)
    aef_dir = Path(aef_prepooled_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load ERA5 25 km grid
    lats_25km = np.load(prep_dir / "era5_lats.npy")
    lons_25km = np.load(prep_dir / "era5_lons.npy")
    log.info(
        "ERA5 25 km grid: H=%d W=%d  lat=[%.3f, %.3f]  lon=[%.3f, %.3f]",
        len(lats_25km), len(lons_25km),
        lats_25km[0], lats_25km[-1],
        lons_25km[0], lons_25km[-1],
    )

    # Scale factors relative to 25 km
    scales = {
        "50km":  2.0,
        "25km":  1.0,
        "12km":  0.5,
        "6km":   0.25,
        "3km":   0.125,
        "1.5km": 0.0625,
    }

    for scale_name, scale_factor in scales.items():
        file_res = SCALE_TO_FILE_RES[scale_name]

        # Try file naming conventions:
        # 1. aef_illinois_t{T}_{R}km.nc  (multi-year format)
        # 2. aef_illinois_{R}km.nc       (single-year format)
        candidates = [
            aef_dir / f"aef_illinois_t{time_index}_{file_res}km.nc",
            aef_dir / f"aef_illinois_{file_res}km.nc",
        ]
        nc_path = None
        for c in candidates:
            if c.exists():
                nc_path = c
                break

        if nc_path is None:
            raise FileNotFoundError(
                f"No pre-pooled file found for {scale_name} ({file_res} km). "
                f"Tried: {[str(c) for c in candidates]}"
            )

        log.info("Loading pre-pooled AEF for %s from %s", scale_name, nc_path.name)
        embeddings, aef_lats, aef_lons = _load_prepooled_nc(nc_path)
        log.info("  Shape: %s (H=%d, W=%d, D=%d)", embeddings.shape, *embeddings.shape)

        # Determine target ERA5 grid dimensions at this scale
        target_lats, target_lons = _derive_grid_at_scale(
            lats_25km, lons_25km, scale_factor
        )
        target_h = len(target_lats)
        target_w = len(target_lons)

        # If pre-pooled dimensions don't exactly match ERA5 grid, do a
        # light GeM resize to align. This is a minor adjustment (e.g.
        # 13→12 pixels) since the pre-pooled files are already at
        # approximately the right resolution.
        if embeddings.shape[0] != target_h or embeddings.shape[1] != target_w:
            log.info(
                "  Grid alignment: (%d, %d) → (%d, %d)",
                embeddings.shape[0], embeddings.shape[1],
                target_h, target_w,
            )
            embeddings = gem_pool_numpy(embeddings, target_h, target_w, q=q)

        out_path = out / f"aef_alpha_{scale_name}.npy"
        np.save(out_path, embeddings.astype(np.float32))
        log.info("  → %s  shape=%s", out_path.name, embeddings.shape)

    log.info("AEF multi-scale loading complete → %s", out)


def load_aef_pair(
    aef_dir: str,
    coarse_scale: str,
    fine_scale: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience loader for (alpha_coarse, alpha_fine) at a given resolution pair.

    API-compatible with the original load_aef.load_aef_pair().

    Args:
        aef_dir: Directory containing aef_alpha_*.npy files.
        coarse_scale: e.g. "25km"
        fine_scale:   e.g. "12km"

    Returns:
        alpha_coarse: (H_c, W_c, D)
        alpha_fine:   (H_f, W_f, D)
    """
    aef_dir = Path(aef_dir)
    alpha_coarse = np.load(aef_dir / f"aef_alpha_{coarse_scale}.npy")
    alpha_fine = np.load(aef_dir / f"aef_alpha_{fine_scale}.npy")
    return alpha_coarse, alpha_fine


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        description="Load pre-pooled AEF embeddings and align to ERA5 grid"
    )
    parser.add_argument("--aef-dir", required=True,
                        help="Directory with pre-pooled aef_illinois_*km.nc files")
    parser.add_argument("--era5-dir", required=True,
                        help="ERA5 preprocessed directory (with era5_lats/lons.npy)")
    parser.add_argument("--output", required=True,
                        help="Output directory for aligned AEF .npy files")
    parser.add_argument("--time-index", type=int, default=0,
                        help="Time index (for multi-year file naming)")
    parser.add_argument("--gem-q", type=float, default=3.0,
                        help="GeM exponent for grid alignment (default 3.0)")
    args = parser.parse_args()

    pool_aef_multiscale(
        aef_prepooled_dir=args.aef_dir,
        era5_preprocessed_dir=args.era5_dir,
        output_dir=args.output,
        q=args.gem_q,
        time_index=args.time_index,
    )