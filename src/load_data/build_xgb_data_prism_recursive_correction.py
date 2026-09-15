#!/usr/bin/env python3
"""
Build XGBoost feature matrices for the final recursive-step correction baseline.

This is the corrected PRISM-supervised setup:

    fine target = PRISM 800 m coarsened / regridded to 1.5625 km
    input       = 1.5625 km precipitation produced by the recursive ensemble
    target y    = fine target - input

So the model learns a residual correction for the final recursive output. In the
best scenario, recursive input already matches PRISM_1.5625 and the residual is 0.

Typical usage:

    python -u build_xgb_data_prism_recursive_correction.py \
        --aef-dir data/aef_downsampled_by_year \
        --prism-dir data/prism_processed \
        --recursive-input results/xgb_recursive_1p5625.npy \
        --output-dir data/xgb_features_prism_recursive_correction \
        --train-end-year 2024 \
        --test-year 2025

If recursive-input is an ensemble with shape (T, E, H, W) or (E, T, H, W), the
script uses ensemble mean as the correction input by default. Optionally pass
--add-ensemble-std to include ensemble spread as one extra feature.

Expected PRISM files in --prism-dir, with flexible names:
    precipitation: prism_800m.npy, prism_native.npy, prism_daily.npy, prism.npy,
                   prism_regridded.npy, or prism_1p5625.npy
    dates:         prism_dates.npy, valid_times_prism.npy, valid_times_P.npy, dates.npy
    lats/lons:     prism_lats.npy/prism_lons.npy or lats.npy/lons.npy

If PRISM is already on the 1.5625 km grid, pass --prism-already-1p5625.
If PRISM is mm/day while recursive input is m/day, pass --prism-scale 0.001.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

AEF_YEAR_OFFSET = 2017


def fmt_res(res_km: float) -> str:
    return str(int(res_km)) if float(res_km).is_integer() else str(res_km)


def date_to_year(d) -> int:
    return int(str(d)[:4])


def day_of_year(d) -> int:
    dt = np.datetime64(d, "D")
    y0 = np.datetime64(f"{str(dt)[:4]}-01-01", "D")
    return int((dt - y0).astype(int)) + 1


def require_existing(candidates: Sequence[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find {label}. Tried: {[str(p) for p in candidates]}")


def optional_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def load_aef_nc(nc_path: Path) -> np.ndarray:
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32)  # (D,H,W)
    ds.close()
    return arr.transpose(1, 2, 0)  # (H,W,D)


def find_aef_file(aef_dir: Path, t_idx: int, res_km: float) -> Path:
    r1, r2 = fmt_res(res_km), str(res_km)
    candidates = [
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{r1}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{r1}km.nc",
        aef_dir / f"aef_illinois_t{t_idx}_{r1}km.nc",
        aef_dir / f"aef_illinois_{r1}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{r2}km.nc",
        aef_dir / f"t{t_idx}" / f"aef_illinois_{r2}km.nc",
        aef_dir / f"aef_illinois_t{t_idx}_{r2}km.nc",
        aef_dir / f"aef_illinois_{r2}km.nc",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"AEF not found for t{t_idx} at {res_km} km")


def resize_2d(arr: np.ndarray, target_h: int, target_w: int, order: int = 1) -> np.ndarray:
    if arr.shape == (target_h, target_w):
        return arr.astype(np.float32, copy=False)
    return zoom(arr, (target_h / arr.shape[0], target_w / arr.shape[1]), order=order).astype(np.float32)


def resize_hwd(arr: np.ndarray, target_h: int, target_w: int, order: int = 1) -> np.ndarray:
    if arr.ndim == 2:
        return resize_2d(arr, target_h, target_w, order=order)
    if arr.ndim == 3:
        if arr.shape[:2] == (target_h, target_w):
            return arr.astype(np.float32, copy=False)
        return np.stack(
            [resize_2d(arr[:, :, d], target_h, target_w, order=order) for d in range(arr.shape[2])],
            axis=-1,
        ).astype(np.float32)
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def load_aef_by_year(
    aef_dir: Path,
    years: Iterable[int],
    coarse_res: float,
    fine_res: float,
    target_h: int,
    target_w: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for year in sorted(set(int(y) for y in years)):
        t_idx = year - AEF_YEAR_OFFSET
        try:
            cp = find_aef_file(aef_dir, t_idx, coarse_res)
            fp = find_aef_file(aef_dir, t_idx, fine_res)
        except FileNotFoundError as e:
            log.warning("  %s", e)
            continue
        ac = resize_hwd(load_aef_nc(cp), target_h, target_w, order=1)
        af = resize_hwd(load_aef_nc(fp), target_h, target_w, order=1)
        d = ac.shape[2]
        out[year] = (ac.reshape(-1, d), af.reshape(-1, d))
        log.info(
            "  AEF %s: coarse %.4g km -> %s, fine %.4g km -> %s",
            year, coarse_res, ac.shape, fine_res, af.shape,
        )
    if not out:
        raise RuntimeError(f"No AEF loaded for coarse={coarse_res}, fine={fine_res}")
    return out


def split_indices_by_year(dates: np.ndarray, train_end_year: int, test_year: int) -> Tuple[np.ndarray, np.ndarray]:
    years = np.array([date_to_year(d) for d in dates])
    train = np.where((years <= train_end_year) & (years != test_year))[0].astype(np.int64)
    test = np.where(years == test_year)[0].astype(np.int64)
    if len(train) == 0:
        raise RuntimeError("Year split produced zero training timesteps")
    if len(test) == 0:
        raise RuntimeError(f"Year split produced zero test timesteps for {test_year}")
    return train, test


def load_prism_inputs(prism_dir: Path):
    data_path = require_existing([
        prism_dir / "prism_800m.npy",
        prism_dir / "prism_native.npy",
        prism_dir / "prism_daily.npy",
        prism_dir / "prism.npy",
        prism_dir / "prism_regridded.npy",
        prism_dir / "prism_1p5625.npy",
    ], "PRISM precipitation array")
    dates_path = require_existing([
        prism_dir / "prism_dates.npy",
        prism_dir / "valid_times_prism.npy",
        prism_dir / "valid_times_P.npy",
        prism_dir / "dates.npy",
    ], "PRISM dates")
    lats_path = require_existing([
        prism_dir / "prism_lats.npy",
        prism_dir / "lats_1p5625.npy",
        prism_dir / "lats.npy",
    ], "PRISM latitudes")
    lons_path = require_existing([
        prism_dir / "prism_lons.npy",
        prism_dir / "lons_1p5625.npy",
        prism_dir / "lons.npy",
    ], "PRISM longitudes")
    return data_path, dates_path, lats_path, lons_path


def load_prism_target_1p5625(
    prism_dir: Path,
    prism_scale: float,
    prism_source_res_km: float,
    fine_res_km: float,
    prism_already_1p5625: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_path, dates_path, lats_path, lons_path = load_prism_inputs(prism_dir)
    prism = np.load(data_path, mmap_mode="r")
    dates = np.load(dates_path)
    src_lats = np.load(lats_path)
    src_lons = np.load(lons_path)

    if prism.ndim != 3:
        raise ValueError(f"Expected PRISM shape (T,H,W), got {prism.shape}")

    if prism_already_1p5625:
        h_f, w_f = int(prism.shape[1]), int(prism.shape[2])
    else:
        h_f = max(2, int(round(prism.shape[1] * prism_source_res_km / fine_res_km)))
        w_f = max(2, int(round(prism.shape[2] * prism_source_res_km / fine_res_km)))

    target = np.empty((prism.shape[0], h_f, w_f), dtype=np.float32)
    log.info("PRISM target: source=%s, source shape=%s, target 1.5625km grid=%dx%d", data_path.name, prism.shape, h_f, w_f)
    for i in range(prism.shape[0]):
        src = np.asarray(prism[i], dtype=np.float32) * prism_scale
        src = np.clip(src, 0.0, None)
        target[i] = src if prism_already_1p5625 else resize_2d(src, h_f, w_f, order=1)
        if (i + 1) % 250 == 0:
            log.info("  built PRISM target %d/%d", i + 1, prism.shape[0])

    lats = np.linspace(float(src_lats[0]), float(src_lats[-1]), h_f).astype(np.float32)
    lons = np.linspace(float(src_lons[0]), float(src_lons[-1]), w_f).astype(np.float32)
    return target, dates, lats, lons


def load_recursive_input(
    recursive_path: Path,
    target_shape: Tuple[int, int, int],
    ensemble_axis: str,
    add_ensemble_std: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.load(recursive_path, mmap_mode="r")
    log.info("Recursive input: %s shape=%s", recursive_path, arr.shape)

    if arr.ndim == 3:
        rec = np.asarray(arr, dtype=np.float32)
        spread = None
    elif arr.ndim == 4:
        # Support either (T,E,H,W) or (E,T,H,W). Auto-detect by target T if requested.
        T_target = target_shape[0]
        if ensemble_axis == "auto":
            if arr.shape[0] == T_target:
                ensemble_axis = "1"  # (T,E,H,W)
            elif arr.shape[1] == T_target:
                ensemble_axis = "0"  # (E,T,H,W)
            else:
                raise ValueError(
                    f"Cannot auto-detect ensemble axis for recursive input shape {arr.shape} and target T={T_target}. "
                    "Pass --ensemble-axis 0 or --ensemble-axis 1."
                )
        axis = int(ensemble_axis)
        rec = np.asarray(arr.mean(axis=axis), dtype=np.float32)
        spread = np.asarray(arr.std(axis=axis), dtype=np.float32) if add_ensemble_std else None
        log.info("  using ensemble mean over axis %d -> %s", axis, rec.shape)
    else:
        raise ValueError(f"Expected recursive input shape (T,H,W), (T,E,H,W), or (E,T,H,W), got {arr.shape}")

    if rec.shape[0] != target_shape[0]:
        raise ValueError(f"Recursive input T={rec.shape[0]} does not match PRISM target T={target_shape[0]}")

    h, w = target_shape[1], target_shape[2]
    if rec.shape[1:] != (h, w):
        log.info("  resizing recursive input from %s to %s", rec.shape[1:], (h, w))
        rec_resized = np.empty(target_shape, dtype=np.float32)
        spread_resized = np.empty(target_shape, dtype=np.float32) if spread is not None else None
        for i in range(rec.shape[0]):
            rec_resized[i] = resize_2d(np.asarray(rec[i], dtype=np.float32), h, w, order=1)
            if spread is not None:
                spread_resized[i] = resize_2d(np.asarray(spread[i], dtype=np.float32), h, w, order=1)
        rec, spread = rec_resized, spread_resized

    return np.clip(rec, 0.0, None).astype(np.float32), spread


def build_features_for_correction(
    name: str,
    recursive_input: np.ndarray,
    residual: np.ndarray,
    dates: np.ndarray,
    indices: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    aef_by_year: Dict[int, Tuple[np.ndarray, np.ndarray]],
    use_log_precip: bool,
    ensemble_std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w = recursive_input.shape[1:]
    n_pix = h * w
    n = len(indices) * n_pix
    fallback_year = sorted(aef_by_year.keys())[0]
    d = aef_by_year[fallback_year][0].shape[1]
    extra = 1 if ensemble_std is not None else 0
    n_features = 1 + extra + d + d + 2 + 2
    log.info("  %s: %d timesteps x %d pixels = %s rows, %d features", name, len(indices), n_pix, f"{n:,}", n_features)

    X = np.empty((n, n_features), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)

    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    lat_flat = lat_grid.ravel().astype(np.float32)
    lon_flat = lon_grid.ravel().astype(np.float32)
    lat_norm = (lat_flat - lat_flat.min()) / (lat_flat.max() - lat_flat.min() + 1e-8)
    lon_norm = (lon_flat - lon_flat.min()) / (lon_flat.max() - lon_flat.min() + 1e-8)

    for i, t_raw in enumerate(indices):
        t = int(t_raw)
        lo, hi = i * n_pix, (i + 1) * n_pix

        rec_flat = recursive_input[t].ravel().astype(np.float32)
        if use_log_precip:
            rec_flat = np.log1p(np.clip(rec_flat, 0.0, None)).astype(np.float32)

        year = date_to_year(dates[t])
        aef_c, aef_f = aef_by_year.get(year, aef_by_year[fallback_year])
        doy = day_of_year(dates[t])
        sin_doy = np.float32(np.sin(2 * np.pi * doy / 365.25))
        cos_doy = np.float32(np.cos(2 * np.pi * doy / 365.25))

        col = 0
        X[lo:hi, col] = rec_flat
        col += 1
        if ensemble_std is not None:
            std_flat = ensemble_std[t].ravel().astype(np.float32)
            if use_log_precip:
                std_flat = np.log1p(np.clip(std_flat, 0.0, None)).astype(np.float32)
            X[lo:hi, col] = std_flat
            col += 1
        X[lo:hi, col:col + d] = aef_c
        col += d
        X[lo:hi, col:col + d] = aef_f
        col += d
        X[lo:hi, col] = lat_norm
        col += 1
        X[lo:hi, col] = lon_norm
        col += 1
        X[lo:hi, col] = sin_doy
        col += 1
        X[lo:hi, col] = cos_doy

        y[lo:hi] = residual[t].ravel().astype(np.float32)

    return X, y


def main():
    parser = argparse.ArgumentParser(description="Build XGBoost data for PRISM correction of final recursive 1.5625km output")
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--prism-dir", required=True)
    parser.add_argument("--recursive-input", required=True, help=".npy file for recursive 1.5625km output, shape (T,H,W), (T,E,H,W), or (E,T,H,W)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-end-year", type=int, default=2024)
    parser.add_argument("--test-year", type=int, default=2025)
    parser.add_argument("--no-log-precip", action="store_true")
    parser.add_argument("--prism-scale", type=float, default=1.0, help="Use 0.001 if PRISM is mm/day and recursive input is m/day")
    parser.add_argument("--recursive-scale", type=float, default=1.0, help="Scale recursive input to match PRISM units")
    parser.add_argument("--prism-source-res-km", type=float, default=0.8)
    parser.add_argument("--prism-already-1p5625", action="store_true")
    parser.add_argument("--fine-res-km", type=float, default=1.5625)
    parser.add_argument("--ensemble-axis", choices=["auto", "0", "1"], default="auto")
    parser.add_argument("--add-ensemble-std", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    aef_dir = Path(args.aef_dir)
    prism_dir = Path(args.prism_dir)
    out_dir = Path(args.output_dir)
    recursive_path = Path(args.recursive_input)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_log = not args.no_log_precip

    log.info("=" * 70)
    log.info("PRISM correction pair: recursive 1.5625km input -> PRISM_1.5625km target")

    prism_target, dates, lats, lons = load_prism_target_1p5625(
        prism_dir=prism_dir,
        prism_scale=args.prism_scale,
        prism_source_res_km=args.prism_source_res_km,
        fine_res_km=args.fine_res_km,
        prism_already_1p5625=args.prism_already_1p5625,
    )

    recursive_input, ensemble_std = load_recursive_input(
        recursive_path=recursive_path,
        target_shape=prism_target.shape,
        ensemble_axis=args.ensemble_axis,
        add_ensemble_std=args.add_ensemble_std,
    )
    recursive_input = (recursive_input * args.recursive_scale).astype(np.float32)
    if ensemble_std is not None:
        ensemble_std = (ensemble_std * args.recursive_scale).astype(np.float32)

    residual = (prism_target - recursive_input).astype(np.float32)
    train_idx, test_idx = split_indices_by_year(dates, args.train_end_year, args.test_year)

    aef = load_aef_by_year(
        aef_dir,
        [date_to_year(d) for d in dates],
        coarse_res=3.125,
        fine_res=1.5625,
        target_h=recursive_input.shape[1],
        target_w=recursive_input.shape[2],
    )

    X_train, y_train = build_features_for_correction(
        "P_correction_train",
        recursive_input,
        residual,
        dates,
        train_idx,
        lats,
        lons,
        aef,
        use_log,
        ensemble_std,
    )
    X_test, y_test = build_features_for_correction(
        "P_correction_test",
        recursive_input,
        residual,
        dates,
        test_idx,
        lats,
        lons,
        aef,
        use_log,
        ensemble_std,
    )

    # Main outputs for this corrected phase.
    np.save(out_dir / "P_train_X.npy", X_train)
    np.save(out_dir / "P_train_y.npy", y_train)
    np.save(out_dir / "P_test_X.npy", X_test)
    np.save(out_dir / "P_test_y.npy", y_test)
    np.save(out_dir / "P_train_indices.npy", train_idx)
    np.save(out_dir / "P_test_indices.npy", test_idx)

    # Also save phase2 aliases so the existing training script can point to these.
    np.save(out_dir / "phase2_train_X.npy", X_train)
    np.save(out_dir / "phase2_train_y.npy", y_train)
    np.save(out_dir / "phase2_test_X.npy", X_test)
    np.save(out_dir / "phase2_test_y.npy", y_test)

    # Save fields for reconstruction / evaluation:
    # corrected_prediction = recursive_input + xgb_predicted_residual.
    np.save(out_dir / "P_recursive_input.npy", recursive_input)
    np.save(out_dir / "P_prism_target_1p5625.npy", prism_target)
    np.save(out_dir / "P_target_residual.npy", residual)
    np.save(out_dir / "P_dates.npy", dates)
    np.save(out_dir / "P_lats.npy", lats)
    np.save(out_dir / "P_lons.npy", lons)
    if ensemble_std is not None:
        np.save(out_dir / "P_recursive_ensemble_std.npy", ensemble_std)

    d = aef[sorted(aef.keys())[0]][0].shape[1]
    feature_names = ["log1p_recursive_input" if use_log else "recursive_input"]
    if ensemble_std is not None:
        feature_names.append("log1p_recursive_ensemble_std" if use_log else "recursive_ensemble_std")
    feature_names += [f"aef_coarse_3p125_{i}" for i in range(d)]
    feature_names += [f"aef_fine_1p5625_{i}" for i in range(d)]
    feature_names += ["lat", "lon", "sin_doy", "cos_doy"]
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    metadata = {
        "corrected_setup": "target residual = PRISM_1.5625km - recursive_ensemble_1.5625km_input",
        "best_case_residual": 0.0,
        "train_years": sorted(set(date_to_year(dates[int(i)]) for i in train_idx)),
        "test_years": sorted(set(date_to_year(dates[int(i)]) for i in test_idx)),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "grid_shape": [int(recursive_input.shape[1]), int(recursive_input.shape[2])],
        "use_log_precip": use_log,
        "prism_scale": args.prism_scale,
        "recursive_scale": args.recursive_scale,
        "recursive_input_file": str(recursive_path),
        "add_ensemble_std": bool(args.add_ensemble_std),
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("Saved corrected PRISM recursive-correction XGBoost data to %s", out_dir)
    log.info("Train X/y: %s / %s", X_train.shape, y_train.shape)
    log.info("Test  X/y: %s / %s", X_test.shape, y_test.shape)
    log.info("Done in %.0fs", time.time() - t0)


if __name__ == "__main__":
    main()
