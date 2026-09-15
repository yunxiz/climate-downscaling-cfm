"""
PyTorch Dataset for ERA5 downscaling with per-year AEF embeddings.

Supports both Pair A (50 km → 25 km) and Pair B (25 km → 12.5 km).
AEF embeddings vary by year — each sample loads the AEF for its calendar year.

Returns batches of:
  x_coarse_up:  bicubic-upsampled coarse field on the fine grid  (1, H, W) [raw, unnormalized]
  residual:     fine - x_coarse_up  (the training target)        (1, H, W) [raw, unnormalized]
  x_fine:       native fine-resolution field                     (1, H, W) [raw, unnormalized]
  alpha_coarse: AEF embeddings at the coarse resolution          (D, H, W)
  alpha_fine:   AEF embeddings at the fine resolution            (D, H, W)
  pair_id:      0 for Pair A, 1 for Pair B                       scalar

NOTE: Fields are returned RAW (unnormalized) because:
  - The CFM training loop constructs flow interpolants r_t = (1-t)*z + t*residual,
    which requires the raw residual values
  - x_coarse_up is concatenated with r_t as model input, so it must be in the
    same scale
  - Normalization (if needed) should be applied inside the training loop after
    flow interpolation, not in the dataset

NOTE: Pair A (23x17) and Pair B (45x33) have different grid sizes and CANNOT
  be batched together. Use build_paired_dataloaders() which returns separate
  loaders that are alternated during training.

Usage:
    from dataset import build_datasets, build_paired_dataloaders
    datasets = build_datasets("data/processed", "data/aef")
    loaders = build_paired_dataloaders(datasets, batch_size=32)
    for batch_A, batch_B in zip(loaders["train_A"], loaders["train_B"]):
        ...
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

log = logging.getLogger(__name__)

# AEF year-to-index mapping: AEF indices 0-8 correspond to 2017-2025
AEF_YEAR_OFFSET = 2017

# Resolution scale names for AEF file lookup
PAIR_AEF_SCALES = {
    "A": {"coarse": 50.0, "fine": 25.0},
    "B": {"coarse": 25.0, "fine": 12.5},
}


def _load_aef_nc(nc_path: Path) -> np.ndarray:
    """Load a pre-pooled AEF NetCDF → (H, W, D) float32."""
    ds = xr.open_dataset(nc_path, engine="netcdf4")
    arr = ds["embeddings"].values.astype(np.float32)  # (D, H, W)
    ds.close()
    return arr.transpose(1, 2, 0)


def _format_res(res_km: float) -> str:
    """Format resolution for filename: 50.0 → '50', 12.5 → '12.5', 1.5625 → '1.5625'."""
    if res_km == int(res_km):
        return str(int(res_km))
    return str(res_km)


def _load_aef_for_years(
    aef_base_dir: Path,
    years: list,
    coarse_res: float,
    fine_res: float,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Load AEF for multiple years → dict: year → (coarse, fine), each (H, W, D)."""
    coarse_str = _format_res(coarse_res)
    fine_str = _format_res(fine_res)

    aef_by_year = {}
    for year in years:
        t_idx = year - AEF_YEAR_OFFSET
        candidates_coarse = [
            aef_base_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{coarse_str}km.nc",
            aef_base_dir / f"t{t_idx}" / f"aef_illinois_{coarse_str}km.nc",
            aef_base_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{coarse_res}km.nc",
            aef_base_dir / f"t{t_idx}" / f"aef_illinois_{coarse_res}km.nc",
        ]
        candidates_fine = [
            aef_base_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{fine_str}km.nc",
            aef_base_dir / f"t{t_idx}" / f"aef_illinois_{fine_str}km.nc",
            aef_base_dir / f"t{t_idx}" / f"aef_illinois_t{t_idx}_{fine_res}km.nc",
            aef_base_dir / f"t{t_idx}" / f"aef_illinois_{fine_res}km.nc",
        ]

        coarse_path = next((c for c in candidates_coarse if c.exists()), None)
        fine_path = next((c for c in candidates_fine if c.exists()), None)

        if coarse_path is None or fine_path is None:
            log.warning(f"  AEF not found for year {year} (t{t_idx}), skipping")
            log.warning(f"    Tried coarse: {[str(c) for c in candidates_coarse[:2]]}")
            log.warning(f"    Tried fine:   {[str(c) for c in candidates_fine[:2]]}")
            continue

        aef_by_year[year] = (_load_aef_nc(coarse_path), _load_aef_nc(fine_path))
        log.info(f"  Year {year} (t{t_idx}): coarse={aef_by_year[year][0].shape}, "
                 f"fine={aef_by_year[year][1].shape}")

    return aef_by_year


class PairDataset(Dataset):
    """
    Dataset for one resolution pair (A or B) with per-year AEF conditioning.

    Data augmentation (training only):
      - Heavy precipitation oversampling: days above the 60th percentile of
        domain-mean precipitation get 4 deterministic copies (original + 3 flips).
      - Light-rain days are kept as-is (original only).

    All precipitation fields are returned in mm/day (×1000 from meters).
    AEF embeddings are interpolated to the fine grid for uniform batching.

    Fields returned:
      - x_coarse: original coarse-resolution field (for cycle loss target)
      - x_coarse_up: bicubic-upsampled coarse field (model input)
      - residual: fine - coarse_up (flow matching target)
      - x_fine: fine-resolution ground truth
      - alpha_coarse, alpha_fine: AEF embeddings
    """

    def __init__(
        self,
        coarse: np.ndarray,             # (T, H_c, W_c) original coarse field
        coarse_up: np.ndarray,          # (T, H, W) upsampled
        residual: np.ndarray,           # (T, H, W)
        fine: np.ndarray,               # (T, H, W)
        dates: np.ndarray,              # (T,) datetime64
        indices: np.ndarray,            # which time indices to expose
        aef_by_year: Dict[int, Tuple[np.ndarray, np.ndarray]],
        pair_id: int,                   # 0=A, 1=B
        augment: bool = False,
        heavy_precip_percentile: float = 60.0,
    ) -> None:
        self.coarse = coarse
        self.coarse_up = coarse_up
        self.residual = residual
        self.fine = fine
        self.dates = dates
        self.pair_id = pair_id
        self.augment = augment

        H_fine, W_fine = fine.shape[1], fine.shape[2]
        self.H_fine = H_fine
        self.W_fine = W_fine

        # Build index list with heavy-rain oversampling + deterministic augmentation
        # For each heavy-rain day, we create 3 augmented copies:
        #   0 = original (no flip)
        #   1 = horizontal flip
        #   2 = vertical flip
        #   3 = both flips
        # Non-heavy days are kept as-is (original only).
        if augment and len(indices) > 0:
            domain_means = np.array([
                fine[int(t)].mean() for t in indices
            ])
            threshold = np.percentile(domain_means, heavy_precip_percentile)
            heavy_mask = domain_means >= threshold
            heavy_indices = indices[heavy_mask]
            light_indices = indices[~heavy_mask]

            # Build (time_index, flip_code) pairs
            # Light days: only original
            index_pairs = [(int(t), 0) for t in light_indices]
            # Heavy days: original + 3 augmented versions
            for t in heavy_indices:
                index_pairs.append((int(t), 0))  # original
                index_pairs.append((int(t), 1))  # h-flip
                index_pairs.append((int(t), 2))  # v-flip
                index_pairs.append((int(t), 3))  # both

            self.index_pairs = index_pairs

            n_orig = len(indices)
            n_heavy = len(heavy_indices)
            n_light = len(light_indices)
            log.info(f"    Augmentation: {n_heavy} heavy-rain days × 4 orientations + "
                     f"{n_light} light days × 1 = {len(self.index_pairs)} total samples "
                     f"(was {n_orig})")
        else:
            self.index_pairs = [(int(t), 0) for t in indices]

        # Pre-process AEF: convert to torch, resize to fine grid
        self.aef_tensors = {}
        for year, (ac_np, af_np) in aef_by_year.items():
            ac = torch.from_numpy(ac_np.transpose(2, 0, 1)).float()  # (D, H, W)
            af = torch.from_numpy(af_np.transpose(2, 0, 1)).float()

            if ac.shape[1:] != (H_fine, W_fine):
                ac = F.interpolate(ac.unsqueeze(0), size=(H_fine, W_fine),
                                   mode="bilinear", align_corners=False).squeeze(0)
            if af.shape[1:] != (H_fine, W_fine):
                af = F.interpolate(af.unsqueeze(0), size=(H_fine, W_fine),
                                   mode="bilinear", align_corners=False).squeeze(0)

            self.aef_tensors[year] = (ac, af)

        self.fallback_year = sorted(self.aef_tensors.keys())[0] if self.aef_tensors else None

    def _get_year(self, t_idx: int) -> int:
        return int(str(self.dates[t_idx])[:4])

    def __len__(self) -> int:
        return len(self.index_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t, flip_code = self.index_pairs[idx]

        # Convert from meters/day to mm/day (×1000) for better numerical range.
        M_TO_MM = 1000.0

        x_coarse = torch.from_numpy(self.coarse[t].astype(np.float32)).unsqueeze(0) * M_TO_MM
        x_coarse_up = torch.from_numpy(self.coarse_up[t].astype(np.float32)).unsqueeze(0) * M_TO_MM
        residual = torch.from_numpy(self.residual[t].astype(np.float32)).unsqueeze(0) * M_TO_MM
        x_fine = torch.from_numpy(self.fine[t].astype(np.float32)).unsqueeze(0) * M_TO_MM

        # AEF for this year
        year = self._get_year(t)
        if year in self.aef_tensors:
            alpha_coarse, alpha_fine = self.aef_tensors[year]
        else:
            alpha_coarse, alpha_fine = self.aef_tensors[self.fallback_year]

        # Deterministic augmentation based on flip_code:
        #   0 = original, 1 = h-flip, 2 = v-flip, 3 = both
        if flip_code in (1, 3):  # horizontal flip
            x_coarse = torch.flip(x_coarse, dims=[-1])
            x_coarse_up = torch.flip(x_coarse_up, dims=[-1])
            residual = torch.flip(residual, dims=[-1])
            x_fine = torch.flip(x_fine, dims=[-1])
            alpha_coarse = torch.flip(alpha_coarse, dims=[-1])
            alpha_fine = torch.flip(alpha_fine, dims=[-1])

        if flip_code in (2, 3):  # vertical flip
            x_coarse = torch.flip(x_coarse, dims=[-2])
            x_coarse_up = torch.flip(x_coarse_up, dims=[-2])
            residual = torch.flip(residual, dims=[-2])
            x_fine = torch.flip(x_fine, dims=[-2])
            alpha_coarse = torch.flip(alpha_coarse, dims=[-2])
            alpha_fine = torch.flip(alpha_fine, dims=[-2])

        return {
            "x_coarse": x_coarse,             # (1, H_c, W_c) original coarse
            "x_coarse_up": x_coarse_up,        # (1, H, W) mm/day
            "residual": residual,              # (1, H, W) mm/day
            "x_fine": x_fine,                  # (1, H, W) mm/day
            "alpha_coarse": alpha_coarse,      # (D, H, W)
            "alpha_fine": alpha_fine,           # (D, H, W)
            "pair_id": torch.tensor(self.pair_id, dtype=torch.long),
        }


def build_datasets(
    era5_dir: str,
    aef_dir: str,
    augment_train: bool = True,
) -> Dict[str, PairDataset]:
    """
    Build train and test PairDatasets for both pairs.

    Returns a dict with keys: train_A, test_A, train_B, test_B.
    Pairs are kept separate because they have different grid sizes
    and cannot be stacked in the same batch.
    """
    era5_dir = Path(era5_dir)
    aef_dir = Path(aef_dir)

    # Load ERA5 arrays (memory-mapped)
    A_coarse = np.load(era5_dir / "pair_A_coarse.npy", mmap_mode="r")
    A_coarse_up = np.load(era5_dir / "pair_A_coarse_up.npy", mmap_mode="r")
    A_residual = np.load(era5_dir / "pair_A_residual.npy", mmap_mode="r")
    A_fine = np.load(era5_dir / "pair_A_fine.npy", mmap_mode="r")
    A_dates = np.load(era5_dir / "valid_times_A.npy")
    A_train_idx = np.load(era5_dir / "train_indices_A.npy")
    A_test_idx = np.load(era5_dir / "test_indices_A.npy")

    B_coarse_up = np.load(era5_dir / "pair_B_coarse_up.npy", mmap_mode="r")
    B_residual = np.load(era5_dir / "pair_B_residual.npy", mmap_mode="r")
    B_fine = np.load(era5_dir / "pair_B_fine.npy", mmap_mode="r")
    B_dates = np.load(era5_dir / "valid_times_B.npy")
    B_train_idx = np.load(era5_dir / "train_indices_B.npy")
    B_test_idx = np.load(era5_dir / "test_indices_B.npy")

    # Pair B coarse = Pair A fine (25km is the coarse input for Pair B)
    B_coarse = A_fine

    log.info(f"Pair A: {A_fine.shape}, train={len(A_train_idx)}, test={len(A_test_idx)}")
    log.info(f"Pair B: {B_fine.shape}, train={len(B_train_idx)}, test={len(B_test_idx)}")

    # Determine years
    all_years = sorted(set(
        [int(str(d)[:4]) for d in A_dates] +
        [int(str(d)[:4]) for d in B_dates]
    ))
    log.info(f"Calendar years: {all_years}")

    # Load AEF per year
    log.info("Loading AEF for Pair A (50km, 25km)...")
    aef_A = _load_aef_for_years(aef_dir, all_years,
                                 PAIR_AEF_SCALES["A"]["coarse"],
                                 PAIR_AEF_SCALES["A"]["fine"])

    log.info("Loading AEF for Pair B (25km, 12.5km)...")
    aef_B = _load_aef_for_years(aef_dir, all_years,
                                 PAIR_AEF_SCALES["B"]["coarse"],
                                 PAIR_AEF_SCALES["B"]["fine"])

    def _make(c, cu, res, fine, dates, idx, aef, pid, aug):
        return PairDataset(c, cu, res, fine, dates, idx, aef, pid, aug)

    datasets = {
        "train_A": _make(A_coarse, A_coarse_up, A_residual, A_fine, A_dates,
                         A_train_idx, aef_A, 0, augment_train),
        "test_A":  _make(A_coarse, A_coarse_up, A_residual, A_fine, A_dates,
                         A_test_idx, aef_A, 0, False),
        "train_B": _make(B_coarse, B_coarse_up, B_residual, B_fine, B_dates,
                         B_train_idx, aef_B, 1, augment_train),
        "test_B":  _make(B_coarse, B_coarse_up, B_residual, B_fine, B_dates,
                         B_test_idx, aef_B, 1, False),
    }

    for name, ds in datasets.items():
        log.info(f"  {name}: {len(ds)} samples, grid={ds.H_fine}x{ds.W_fine}")

    return datasets


def build_paired_dataloaders(
    datasets: Dict[str, PairDataset],
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Build separate DataLoaders for each pair.

    Returns dict with keys: train_A, test_A, train_B, test_B.

    In the training loop, alternate between pairs:
        for batch_A, batch_B in zip(loaders["train_A"], loaders["train_B"]):
            loss_A = train_step(batch_A)
            loss_B = train_step(batch_B)
    """
    use_ddp = world_size > 1
    loaders = {}

    for name, ds in datasets.items():
        is_train = "train" in name
        if use_ddp:
            sampler = DistributedSampler(
                ds, num_replicas=world_size, rank=rank,
                shuffle=is_train, seed=seed,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = is_train

        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=is_train,
            persistent_workers=(num_workers > 0),
        )

    return loaders


# ── Convenience: interleaved training iterator ───────────────────────────────

class InterleavedPairIterator:
    """
    Yields batches alternating between Pair A and Pair B.
    Handles different-length loaders by cycling the shorter one.

    Usage:
        for batch in InterleavedPairIterator(loader_A, loader_B):
            # batch has same keys as PairDataset.__getitem__
            loss = train_step(batch)
    """

    def __init__(self, loader_A: DataLoader, loader_B: DataLoader):
        self.loader_A = loader_A
        self.loader_B = loader_B

    def __iter__(self):
        iter_A = iter(self.loader_A)
        iter_B = iter(self.loader_B)
        exhausted_A = False
        exhausted_B = False

        while True:
            if not exhausted_A:
                try:
                    yield next(iter_A)
                except StopIteration:
                    exhausted_A = True

            if not exhausted_B:
                try:
                    yield next(iter_B)
                except StopIteration:
                    exhausted_B = True

            if exhausted_A and exhausted_B:
                break

    def __len__(self):
        return len(self.loader_A) + len(self.loader_B)


if __name__ == "__main__":
    """
    Quick shape check:
        python dataset.py --era5-dir data/processed --aef-dir data/aef_downsampled_by_year
    """
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--era5-dir", required=True)
    parser.add_argument("--aef-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    datasets = build_datasets(args.era5_dir, args.aef_dir)
    loaders = build_paired_dataloaders(datasets, batch_size=args.batch_size)

    print(f"\nDataset sizes:")
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} samples")

    print(f"\nSample batch shapes (Pair A):")
    batch = next(iter(loaders["train_A"]))
    for k, v in batch.items():
        print(f"  {k:20s}: {tuple(v.shape)} dtype={v.dtype}")

    print(f"\nSample batch shapes (Pair B):")
    batch = next(iter(loaders["train_B"]))
    for k, v in batch.items():
        print(f"  {k:20s}: {tuple(v.shape)} dtype={v.dtype}")

    print(f"\nInterleaved iterator: {len(InterleavedPairIterator(loaders['train_A'], loaders['train_B']))} total batches")