# Recursive Latent Alignment: Multi-Scale Climate Downscaling via Foundation Embeddings

Probabilistic precipitation downscaling from **25 km ERA5 reanalysis to 1.5625 km** over Illinois,
using a Conditional Flow Matching (CFM) super-resolution model conditioned on **AlphaEarth
foundation-model embeddings** as terrain-aware context. The model is applied recursively (four 2×
steps) and generates calibrated ensembles via noise injection, validated against 800 m PRISM
observations.

<!-- FILL 1: hero image. Export Figure 2 (CFM vs. PRISM comparison) from the report to figures/cfm_vs_prism.png
![CFM recursive downscaling vs. PRISM ground truth](figures/cfm_vs_prism.png) -->

---

## Overview

High-resolution precipitation fields matter for hydrology, agriculture, and infrastructure risk, but
global reanalysis products like ERA5 are far too coarse, and dynamical (physics-based) downscaling is
too expensive to run as an ensemble. This project asks whether a generative deep-learning model,
conditioned on rich geospatial foundation-model embeddings, can produce fine-scale *probabilistic*
precipitation estimates cheaply.

The core idea has two parts:

- **Recursive 2× super-resolution.** A single learned operator is applied repeatedly (25 → 12.5 → 6.25
  → 3.125 → 1.5625 km) rather than training a separate model per scale, so the method is
  scale-invariant and generalizes across resolutions.
- **Foundation-model conditioning.** Instead of the elevation-only terrain context used in prior work,
  the model conditions on the full 64-dimensional AlphaEarth embedding (topography, land cover, soil,
  hydrology, vegetation) via dual-resolution cross-attention — to our knowledge, the first use of GeoFM
  embeddings as conditioning for climate downscaling.

## Key results

Two tasks are evaluated separately because they operate at different scales.

**Single-step reconstruction** (one 2× step, reconstructed fine field vs. target):

| Model | RMSE | R² |
|---|---|---|
| Bicubic interpolation | 1.122 | −0.286 |
| XGBoost (no AEF) | 0.917 | 0.141 |
| **XGBoost + AEF** | **0.896** | **0.180** |

→ AlphaEarth embeddings add real spatial information: they improve RMSE and R² over both bicubic and
the no-embedding baseline.

**Recursive downscaling** (25 km → 1.5625 km, 4 steps, ensemble mean vs. PRISM, 60 test days):

| Model | Ensemble-mean RMSE | CRPS | Spread/skill |
|---|---|---|---|
| Bicubic baseline | 4.568 | — | — |
| XGBoost + AEF (recursive) | 4.748 | 2.404 | 0.89 |
| **CFM (recursive)** | **4.274** | 2.344 | 0.56 |

→ The CFM model is the only approach that **beats bicubic** under recursive rollout; the XGBoost
ensemble, using empirical noise injection, degrades below bicubic as errors accumulate across steps.
The trade-off: CFM's ensembles are **underdispersive** (spread/skill and coverage below target) — it
nails the conditional mean but underestimates fine-scale uncertainty. See
[Limitations](#limitations--future-work).

## Method

**Data pipeline.** Three sources over the Illinois domain (36.97°–42.51°N, 91.51°–87.50°W): ERA5 /
ERA5-Land daily precipitation (coarse input), AlphaEarth embeddings (64-dim, 10 m, streamed from a
>400 GB Zarr store and GeM-pooled to six resolutions), and PRISM 800 m (fine-scale validation). Models
train on scale-to-scale *residuals* over bicubic upsampling, which shrinks the generative target's
dynamic range.

**Model.** OT-CFM (via TorchCFM, σ=0) learns a conditional distribution over fine-scale residuals. The
velocity field is a U-Net (encoder channels 48/96/192, GroupNorm residual blocks, multi-head
attention) with **dual-resolution cross-attention** injecting AlphaEarth embeddings at both coarse and
fine scale, so the model modulates terrain influence by precipitation state rather than treating
embeddings as static per-pixel features.

**Three-phase training.** (1) scale-invariant 2× residual generation with flow-matching,
cycle-consistency (mass conservation), and spectral losses; (2) two-step rollout for recursion
robustness; (3) PRISM-supervised fine-tuning to correct ERA5 bias.

**Inference.** From 25 km ERA5, four recursive 2× steps to 1.5625 km; different Gaussian noise seeds
produce spatially coherent ensemble members that carry through the rollout.

Full detail, ablations, and per-day examples are in the [report](docs/) linked below.

## Repository structure

<!-- FILL 2: replace with your actual layout once the code is off DeltaAI and organized -->
```
climate-downscaling-cfm/
├── src/                # model, training loops, inference
├── scripts/            # data preprocessing pipeline
├── configs/            # experiment / training configs
├── figures/            # some example result figures
├── docs/               # final report PDF + slides
├── requirements.txt
└── README.md
```

## Data

None of the raw or preprocessed data is committed (the AlphaEarth store alone exceeds 400 GB). To
reproduce the pipeline, obtain the sources directly:

- **ERA5 / ERA5-Land** — Copernicus Climate Data Store (CDS)
- **PRISM** — Oregon State PRISM Group (https://prism.oregonstate.edu)
- **AlphaEarth embeddings** — Source Cooperative mosaic (https://source.coop/tge-labs/aef-mosaic)

The preprocessing scripts in `scripts/` handle regridding, GeM pooling of the embeddings, residual-pair
construction, and the rainy-day augmentation described in the report.

## Running it

> **Note:** This is research code. A full training run takes several hours on GPU (developed on an
> NCSA Delta AI allocation) and depends on large external datasets, so it is documented to be
> reproducible *in principle* rather than runnable end-to-end from a single command.

<!-- FILL 3: your actual environment + entry-point commands. Example scaffold below — correct to match the repo. -->
```bash
# environment
pip install -r requirements.txt   # Python, PyTorch, TorchGeo, TorchCFM, XGBoost, xarray/zarr

# 1. preprocess (after acquiring the data sources above)
python scripts/preprocess.py --config configs/illinois.yaml

# 2. train (multi-phase; hours on GPU)
python src/train.py --config configs/cfm.yaml

# 3. recursive ensemble inference + evaluation
python src/infer.py --checkpoint <path> --days 2025 --ensemble 60
```

<!-- FILL 4 (optional but high-value): if you can share a trained checkpoint and a small inference
demo that regenerates one comparison figure in minutes, link it here. This is the single best addition. -->

## Limitations & future work

Reported honestly, because these are the interesting parts:

- **Underdispersive ensembles.** CFM captures the conditional mean well but its ensemble spread is too
  narrow (coverage below the 50/80/90% targets). Better uncertainty calibration is the main open
  problem.
- **Partial convergence.** Training phases 2 and 3 do not fully converge; phase 3 (PRISM fine-tuning)
  shows little improvement over phase 2.
- **Data-limited.** AlphaEarth embeddings only span 2017–2025, capping training to 8 years and one
  region.

Directions: an inherently stochastic model (diffusion) for a calibration comparison; other geospatial
foundation models (Prithvi-EO-2.0, Clay, TESSERA) to extend the temporal range; and smoothly
downscaling to arbitrary resolution.

## Team & contributions

Course project for CS 598 (Generative AI), Spring 2026 — Rishab Sakalkale, Kassidy He, and Yunxi Zeng.
The work was collaborative without rigidly assigned roles.

**My contributions (Yunxi Zeng):** led the data preprocessing pipeline (ERA5/ERA5-Land/PRISM regridding
and residual-pair construction), implemented the XGBoost
baselines, and help developed the Conditional Flow Matching model and its training.

<!-- FILL 5: link teammates' GitHub/LinkedIn if they're happy to be tagged, and add the repo URL -->

## Paper & slides

- 📄 [Final report](docs/Project_Final_Report.pdf)
- 📊 [Presentation](docs/Final_Presentation.pptx)
