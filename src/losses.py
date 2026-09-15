'''
Losses terms for the models

in the spec file, loss terms are identical for U-Net and CFM, except term 1 which differs by architecture:
  - U-Net:  Gaussian NLL on the residual
  - CFM:    Flow matching loss (implemented in the CFM model file)

Spec reference:
  Term 1: Gaussian NLL            (U-Net only, Phase 1+2, weight 1.0)
  Term 2: Cycle-consistency L1    (Phase 1+2, weight 0.8)
  Term 3: Spectral PSD            (Phase 1 from epoch 5, Phase 2, weight 0.4)
  Term 4: Rollout consistency     (Phase 2 only, weight 0.1)
  Term 5: CRPS                    (Phase 2 only, weight 0.5)

'''

from typing import Optional
 
import torch
import torch.nn.functional as F
 
# Term 1 – Gaussian NLL on residual  (U-Net heteroscedastic loss)
 
def nll_loss(
    mu:       torch.Tensor,   # (B, C, H, W) predicted residual mean
    log_var:  torch.Tensor,   # (B, C, H, W) predicted residual log-variance
    target_r: torch.Tensor,   # (B, C, H, W) true residual
    mask:     Optional[torch.Tensor] = None,  # optional (B, 1, H, W) bool mask
) -> torch.Tensor:
    """
    Pixel-wise Gaussian negative log-likelihood:
        L = 0.5 * (log_var + (target - mu)^2 / exp(log_var))
 
    log_var is expected to already be clamped by the model output head.
    """
    var = log_var.exp()
    loss = 0.5 * (log_var + (target_r - mu).pow(2) / (var + 1e-8))
 
    if mask is not None:
        loss = loss[mask.expand_as(loss)]
 
    return loss.mean()
 
 
# Term 2 – Cycle-consistency (mass conservation)  L1
 
def cycle_consistency_loss(
    x_coarse:    torch.Tensor,   # (B, C, H_c, W_c)  original coarse field (not upsampled)
    x_fine_hat:  torch.Tensor,   # (B, C, H_f, W_f)  reconstructed fine field
    pool_factor: int = 2,
) -> torch.Tensor:
    """
    Enforces water-mass conservation:
        L_cycle = ||x_coarse - AvgPool(x_fine_hat, 2x)||_1
 
    x_coarse must be on the COARSE grid (not bicubic-upsampled).
    x_fine_hat = x_coarse_up + mu  (full reconstruction, not just the residual).
 
    L1 is used (not L2) because precipitation is heavy-tailed; L2 would
    over-penalise rare but physically real extreme events (spec §5 Term 2).
 
    Args:
        x_coarse:    Coarse field at native coarse resolution.
        x_fine_hat:  Reconstructed fine-resolution field.
        pool_factor: Factor by which fine is finer than coarse (2 for 2x steps).
    """
    # Downsample fine reconstruction back to coarse grid
    x_repooled = F.avg_pool2d(x_fine_hat, kernel_size=pool_factor, stride=pool_factor)
 
    # Ensure spatial sizes match (handle odd dims from rounding)
    if x_repooled.shape[2:] != x_coarse.shape[2:]:
        x_repooled = F.interpolate(
            x_repooled, size=x_coarse.shape[2:], mode="bilinear", align_corners=False
        )
 
    return F.l1_loss(x_coarse, x_repooled)
 
 
# Term 3 – Spectral PSD loss
 
def spectral_psd_loss(
    residual_true: torch.Tensor,  # (B, C, H, W)
    residual_pred: torch.Tensor,  # (B, C, H, W)
    k_min_frac:    float = 0.25,  # lower bound of sub-grid wavenumber band
    k_max_frac:    float = 1.00,  # upper bound of sub-grid wavenumber band
    n_bins:        int   = 64,    # number of radial wavenumber bins
) -> torch.Tensor:
    """
    Spectral Power Spectral Density loss on the residual:
        L_spec = ||log PSD(r_true)[k_subgrid] - log PSD(r_pred)[k_subgrid]||_2
 
    Forces the predicted residual to have the same spatial texture (frequency
    content) as the true residual.  Without this, the NLL loss alone allows
    the model to produce over-smooth residuals that score well pixel-wise but
    look unrealistically blurred.
 
    Computed only in the sub-grid wavenumber band [k_min_frac, k_max_frac] of
    the Nyquist frequency — the frequencies the model is actually generating
    rather than inheriting from the coarse input.
 
    Args:
        residual_true:  True residual  r      = x_fine - BicubicInterp(x_coarse)
        residual_pred:  Predicted residual mu  (U-Net output)
        k_min_frac:     Start of sub-grid band as fraction of max radial frequency
        k_max_frac:     End   of sub-grid band as fraction of max radial frequency
        n_bins:         Number of radial wavenumber bins (fixed, not data-dependent)
    """
    B, C, H, W = residual_true.shape
 
    def radial_psd(x: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean radial power spectrum for a (B, C, H, W) tensor.
 
        Steps:
          1. 2D real FFT  →  complex spectrum of shape (B, C, H, W//2+1)
          2. Power = |FFT|²  →  (B, C, H, W//2+1)
          3. Build a radial wavenumber grid k of shape (H, W//2+1)
               k[i,j] = sqrt(fy[i]² + fx[j]²)
             Note: rfft2 output has shape (H, W//2+1), so fy indexes rows (H)
             and fx indexes columns (W//2+1).  We use indexing="ij" to get
             the correct (H, W//2+1) output from meshgrid.
          4. Assign each (i,j) frequency bin to one of n_bins radial bands
             using bucketize — fully vectorised, no Python loop.
          5. Average power within each band → (B, C, n_bins)
        """
        # Step 1 & 2: FFT and power
        fft   = torch.fft.rfft2(x, norm="ortho")   # (B, C, H, W//2+1)
        power = fft.abs().pow(2)                    # (B, C, H, W//2+1)
 
        # Step 3: Radial wavenumber grid
        # rfft2 output rows ↔ fy (length H), columns ↔ fx (length W//2+1)
        fy = torch.fft.fftfreq(H, device=x.device)   # (H,)
        fx = torch.fft.rfftfreq(W, device=x.device)  # (W//2+1,)
        # indexing="ij": FY shape (H, W//2+1) with fy varying along dim 0
        #                FX shape (H, W//2+1) with fx varying along dim 1
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")   # both (H, W//2+1)
        k_grid = (FX.pow(2) + FY.pow(2)).sqrt()           # (H, W//2+1)
 
        # Step 4: Assign frequencies to radial bins via bucketize (vectorised)
        k_max  = k_grid.max()
        edges  = torch.linspace(0.0, float(k_max), n_bins + 1, device=x.device)
        # bin_idx in [0, n_bins-1]; clamp handles the edge case k == k_max
        bin_idx = torch.bucketize(k_grid, edges[1:], right=False)   # (H, W//2+1)
        bin_idx = bin_idx.clamp(max=n_bins - 1)
 
        # Step 5: Mean power per bin — scatter_add over flattened frequency dim
        # Flatten spatial freq dims: power (B, C, H, W//2+1) → (B, C, H*W//2+1)
        n_freq  = H * (W // 2 + 1)
        p_flat  = power.reshape(B, C, n_freq)              # (B, C, N_freq)
        idx_flat = bin_idx.reshape(n_freq)                 # (N_freq,)
 
        # Accumulate power into bins
        psd = torch.zeros(B, C, n_bins, device=x.device, dtype=x.dtype)
        psd.scatter_add_(
            2,
            idx_flat.unsqueeze(0).unsqueeze(0).expand(B, C, -1),
            p_flat,
        )
        # Count how many frequencies fell in each bin for proper averaging
        counts = torch.bincount(idx_flat, minlength=n_bins).float()   # (n_bins,)
        counts = counts.clamp(min=1.0)  # avoid divide by zero for empty bins
        psd = psd / counts.unsqueeze(0).unsqueeze(0)                  # (B, C, n_bins)
 
        return psd
 
    psd_true = radial_psd(residual_true)  # (B, C, n_bins)
    psd_pred = radial_psd(residual_pred)  # (B, C, n_bins)
 
    # Restrict to the sub-grid wavenumber band
    k_min_idx = int(k_min_frac * n_bins)
    k_max_idx = int(k_max_frac * n_bins)
    psd_true  = psd_true[..., k_min_idx:k_max_idx]   # (B, C, n_band)
    psd_pred  = psd_pred[..., k_min_idx:k_max_idx]
 
    # Compare in log space — emphasises relative differences across the band
    # rather than letting high-power low-frequency bins dominate
    eps = 1e-10
    return F.mse_loss(
        torch.log(psd_pred + eps),
        torch.log(psd_true + eps),
    )
 
 
# Term 4 – Rollout consistency (Phase 2 only)
 
def rollout_consistency_loss(
    x_coarse_start: torch.Tensor,  # (B, C, H_c, W_c)  original coarse field (step 0)
    x_fine_2:       torch.Tensor,  # (B, C, H_f2, W_f2) 2-step reconstruction
    pool_factor:    int = 4,       # 2× per step × 2 steps = 4×
) -> torch.Tensor:
    """
    Term 4 from the spec (Phase 2):
        L_rollout = ||x_coarse - AvgPool(x_fine_2_hat, 4x)||_1
 
    Forces the 2-step recursive output, when pooled back 4× to the
    original coarse resolution, to match the original coarse input.
    This directly combats error accumulation across recursive steps.
 
    Args:
        x_coarse_start:  The 25 km ERA5 input at the start of the 2-step rollout.
        x_fine_2:        The 6.25 km reconstruction after 2 recursive steps.
        pool_factor:     2^n_steps (= 4 for a 2-step rollout).
    """
    x_repooled = F.avg_pool2d(x_fine_2, kernel_size=pool_factor, stride=pool_factor)
 
    if x_repooled.shape[2:] != x_coarse_start.shape[2:]:
        x_repooled = F.interpolate(
            x_repooled, size=x_coarse_start.shape[2:],
            mode="bilinear", align_corners=False,
        )
 
    return F.l1_loss(x_coarse_start, x_repooled)
 
 
# Term 5 – CRPS  (approximated via energy score on samples)
 
def crps_loss(
    x_fine_samples: torch.Tensor,  # (B, N_samples, C, H, W)  ensemble of predictions
    x_fine_true:    torch.Tensor,  # (B, C, H, W)              ground truth
) -> torch.Tensor:
    """
    Continuous Ranked Probability Score (CRPS) via the energy-score formulation:
        CRPS(F, y) = E[|X - y|] - 0.5 * E[|X - X'|]
        where X, X' ~ F (independent draws from the predictive distribution)
 
    For U-Net heteroscedastic: generate N samples from N(mu, sigma^2) and
    use them as the ensemble.
 
    For CFM: run N forward passes with different noise vectors.
 
    Args:
        x_fine_samples: (B, N, C, H, W) ensemble
        x_fine_true:    (B, C, H, W)    observation
    """
    B, N, C, H, W = x_fine_samples.shape
 
    y = x_fine_true.unsqueeze(1)  # (B, 1, C, H, W)
 
    # E[|X - y|]
    term1 = (x_fine_samples - y).abs().mean(dim=1)  # (B, C, H, W)
 
    # E[|X - X'|] via pairwise average over the N dimension
    # For memory efficiency we use the identity:
    # E[|X - X'|] = 2/N * sum_i |X_i - mean(X)|  (exact only for gaussian,
    # approximation for general distributions; full pairwise is O(N^2))
    # Full pairwise for small N (N <= 50):
    if N <= 50:
        diff = x_fine_samples.unsqueeze(2) - x_fine_samples.unsqueeze(1)  # (B,N,N,C,H,W)
        term2 = diff.abs().mean(dim=(1, 2))  # (B, C, H, W)
    else:
        # Approximation for large N
        mean_pred = x_fine_samples.mean(dim=1, keepdim=True)
        term2 = 2.0 * (x_fine_samples - mean_pred).abs().mean(dim=1)
 
    crps = term1 - 0.5 * term2
    return crps.mean()
 
 
def compute_unet_loss(
    mu:             torch.Tensor,
    log_var:        torch.Tensor,
    residual_true:  torch.Tensor,
    x_coarse:       torch.Tensor,   # coarse at NATIVE coarse resolution
    x_coarse_up:    torch.Tensor,   # bicubic-upsampled coarse at fine grid
    epoch:          int,
    phase:          int = 1,        # 1 or 2
    # Loss weights (from spec)
    w_nll:     float = 1.0,
    w_cycle:   float = 0.8,
    w_spectral: float = 0.4,
    w_crps:    float = 0.5,
    spectral_start_epoch: int = 5,
    # Phase 2 rollout  (set these when phase==2)
    x_coarse_start: Optional[torch.Tensor] = None,
    x_fine_2:       Optional[torch.Tensor] = None,
    w_rollout: float = 0.1,
    n_crps_samples: int = 10,
) -> tuple:
    """
    Computes the full training loss for the heteroscedastic U-Net.
 
    Returns:
        total_loss: scalar tensor
        loss_dict:  {name: scalar} for logging
    """
    x_fine_hat = x_coarse_up + mu   # full reconstruction on fine grid
 
    # Term 1: NLL
    l_nll = nll_loss(mu, log_var, residual_true)
 
    # Term 2: Cycle-consistency
    l_cycle = cycle_consistency_loss(x_coarse, x_fine_hat)
 
    # Term 3: Spectral (delayed start)
    if epoch >= spectral_start_epoch:
        l_spectral = spectral_psd_loss(residual_true, mu)
    else:
        l_spectral = torch.tensor(0.0, device=mu.device)
 
    loss_dict = {
        "nll":      l_nll.item(),
        "cycle":    l_cycle.item(),
        "spectral": l_spectral.item(),
    }
 
    total = w_nll * l_nll + w_cycle * l_cycle + w_spectral * l_spectral
 
    # Phase 2 additional terms
    if phase == 2:
        # Term 4: Rollout consistency
        if x_coarse_start is not None and x_fine_2 is not None:
            l_rollout = rollout_consistency_loss(x_coarse_start, x_fine_2)
            total = total + w_rollout * l_rollout
            loss_dict["rollout"] = l_rollout.item()
 
        # Term 5: CRPS (draw samples from N(mu, sigma^2))
        sigma   = torch.exp(0.5 * log_var)
        eps     = torch.randn(mu.shape[0], n_crps_samples, *mu.shape[1:],
                              device=mu.device, dtype=mu.dtype)
        samples = (x_coarse_up.unsqueeze(1) + mu.unsqueeze(1)
                   + sigma.unsqueeze(1) * eps)  # (B, N, C, H, W)
        x_fine_true = x_coarse_up + residual_true
        l_crps = crps_loss(samples, x_fine_true)
        total = total + w_crps * l_crps
        loss_dict["crps"] = l_crps.item()
 
    loss_dict["total"] = total.item()
    return total, loss_dict
 