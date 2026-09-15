"""
U-Net vector field network for [SF]²M precipitation downscaling.

Architecture (per spec, adapted to 2 downsampling stages):
  - Input: [x_coarse_up, r_noised] concatenated → 2 channels
  - Encoder: 2 downsampling stages with residual blocks + GroupNorm
    Channels: [128, 256, 512]
  - Bottleneck: self-attention for global spatial context
  - Decoder: 2 upsampling stages with skip connections
  - Conditioning: dual-resolution cross-attention at every level
  - Time embedding: sinusoidal → MLP, injected via AdaGN in every ResBlock
  - Output: 1 channel (flow velocity) + 1 channel (score), or separate heads

The network outputs both v_θ (flow) and s_θ (score) from a shared trunk
with separate final projection layers, as recommended by the SF2M paper.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for flow time t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) → (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLPEmbedding(nn.Module):
    """Time embedding: sinusoidal → Linear → SiLU → Linear."""

    def __init__(self, time_dim: int, out_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sinusoidal(t))


class AdaGroupNorm(nn.Module):
    """
    Adaptive Group Normalization (FiLM-style).
    Applies GroupNorm then modulates with scale/shift from time embedding.
    """

    def __init__(self, channels: int, num_groups: int, emb_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.proj = nn.Linear(emb_dim, channels * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), emb: (B, emb_dim)
        h = self.norm(x)
        scale_shift = self.proj(emb)[:, :, None, None]  # (B, 2C, 1, 1)
        scale, shift = scale_shift.chunk(2, dim=1)
        return h * (1 + scale) + shift


class ResBlock(nn.Module):
    """
    Residual block with AdaGN time conditioning.
    Conv → AdaGN → SiLU → Conv → AdaGN → SiLU → skip
    """

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = AdaGroupNorm(out_ch, num_groups, emb_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_ch, num_groups, emb_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.act(self.norm1(h, t_emb))
        h = self.conv2(h)
        h = self.act(self.norm2(h, t_emb))
        return h + self.skip(x)


class DualResolutionCrossAttention(nn.Module):
    """
    Dual-resolution cross-attention from AEF embeddings.

    Q = W_q · h_i           (from U-Net hidden state)
    K/V = W_kv · [α_coarse ; α_fine]   (from AEF at both scales)
    h_out = h_i + softmax(QK^T / √d) V

    AEF tensors are spatially interpolated to match the hidden state
    resolution at each U-Net level.
    """

    def __init__(self, hidden_dim: int, aef_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        # AEF has 2 * aef_dim channels (coarse + fine concatenated)
        self.kv_proj = nn.Conv2d(aef_dim * 2, hidden_dim * 2, 1)
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.norm = nn.GroupNorm(8, hidden_dim)

    def forward(
        self,
        h: torch.Tensor,             # (B, C, H, W) hidden state
        alpha_coarse: torch.Tensor,   # (B, D, H_aef, W_aef)
        alpha_fine: torch.Tensor,     # (B, D, H_aef, W_aef)
    ) -> torch.Tensor:
        B, C, H, W = h.shape

        # Interpolate AEF to match hidden state spatial dims
        if alpha_coarse.shape[2:] != (H, W):
            alpha_coarse = F.interpolate(alpha_coarse, size=(H, W), mode="bilinear",
                                         align_corners=False)
        if alpha_fine.shape[2:] != (H, W):
            alpha_fine = F.interpolate(alpha_fine, size=(H, W), mode="bilinear",
                                       align_corners=False)

        # Concatenate coarse and fine AEF: (B, 2D, H, W)
        aef_cat = torch.cat([alpha_coarse, alpha_fine], dim=1)

        # Project
        q = self.q_proj(h)                      # (B, C, H, W)
        kv = self.kv_proj(aef_cat)              # (B, 2C, H, W)
        k, v = kv.chunk(2, dim=1)               # each (B, C, H, W)

        # Reshape for multi-head attention: (B, heads, N, head_dim)
        def reshape_heads(x):
            return x.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        q = reshape_heads(q)  # (B, heads, N, d)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Use PyTorch's memory-efficient attention (FlashAttention when available)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, d)
        out = out.transpose(2, 3).reshape(B, C, H, W)

        return h + self.out_proj(self.norm(out))


class SelfAttention2D(nn.Module):
    """Spatial self-attention for the bottleneck."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.out_proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        qkv = self.qkv(x).view(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # (B, heads, d, N) → (B, heads, N, d) for scaled_dot_product_attention
        q = q.transpose(2, 3)
        k = k.transpose(2, 3)
        v = v.transpose(2, 3)

        out = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, d)
        out = out.transpose(2, 3).reshape(B, C, H, W)

        return x + self.out_proj(self.norm(out))


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=target_size, mode="nearest")
        return self.conv(x)


class DownscalingUNet(nn.Module):
    """
    U-Net for [SF]²M precipitation downscaling.

    2 downsampling stages, dual-resolution AEF cross-attention at every level,
    shared trunk with separate flow (v_θ) and score (s_θ) output heads.

    Input: (B, 2, H, W) — [x_coarse_up, r_noised] concatenated
    Time:  (B,) — flow time t ∈ [0, 1]
    AEF:   alpha_coarse (B, 64, H, W), alpha_fine (B, 64, H, W)
    Output: v_θ (B, 1, H, W), s_θ (B, 1, H, W)
    """

    def __init__(
        self,
        in_channels: int = 2,       # coarse_up + r_noised
        out_channels: int = 1,      # residual prediction
        base_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4),  # 3 levels, 2 downsamplings
        aef_dim: int = 64,
        time_dim: int = 128,
        num_heads: int = 4,
        num_res_blocks: int = 2,
    ):
        super().__init__()

        channels = [base_channels * m for m in channel_mult]  # [128, 256, 512]
        emb_dim = time_dim * 4  # 512
        self.channels = channels
        self.num_levels = len(channels)

        # Time embedding
        self.time_emb = TimeMLPEmbedding(time_dim, emb_dim)

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for level in range(self.num_levels):
            ch = channels[level]
            ch_in = channels[level - 1] if level > 0 else channels[0]

            # ResBlocks at this level
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                blocks.append(ResBlock(ch_in if i == 0 else ch, ch, emb_dim))
            self.encoder_blocks.append(blocks)

            # Cross-attention with AEF at this level
            self.encoder_attns.append(
                DualResolutionCrossAttention(ch, aef_dim, num_heads)
            )

            # Downsample (except at last level = bottleneck)
            if level < self.num_levels - 1:
                self.downsamplers.append(Downsample(ch))
            else:
                self.downsamplers.append(nn.Identity())

        # Bottleneck
        self.bottleneck_res = ResBlock(channels[-1], channels[-1], emb_dim)
        self.bottleneck_attn = SelfAttention2D(channels[-1], num_heads)
        self.bottleneck_cross = DualResolutionCrossAttention(
            channels[-1], aef_dim, num_heads
        )

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(self.num_levels - 1)):
            ch = channels[level]
            ch_in = channels[level + 1]

            self.upsamplers.append(Upsample(ch_in))

            # Skip connection doubles channels, then ResBlocks reduce
            blocks = nn.ModuleList()
            for i in range(num_res_blocks):
                skip_ch = ch if i > 0 else ch_in + ch  # first block receives skip
                blocks.append(ResBlock(skip_ch, ch, emb_dim))
            self.decoder_blocks.append(blocks)

            self.decoder_attns.append(
                DualResolutionCrossAttention(ch, aef_dim, num_heads)
            )

        # Output heads
        self.flow_head = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1),
        )
        self.score_head = nn.Sequential(
            nn.GroupNorm(8, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1),
        )

    def forward(
        self,
        t: torch.Tensor,             # (B,)
        x: torch.Tensor,             # (B, 2, H, W) = [coarse_up, r_noised]
        alpha_coarse: torch.Tensor,   # (B, 64, H, W)
        alpha_fine: torch.Tensor,     # (B, 64, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            v_theta: (B, 1, H, W) — flow velocity prediction
            s_theta: (B, 1, H, W) — score prediction
        """
        t_emb = self.time_emb(t)  # (B, emb_dim)

        # Input projection
        h = self.input_proj(x)  # (B, C0, H, W)

        # Encoder
        skips = []
        for level in range(self.num_levels):
            for block in self.encoder_blocks[level]:
                h = block(h, t_emb)

            h = self.encoder_attns[level](h, alpha_coarse, alpha_fine)
            skips.append(h)

            if level < self.num_levels - 1:
                h = self.downsamplers[level](h)

        # Bottleneck
        h = self.bottleneck_res(h, t_emb)
        h = self.bottleneck_attn(h)
        h = self.bottleneck_cross(h, alpha_coarse, alpha_fine)

        # Decoder
        for level_idx, level in enumerate(reversed(range(self.num_levels - 1))):
            skip = skips[level]
            target_size = skip.shape[2:]

            h = self.upsamplers[level_idx](h, target_size)
            h = torch.cat([h, skip], dim=1)  # skip connection

            for block in self.decoder_blocks[level_idx]:
                h = block(h, t_emb)

            h = self.decoder_attns[level_idx](h, alpha_coarse, alpha_fine)

        # Output
        v_theta = self.flow_head(h)   # (B, 1, H, W)
        s_theta = self.score_head(h)  # (B, 1, H, W)

        return v_theta, s_theta


class DownscalingSDE(nn.Module):
    """
    Wraps the U-Net as a torchsde-compatible SDE for ensemble inference.

    dx = f(t, x) dt + g(t, x) dW

    where:
      f(t, x) = v_θ(t, x) + σ² · s_θ(t, x)    (SB drift)
      g(t, x) = σ                                 (constant diffusion)
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, unet: DownscalingUNet, sigma: float,
                 coarse_up: torch.Tensor, alpha_coarse: torch.Tensor,
                 alpha_fine: torch.Tensor):
        super().__init__()
        self.unet = unet
        self.sigma = sigma
        self.coarse_up = coarse_up    # (B, 1, H, W) — fixed conditioning
        self.alpha_coarse = alpha_coarse
        self.alpha_fine = alpha_fine
        self.shape = coarse_up.shape  # for reshaping flat tensors

    def _unflatten(self, y: torch.Tensor) -> torch.Tensor:
        """torchsde passes (B, D) flat; reshape to (B, 1, H, W)."""
        B = y.shape[0]
        return y.view(B, 1, self.shape[2], self.shape[3])

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_rt = self._unflatten(y)
        t_batch = t.expand(x_rt.shape[0])
        x_input = torch.cat([self.coarse_up, x_rt], dim=1)  # (B, 2, H, W)
        v, s = self.unet(t_batch, x_input, self.alpha_coarse, self.alpha_fine)
        drift = v + self.sigma ** 2 * s
        return drift.flatten(1)

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.full_like(y, self.sigma)