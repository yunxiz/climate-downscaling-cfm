"""
U-Net for OT-CFM precipitation downscaling.

Changes from Stochastic Model:
  - No score head (deterministic OT-CFM, not SF2M)
  - base_channels=48 (was 128), channel_mult=(1,2,4) → [48, 96, 192]
  - 1 ResBlock per level (was 2)
  - 2 attention heads (was 4)
  - ~1.5M parameters (was 33M)

Ensemble generation: run ODE from different x0 ~ N(0,I) initializations.
Each x0 produces a different sample from the learned conditional distribution.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeMLPEmbedding(nn.Module):
    def __init__(self, time_dim, out_dim):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


class AdaGroupNorm(nn.Module):
    def __init__(self, channels, num_groups, emb_dim):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.proj = nn.Linear(emb_dim, channels * 2)

    def forward(self, x, emb):
        h = self.norm(x)
        scale_shift = self.proj(emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        return h * (1 + scale) + shift


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, num_groups=8):
        super().__init__()
        # Ensure num_groups divides channels
        ng = min(num_groups, out_ch)
        while out_ch % ng != 0:
            ng -= 1

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = AdaGroupNorm(out_ch, ng, emb_dim)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_ch, ng, emb_dim)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.act(self.norm1(h, t_emb))
        h = self.conv2(h)
        h = self.act(self.norm2(h, t_emb))
        return h + self.skip(x)


class DualResolutionCrossAttention(nn.Module):
    def __init__(self, hidden_dim, aef_dim=64, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.kv_proj = nn.Conv2d(aef_dim * 2, hidden_dim * 2, 1)
        self.out_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)

        ng = min(8, hidden_dim)
        while hidden_dim % ng != 0:
            ng -= 1
        self.norm = nn.GroupNorm(ng, hidden_dim)

    def forward(self, h, alpha_coarse, alpha_fine):
        B, C, H, W = h.shape

        if alpha_coarse.shape[2:] != (H, W):
            alpha_coarse = F.interpolate(alpha_coarse, size=(H, W), mode="bilinear",
                                         align_corners=False)
        if alpha_fine.shape[2:] != (H, W):
            alpha_fine = F.interpolate(alpha_fine, size=(H, W), mode="bilinear",
                                       align_corners=False)

        aef_cat = torch.cat([alpha_coarse, alpha_fine], dim=1)

        q = self.q_proj(h)
        kv = self.kv_proj(aef_cat)
        k, v = kv.chunk(2, dim=1)

        def reshape_heads(x):
            return x.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3).contiguous()

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        with torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.MATH]):
        # with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(q, k, v)

        out = out.transpose(2, 3).contiguous().reshape(B, C, H, W)

        return h + self.out_proj(self.norm(out))


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode="nearest")
        return self.conv(x)


class DownscalingUNet(nn.Module):
    """
    U-Net trained to fit OT-CFM.

    ~1.5M parameters.
    2 downsampling stages, dual-resolution AEF cross-attention.

    Input: (B, 2, H, W) — [x_coarse_up, r_noised]
    Output: (B, 1, H, W) — flow velocity v_θ
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=1,
        base_channels=48,
        channel_mult=(1, 2, 4),
        aef_dim=64,
        time_dim=64,
        num_heads=2,
    ):
        super().__init__()

        channels = [base_channels * m for m in channel_mult]  # [48, 96, 192]
        emb_dim = time_dim * 4  # 256
        self.channels = channels
        self.num_levels = len(channels)

        self.time_emb = TimeMLPEmbedding(time_dim, emb_dim)
        self.input_proj = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder: 1 ResBlock + cross-attention per level
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        for level in range(self.num_levels):
            ch = channels[level]
            ch_in = channels[level - 1] if level > 0 else channels[0]
            self.encoder_blocks.append(ResBlock(ch_in, ch, emb_dim))
            self.encoder_attns.append(DualResolutionCrossAttention(ch, aef_dim, num_heads))
            if level < self.num_levels - 1:
                self.downsamplers.append(Downsample(ch))
            else:
                self.downsamplers.append(nn.Identity())

        # Bottleneck
        self.bottleneck = ResBlock(channels[-1], channels[-1], emb_dim)
        self.bottleneck_cross = DualResolutionCrossAttention(channels[-1], aef_dim, num_heads)

        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        for level in reversed(range(self.num_levels - 1)):
            ch = channels[level]
            ch_in = channels[level + 1]
            self.upsamplers.append(Upsample(ch_in))
            # Skip doubles channels
            self.decoder_blocks.append(ResBlock(ch_in + ch, ch, emb_dim))
            self.decoder_attns.append(DualResolutionCrossAttention(ch, aef_dim, num_heads))

        # Output
        ng = min(8, channels[0])
        while channels[0] % ng != 0:
            ng -= 1
        self.output_head = nn.Sequential(
            nn.GroupNorm(ng, channels[0]),
            nn.SiLU(),
            nn.Conv2d(channels[0], out_channels, 3, padding=1),
        )

    def forward(self, t, x, alpha_coarse, alpha_fine):
        """
        Args:
            t: (B,) flow time
            x: (B, 2, H, W) = [coarse_up, r_noised]
            alpha_coarse, alpha_fine: (B, 64, H, W) AEF embeddings

        Returns:
            v_theta: (B, 1, H, W) flow velocity
        """
        t_emb = self.time_emb(t)
        h = self.input_proj(x)

        # Encoder
        skips = []
        for level in range(self.num_levels):
            h = self.encoder_blocks[level](h, t_emb)
            h = self.encoder_attns[level](h, alpha_coarse, alpha_fine)
            skips.append(h)
            if level < self.num_levels - 1:
                h = self.downsamplers[level](h)

        # Bottleneck
        h = self.bottleneck(h, t_emb)
        h = self.bottleneck_cross(h, alpha_coarse, alpha_fine)

        # Decoder
        for level_idx, level in enumerate(reversed(range(self.num_levels - 1))):
            skip = skips[level]
            h = self.upsamplers[level_idx](h, skip.shape[2:])
            h = torch.cat([h, skip], dim=1)
            h = self.decoder_blocks[level_idx](h, t_emb)
            h = self.decoder_attns[level_idx](h, alpha_coarse, alpha_fine)

        return self.output_head(h)