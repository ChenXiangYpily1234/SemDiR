"""Output-side PCPF V7: noise-aware high-resolution detail relay.

Exploratory method: corrects a 32x32 semantic proposal in the PCPF Haar basis,
then injects input-derived 1/4-resolution detail into the highest-resolution
Feature2Pyramid output. No labels or prediction targets are read at inference.
Do not interpret local feature outliers as ground-truth noise or boundaries.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from research.pcpf_v6.core import contrast_correction

MODES = ('full', 'no_noise_filter', 'no_pcpf', 'coarse_only', 'peer_exact')


def local_detail(x: Tensor) -> Tensor:
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError('Expected nonempty NCHW feature map')
    return x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)


def spatial_median_3x3(x: Tensor) -> Tensor:
    """Median over 3x3 spatial positions for a scalar NCHW map; replicate borders."""
    if x.ndim != 4 or x.shape[1] != 1 or min(x.shape) < 1:
        raise ValueError('Median signal must have shape [B,1,H,W]')
    b, c, h, w = x.shape
    padded = F.pad(x, (1, 1, 1, 1), mode='replicate')
    return F.unfold(padded, kernel_size=3).median(dim=1).values.reshape(b, 1, h, w)


def persistence(early: Tensor, middle: Tensor, eps: float = 1e-6) -> Tensor:
    """Early/middle detail directional agreement; descriptive, not semantic certainty."""
    if early.shape != middle.shape or early.ndim != 4:
        raise ValueError('Persistence requires equally sized NCHW maps')
    a = local_detail(F.layer_norm(early.movedim(1, -1).float(), (early.shape[1],)).movedim(-1, 1))
    b = local_detail(F.layer_norm(middle.movedim(1, -1).float(), (middle.shape[1],)).movedim(-1, 1))
    score = F.cosine_similarity(a, b, dim=1, eps=eps).unsqueeze(1).clamp(0, 1)
    valid = (a.square().sum(dim=1, keepdim=True) > eps**2) & (b.square().sum(dim=1, keepdim=True) > eps**2)
    return (score * valid).detach()


def impulse_indicator(detail: Tensor, eps: float = 1e-6) -> Tensor:
    """Scale-normalized isolated-response indicator, not a semantic noise label."""
    if detail.ndim != 4 or min(detail.shape) < 1:
        raise ValueError('Expected detail NCHW')
    magnitude = detail.float().square().mean(dim=1, keepdim=True).sqrt()
    median = spatial_median_3x3(magnitude)
    local_peak = F.max_pool2d(magnitude, kernel_size=3, stride=1, padding=1)
    return ((magnitude - median).relu() / local_peak.clamp_min(eps)).clamp(0, 1).detach()


class OutputDetailRelay(nn.Module):
    """PCPF coarse correction plus semantic-gated, impulse-filtered pixel details.

    Source feature grid is 1/16 resolution. Pixel stem yields 1/4 resolution.
    All outward projections are initially zero: method equals peer feature
    pyramid at step zero, *provided shared encoder/neck parameters are equal*.
    """
    def __init__(self, channels: int = 384, stem_channels: int = 48, mode: str = 'full'):
        super().__init__()
        if mode not in MODES or type(channels) is not int or channels < 2 or type(stem_channels) is not int or stem_channels < 2:
            raise ValueError('Invalid mode or channels')
        self.mode = mode
        self.coarse_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.coarse_out = nn.Conv2d(channels, channels, 1, bias=False)
        self.stem = nn.Sequential(nn.Conv2d(3, stem_channels, 3, stride=2, padding=1), nn.GELU(),
                                  nn.Conv2d(stem_channels, stem_channels, 3, stride=2, padding=1), nn.GELU())
        self.detail_dw = nn.Conv2d(stem_channels, stem_channels, 3, padding=1, groups=stem_channels, bias=False)
        self.semantic_proj = nn.Conv2d(channels, stem_channels, 1, bias=False)
        self.detail_gate = nn.Conv2d(stem_channels * 2 + 2, 1, 1)
        self.detail_out = nn.Conv2d(stem_channels, channels, 1, bias=False)
        self.reset_relay()

    def reset_relay(self) -> None:
        """Reapply after MMEngine init_weights, which may reinitialize modules."""
        with torch.no_grad():
            nn.init.dirac_(self.coarse_dw.weight, groups=self.coarse_dw.groups)
            nn.init.dirac_(self.detail_dw.weight, groups=self.detail_dw.groups)
            self.coarse_out.weight.zero_()
            self.detail_out.weight.zero_()
            self.detail_gate.weight.zero_()
            self.detail_gate.bias.zero_()

    def _zero_dependency(self, tensor: Tensor) -> Tensor:
        # Keeps every relay parameter in DDP backward graph for a disabled control.
        zero = tensor.new_zeros(())
        for parameter in self.parameters():
            zero = zero + parameter.flatten()[0] * 0.0
        return zero

    def refine_coarse(self, fused: Tensor, early: Tensor, mid: Tensor) -> tuple[Tensor, Tensor]:
        if fused.shape != early.shape or fused.shape != mid.shape or fused.ndim != 4:
            raise ValueError('Coarse inputs must have identical NCHW shape')
        if self.mode == 'peer_exact':
            return fused + self._zero_dependency(fused), fused.new_zeros((fused.shape[0],1,*fused.shape[-2:]))
        support = persistence(early, mid).to(fused.dtype)
        early_norm = F.layer_norm(early.movedim(1,-1).float(), (early.shape[1],)).movedim(-1,1).to(fused.dtype)
        candidate = self.coarse_out(F.gelu(self.coarse_dw(local_detail(early_norm)) * (0.25 + 0.75 * support)))
        candidate = candidate - candidate.mean(dim=(-2,-1), keepdim=True)
        if self.mode != 'no_pcpf':
            correction = contrast_correction(
                fused.movedim(1,-1).contiguous(), candidate.movedim(1,-1).contiguous(),
                early.movedim(1,-1).contiguous(), mode='persistent', offset=(0,0))
            candidate = candidate + correction.movedim(-1,1)
        return fused + candidate, support

    def refine_highres(self, image: Tensor, high: Tensor, support: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3 or high.ndim != 4 or high.shape[1] != self.coarse_out.out_channels:
            raise ValueError('Wrong image/high feature dimensions')
        if self.mode in ('peer_exact', 'coarse_only'):
            return high + self._zero_dependency(high)
        local = self.detail_dw(self.stem(image))
        local = local_detail(local)
        if local.shape[-2:] != high.shape[-2:]:
            # Keep the image-detail path aligned with the actual ViT/FPN grid.
            # Validation uses keep-ratio resize, so its width is not generally
            # divisible by four (or by the ViT patch size). The two strided
            # paths then round dimensions differently. Training crops are
            # 512x512 and retain the original exact-size behavior.
            local = F.interpolate(local, size=high.shape[-2:], mode='bilinear',
                                  align_corners=False)
        confidence = F.interpolate(support.float(), size=local.shape[-2:], mode='nearest').to(local.dtype)
        outlier = impulse_indicator(local).to(local.dtype)
        # Never hard-delete a detail: a thin real object may look like an impulse.
        attenuation = 1.0 - 0.75 * (1.0 - confidence) * outlier
        filtered = local if self.mode == 'no_noise_filter' else local * attenuation
        semantic = self.semantic_proj(high)
        gate = torch.sigmoid(self.detail_gate(torch.cat([filtered, semantic, confidence, outlier], dim=1)))
        proposal = self.detail_out(filtered) * gate
        proposal = proposal - proposal.mean(dim=(-2,-1),keepdim=True)
        return high + proposal

    def forward(self, image: Tensor, early: Tensor, mid: Tensor,
                fused: Tensor, pyramid_neck: nn.Module) -> tuple[Tensor, ...]:
        coarse, support = self.refine_coarse(fused, early, mid)
        pyramid = pyramid_neck((coarse, coarse, coarse, coarse))
        if not isinstance(pyramid, (tuple, list)) or len(pyramid) != 4:
            raise ValueError('Feature2Pyramid must produce exactly four feature maps')
        high = self.refine_highres(image, pyramid[0], support)
        return (high, *pyramid[1:])
