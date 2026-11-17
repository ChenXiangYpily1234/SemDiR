"""Persistent-Contrast Projected Fusion (PCPF), an exploratory V6 operator.

The guarantee concerns chosen orthogonal contrast modes of the fusion update,
NOT every adjacent pair, MHSA itself, decoder margins or segmentation quality.
"""
from __future__ import annotations

import math
import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODES = ('persistent', 'current_only', 'shifted_anchor', 'no_projection')


def _check_offset(offset: tuple[int, int]) -> None:
    if (not isinstance(offset, (tuple, list)) or len(offset) != 2
            or any(type(v) is not int or v not in (0, 1) for v in offset)):
        raise ValueError('offset must contain two integers in {0, 1}')


def _haar_axis(x: Tensor, axis: int, offset: int) -> Tensor:
    """Self-inverse orthonormal pair transform; unpaired sites stay unchanged."""
    size = x.shape[axis]
    start = min(offset, size)
    length = 2 * ((size - start) // 2)
    if length == 0:
        return x
    middle = x.narrow(axis, start, length).movedim(axis, -2)
    paired = middle.reshape(*middle.shape[:-2], length // 2, 2, middle.shape[-1])
    a, b = paired.unbind(-2)
    # Scale BEFORE addition: a+b may overflow even when (a+b)/sqrt(2) is representable.
    a, b = a / math.sqrt(2), b / math.sqrt(2)
    transformed = torch.stack((a + b, a - b), dim=-2)
    transformed = transformed.reshape(middle.shape).movedim(-2, axis)
    return torch.cat((x.narrow(axis, 0, start), transformed,
                      x.narrow(axis, start + length, size - start - length)), dim=axis)


def haar2(x: Tensor, offset: tuple[int, int] = (0, 0)) -> Tensor:
    """Orthonormal local basis for B,H,W,C; also its own inverse.

    No padding or periodic neighbors. Odd edges use 1x2/2x1/1x1 cells.
    """
    _check_offset(offset)
    if x.ndim != 4 or min(x.shape) < 1:
        raise ValueError('expected a nonempty B,H,W,C tensor')
    return _haar_axis(_haar_axis(x, 2, offset[1]), 1, offset[0])


def detail_mask(h: int, w: int, offset: tuple[int, int], device: torch.device) -> Tensor:
    _check_offset(offset)
    row = torch.zeros(h, dtype=torch.bool, device=device)
    col = torch.zeros(w, dtype=torch.bool, device=device)
    for values, size, start in ((row, h, offset[0]), (col, w, offset[1])):
        end = start + 2 * ((size - start) // 2)
        values[start + 1:end:2] = True
    return (row[:, None] | col[None, :])[None, :, :, None]


def _unit(x: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    # Stable L2 validity, not an axis-dependent max-component threshold.
    scale = x.abs().amax(-1, keepdim=True)
    nonzero = scale > 0
    scaled = x / torch.where(nonzero, scale, torch.ones_like(scale))
    length = scaled.square().sum(-1, keepdim=True).sqrt()
    safe_length = torch.where(nonzero, length, torch.ones_like(length))
    # Compare without forming scale*length, which can overflow.
    valid = nonzero & (scale > eps / safe_length)
    unit = scaled / safe_length
    return torch.where(valid, unit, torch.zeros_like(unit)), valid


def mode_correction(update: Tensor, unit: Tensor, reliability: Tensor) -> Tensor:
    """Minimal correction for fixed unit direction and reliability in [0,1].

    q=<unit,update> becomes max(q,0)+(1-r)*min(q,0). ReLU's zero
    subgradient at q=0 leaves the initial proposal Jacobian equal to identity.
    """
    q = (update * unit).sum(-1, keepdim=True)
    return reliability * F.relu(-q) * unit


def contrast_correction(current: Tensor, update: Tensor, anchor: Tensor, *,
                        mode: str = 'persistent', offset: tuple[int, int] = (0, 0),
                        eps: float = 1e-6) -> Tensor:
    """Correct a proposal using stop-gradient contrast directions/reliabilities.

    Inputs must be finite. FP16/BF16 calculations are promoted to FP32; exact
    arithmetic claims are subject to output dtype rounding and representability.
    """
    if mode not in MODES or not math.isfinite(eps) or eps <= 0:
        raise ValueError('invalid PCPF mode or epsilon')
    _check_offset(offset)
    if (current.ndim != 4 or min(current.shape) < 1 or current.shape != update.shape
            or current.shape != anchor.shape or not current.is_floating_point()
            or current.dtype != update.dtype or not anchor.is_floating_point()
            or current.device != update.device or current.device != anchor.device):
        raise ValueError('inputs must share floating B,H,W,C shape/device; current/update dtypes must match')
    if mode == 'no_projection':
        return update * 0
    dtype = torch.float64 if current.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=current.device.type, enabled=False):
        c = haar2(current.detach().to(dtype), offset)
        a = anchor.detach().to(dtype)
        if mode == 'shifted_anchor':
            b, h, w, d = a.shape
            a = torch.roll(a.reshape(b, h*w, d), max(1, h*w//2), dims=1).reshape_as(a)
        a = haar2(a, offset)
        u = haar2(update.to(dtype), offset)
        unit, valid = _unit(c, eps)
        if mode == 'current_only':
            reliability = valid.to(dtype)
        else:
            anchor_unit, anchor_valid = _unit(a, eps)
            reliability = (unit * anchor_unit).sum(-1, keepdim=True).clamp(0, 1)
            reliability = reliability * (valid & anchor_valid).to(dtype)
        mask = detail_mask(current.shape[1], current.shape[2], offset, current.device)
        correction = mode_correction(u, unit, reliability * mask.to(dtype))
        return haar2(correction, offset).to(update.dtype)


class PCPFFusion(nn.Module):
    """Wrap existing plain fusion without adding parameters or RNG draws."""
    def __init__(self, fusion: nn.Module, *, mode: str = 'persistent',
                 offset: tuple[int, int] = (0, 0), eps: float = 1e-6):
        super().__init__()
        if mode not in MODES or not math.isfinite(eps) or eps <= 0:
            raise ValueError('invalid PCPF mode or epsilon')
        _check_offset(offset)
        self.fusion, self.mode, self.offset, self.eps = fusion, mode, offset, eps

    @property
    def message_out(self):
        return self.fusion.message_out

    @property
    def amplitude(self):
        return self.fusion.amplitude

    @property
    def disagreement_weight(self):
        return self.fusion.disagreement_weight

    def forward(self, block_output: Tensor, anchor: Tensor, grid: tuple[int, int], *,
                num_prefix_tokens: int = 1) -> Tensor:
        fused = self.fusion(block_output, anchor, grid, num_prefix_tokens=num_prefix_tokens)
        if self.mode == 'no_projection':
            return fused
        b, n, d = block_output.shape
        h, w = grid
        if n != h*w + num_prefix_tokens or anchor.shape != (b, h*w, d):
            raise ValueError('PCPF anchor/grid/prefix mismatch')
        current = block_output[:, num_prefix_tokens:]
        proposal = fused[:, num_prefix_tokens:] - current
        correction = contrast_correction(
            current.reshape(b,h,w,d), proposal.reshape(b,h,w,d),
            anchor.reshape(b,h,w,d), mode=self.mode, offset=self.offset, eps=self.eps)
        return torch.cat((fused[:, :num_prefix_tokens],
                          fused[:, num_prefix_tokens:] + correction.reshape(b,h*w,d)), dim=1)
