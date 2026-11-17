"""M23: additive, patch-conservative coarse-to-fine residual transport."""
import math
import torch
from torch import nn


class CoarseFineResidualTransport(nn.Module):
    def __init__(self, channels, hidden=32, temperature=1., *, rgb_channels=None,
                 output_channels=None):
        super().__init__()
        if hidden < 1 or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Positive hidden dimension and temperature required')
        self.channels = int(channels)
        self.rgb_channels = self.channels if rgb_channels is None else int(rgb_channels)
        self.output_channels = self.rgb_channels if output_channels is None else int(output_channels)
        self.coarse = nn.Conv2d(4 * channels, hidden, 1)
        self.norm = nn.LayerNorm(hidden)
        self.rgb = nn.Conv2d(self.rgb_channels, hidden, 1)
        self.value = nn.Conv2d(hidden, self.output_channels, 1, bias=False)
        self.score = nn.Sequential(nn.Conv2d(3 * hidden + 2, hidden, 1), nn.GELU(),
                                   nn.Conv2d(hidden, 1, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.temperature = float(temperature)
        self.last_diagnostics = {}

    @staticmethod
    def geometry(height, width, h, w, device):
        if height < h or width < w:
            raise ValueError('High-resolution grid must cover every patch')
        y, x = torch.arange(height, device=device), torch.arange(width, device=device)
        gy, gx = y * h // height, x * w // width
        ids = (gy[:, None] * w + gx[None, :]).reshape(-1)
        # Disjoint bins, including odd shapes; no adaptive-pool overlap.
        sy, ey = (gy * height + h - 1) // h, ((gy + 1) * height + h - 1) // h
        sx, ex = (gx * width + w - 1) // w, ((gx + 1) * width + w - 1) // w
        py = 2 * (y - sy + .5) / (ey - sy) - 1
        px = 2 * (x - sx + .5) / (ex - sx) - 1
        pos = torch.stack((py[:, None].expand(height, width), px[None, :].expand(height, width)))
        return ids, pos

    def forward(self, residuals, rgb):
        if (len(residuals) != 4 or any(r.shape != residuals[0].shape for r in residuals)
                or residuals[0].shape[1] != self.channels):
            raise ValueError('Four aligned actual coarse residuals required')
        if rgb.shape[1] != self.rgb_channels:
            raise ValueError('RGB residual channels do not match the registered transport')
        c = self.coarse(torch.cat(residuals, 1))
        c = self.norm(c.movedim(1, -1)).movedim(-1, 1)
        b, _, h, w = c.shape
        height, width = rgb.shape[-2:]
        ids, pos = self.geometry(height, width, h, w, rgb.device)
        expanded = c.flatten(2).index_select(2, ids).reshape(b, -1, height, width)
        q = self.rgb(rgb)
        score = self.score(torch.cat((expanded*q, (expanded-q).abs(), q,
                                      pos[None].expand(b, -1, -1, -1).to(q.dtype)), 1))
        # The probability and exact group centering use FP32 under BF16 autocast.
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            logits = score.float().flatten(2) / self.temperature
            index = ids[None, None].expand(b, 1, -1)
            maximum = logits.new_full((b, 1, h*w), -float('inf'))
            maximum = maximum.scatter_reduce(2, index, logits, reduce='amax', include_self=True)
            exp = (logits - maximum.gather(2, index)).exp()
            denom = exp.new_zeros((b, 1, h*w)).scatter_add(2, index, exp)
            counts = exp.new_zeros((1, 1, h*w)).scatter_add(
                2, ids[None, None], torch.ones_like(ids, dtype=exp.dtype)[None, None])
            weights = exp / denom.gather(2, index) - counts.reciprocal().gather(2, ids[None, None])
            # Correct roundoff without subtracting anything from the existing RGB update.
            mean = weights.new_zeros((b, 1, h*w)).scatter_add(2, index, weights) / counts
            weights = weights - mean.gather(2, index)
        value = self.value(c).flatten(2).index_select(2, ids)
        extra = (weights * value.float()).reshape(b, -1, height, width).to(rgb.dtype)
        self.last_diagnostics = dict(
            transport_update_rms=extra.detach().float().square().mean().sqrt(),
            transport_value_rms=value.detach().float().square().mean().sqrt(),
            transport_allocation_rms=weights.detach().square().mean().sqrt())
        return extra
