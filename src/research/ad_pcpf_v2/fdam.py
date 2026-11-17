"""MMSeg-1.x/timm matched FDAM adaptation, not an author-score reproduction.

AttInv and BOTH post-attention/post-MLP GroupDynamicScale operations follow
FDAM/FDAM_mmseg/mmcv_custom/deit_fdam.py (Linwei Chen; source revision and
redistribution authorization retained in FDAM/). No legacy mmcv.runner import.
DeiT-S/B retain their pretrained CLS token and original layer-scale policy;
FFT scales act on spatial tokens only. Native released DeiT-III drops CLS.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class StarReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.scale * x.relu().square() + self.bias


class GroupDynamicScale(nn.Module):
    def __init__(self, channels, groups=16, filters=4, size=14):
        super().__init__()
        if channels % groups:
            raise ValueError('FDAM channels must be divisible by groups')
        self.groups, self.filters = groups, filters
        self.reweight = nn.Sequential(nn.Linear(channels, int(channels * .0625), bias=False),
                                      StarReLU(), nn.Linear(int(channels * .0625), groups * filters, bias=False))
        self.complex_weights = nn.Parameter(torch.empty(filters, channels // groups, size, size // 2 + 1))
        nn.init.trunc_normal_(self.complex_weights, std=1e-5)

    def forward(self, x):
        dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            b, c, h, w = x.shape
            spectrum = torch.fft.rfft2(x, norm='ortho')
            routing = self.reweight(x.mean((2, 3))).reshape(b, self.groups, self.filters).tanh()
            weights = self.complex_weights.float()
            if weights.shape[-2:] != spectrum.shape[-2:]:
                weights = F.interpolate(weights, spectrum.shape[-2:], mode='bicubic', align_corners=True)
            modulation = torch.einsum('bgf,fchw->bgchw', routing, weights).reshape_as(spectrum.real)
            result = torch.fft.irfft2(spectrum * modulation, s=(h, w), norm='ortho')
        return result.to(dtype)


class AttentionWithAttInv(nn.Module):
    def __init__(self, base):
        super().__init__()
        if not isinstance(base.q_norm, nn.Identity) or not isinstance(base.k_norm, nn.Identity):
            raise ValueError('This matched FDAM port is registered for unnormalized-QK DeiT only')
        self.qkv, self.proj = base.qkv, base.proj
        self.attn_drop, self.proj_drop = base.attn_drop, base.proj_drop
        self.q_norm, self.k_norm = base.q_norm, base.k_norm
        self.num_heads, self.scale = base.num_heads, base.scale
        channels = self.proj.in_features
        self.dy_freq = nn.Linear(channels, self.num_heads)
        self.dy_freq_2 = nn.Linear(channels, self.num_heads)
        self.dy_freq_starrelu = StarReLU()
        self.lf_gamma = nn.Parameter(torch.full((channels,), 1e-5))
        self.hf_gamma = nn.Parameter(torch.full((channels,), 1e-5))

    def forward(self, x, attn_mask=None, **kwargs):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q * self.scale) @ k.transpose(-2, -1)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float('-inf')) if attn_mask.dtype == torch.bool else scores + attn_mask
        attention = self.attn_drop(scores.softmax(-1))
        context = (attention @ v).transpose(1, 2).reshape(b, n, c)
        values = v.transpose(1, 2).reshape(b, n, c)
        feat = self.dy_freq_starrelu(x)
        low = self.dy_freq_2(feat).tanh().repeat_interleave(c // self.num_heads, -1)
        high2 = F.softplus(self.dy_freq(feat)).square()
        high = (2 * high2 / (high2 + .3678)).repeat_interleave(c // self.num_heads, -1)
        out = context + context * low * self.lf_gamma + (values - context) * high * self.hf_gamma
        return self.proj_drop(self.proj(out))


class FDAMBlock(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.norm1, self.norm2, self.mlp = base.norm1, base.norm2, base.mlp
        self.ls1, self.ls2 = base.ls1, base.ls2
        self.drop_path1, self.drop_path2 = base.drop_path1, base.drop_path2
        self.attn = AttentionWithAttInv(base.attn)
        channels = base.attn.proj.in_features
        self.freq_scale_1, self.freq_scale_2 = GroupDynamicScale(channels), GroupDynamicScale(channels)
        self.grid, self.prefix_tokens = None, 1

    def _scale(self, tokens, module):
        if self.grid is None:
            raise RuntimeError('Backbone must set FDAM spatial grid before each forward')
        b, n, c = tokens.shape
        h, w = self.grid
        p = self.prefix_tokens
        if n != h * w + p:
            raise ValueError('FDAM token/grid mismatch; never assume a square grid')
        spatial = tokens[:, p:].transpose(1, 2).reshape(b, c, h, w)
        scaled = (spatial + module(spatial)).flatten(2).transpose(1, 2)
        return torch.cat((tokens[:, :p], scaled), 1)

    def forward(self, x, attn_mask=None, **kwargs):
        x = x + self.drop_path1(self.ls1(self._scale(self.attn(self.norm1(x), attn_mask), self.freq_scale_1)))
        return x + self.drop_path2(self.ls2(self._scale(self.mlp(self.norm2(x)), self.freq_scale_2)))


def is_fdam_extra(key):
    return any(part in key for part in ('.freq_scale_', '.dy_freq', '.lf_gamma', '.hf_gamma'))
