"""Factor the Full relay RGB update and transport query through its small stem.

No parameters or buffers are registered or cached. The two existing 1x1 weight
matrices are composed on every call, retaining gradients to both. Algebraic
identity is over real numbers, not floating-point operation order: native code
centers the expanded update and performs two rounded convolutions for Q, while
this code centers in stem channels and uses one convolution. In particular BF16
is expected to differ. Composition runs in FP32 outside autocast (FP64 for FP64
weights), then the convolution follows the surrounding autocast/input precision.
The original local-detail, detached impulse signal, soft attenuation and gate
input are kept verbatim. Full model forward still owns Haar/coarse transport.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from research.pcpf_v7.core import impulse_indicator, local_detail


def _require_pointwise(conv, name, *, bias_allowed):
    if not isinstance(conv, nn.Conv2d) or (conv.kernel_size != (1, 1)
            or conv.stride != (1, 1) or conv.padding != (0, 0)
            or conv.dilation != (1, 1) or conv.groups != 1
            or (not bias_allowed and conv.bias is not None)):
        raise ValueError(f'{name} must be a dense 1x1 convolution'
                         + ('' if bias_allowed else ' without bias'))


def refine_rgb_exact(relay, image, high, support):
    """Return ``(high + H, H, Q)`` using the existing Full relay's parameters.

    ``H = W_H X`` and ``Q = (W_Q W_H) X + b_Q`` where
    ``X = gate * filtered - mean_HW(gate * filtered)``. Unsupported ablations
    raise explicitly rather than silently changing their semantics. Input
    tensors remain differentiable according to the native implementation,
    including its FP32 support interpolation and detached outlier indicator.
    """
    if (not getattr(relay, 'use_rgb', False)
            or getattr(relay, 'plain_cnn', False)
            or not getattr(relay, 'use_attenuation', False)
            or getattr(relay, 'readout_only', False)
            or getattr(relay, 'rgb_budget', None) is not None
            or getattr(relay, 'residual_scale', 1.) != 1.):
        raise ValueError('Exact RGB requires the unbudgeted Full RGB path with attenuation')
    if not hasattr(relay, 'transport') or not hasattr(relay.transport, 'rgb'):
        raise ValueError('Exact RGB requires the existing transport.rgb projection')
    h_conv, q_conv = relay.detail_out, relay.transport.rgb
    _require_pointwise(h_conv, 'detail_out', bias_allowed=False)
    _require_pointwise(q_conv, 'transport.rgb', bias_allowed=True)
    if h_conv.out_channels != q_conv.in_channels:
        raise ValueError('RGB update and transport query channel dimensions must agree')
    if any(t.ndim != 4 or min(t.shape) < 1 for t in (image, high, support)):
        raise ValueError('Expected nonempty NCHW image, high and support tensors')
    if (image.shape[0] != high.shape[0] or support.shape[0] != high.shape[0]
            or support.shape[1] != 1 or high.shape[1] != h_conv.out_channels):
        raise ValueError('RGB input batch, support or high channel dimensions do not agree')
    detail = local_detail(relay.detail_dw(relay.stem(image)))
    if detail.shape[-2:] != high.shape[-2:]:
        detail = F.interpolate(detail, high.shape[-2:], mode='bilinear', align_corners=False)
    semantic = relay.semantic_proj(high)
    confidence = F.interpolate(support.float(), detail.shape[-2:], mode='nearest').to(detail.dtype)
    outlier = impulse_indicator(detail).to(detail.dtype)
    filtered = detail * (1. - .75 * (1. - confidence) * outlier)
    gate = relay.detail_gate(torch.cat((filtered, semantic, confidence, outlier), 1)).sigmoid()
    gated = gate * filtered
    x = gated - gated.mean((-2, -1), keepdim=True)
    h = h_conv(x)
    with torch.autocast(device_type=x.device.type, enabled=False):
        compose_dtype = torch.float64 if h_conv.weight.dtype == torch.float64 else torch.float32
        composed = (q_conv.weight[:, :, 0, 0].to(compose_dtype)
                    @ h_conv.weight[:, :, 0, 0].to(compose_dtype))[:, :, None, None]
    # Casting keeps a valid convolution even for explicitly low-precision models
    # outside autocast; .to preserves the graph back to both master matrices.
    bias = None if q_conv.bias is None else q_conv.bias.to(x.dtype)
    q = F.conv2d(x, composed.to(x.dtype), bias)
    return high + h, h, q
