"""Task-conditioned all-depth detail selection for same-grid plain ViTs.

The depth preference is learnable, not a claim that shallow features are more
semantic. The ungated anchor and gated details MUST remain separate tensors.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def local_detail(x: Tensor) -> Tensor:
    if x.ndim != 4 or min(x.shape) < 1 or not x.is_floating_point():
        raise ValueError('Expected nonempty floating BCHW')
    return x - F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)


def depth_weights(depth: int, beta: Tensor) -> Tensor:
    """FP32 weights; never convert beta to float/item in the gradient path."""
    if type(depth) is not int or depth < 2:
        raise ValueError('At least two blocks are required')
    if not isinstance(beta, Tensor) or beta.ndim != 0 or not beta.is_floating_point():
        raise ValueError('beta must be a floating scalar tensor')
    t = torch.linspace(0, 1, depth, device=beta.device, dtype=torch.float32)
    return torch.softmax(-beta.float() * t, dim=0)


def channel_norm(x: Tensor) -> Tensor:
    return F.layer_norm(x.float().movedim(1, -1), (x.shape[1],),
                        eps=1e-5).movedim(-1, 1)


def validate_features(features: Sequence[Tensor], channels: int) -> Tensor:
    if not isinstance(features, (tuple, list)) or len(features) < 2:
        raise ValueError('Provide at least two same-grid feature maps')
    first = features[0]
    if (not isinstance(first, Tensor) or first.ndim != 4 or min(first.shape) < 1
            or not first.is_floating_point() or first.shape[1] != channels):
        raise ValueError('Expected nonempty floating BCHW with configured channels')
    for x in features:
        if (not isinstance(x, Tensor) or x.shape != first.shape
                or x.device != first.device or x.dtype != first.dtype):
            raise ValueError('All block features must share shape, device and dtype')
    return first


@torch.no_grad()
def cross_depth_support(features: Sequence[Tensor], weights: Tensor,
                        eps: float = 1e-6) -> Tensor:
    """Detached descriptive consistency, not a semantic correctness score.

    A bounded queue avoids storing all detached detail maps. Recompute
    normalized features in the differentiable pass rather than retaining two
    complete depth stacks. Pair weights, including their beta path, detach.
    """
    gap = max(1, len(features) // 4)
    history = deque(maxlen=gap)
    pair_weights = weights[:-gap].detach()
    pair_weights = pair_weights / pair_weights.sum()
    support = torch.zeros_like(features[0][:, :1], dtype=torch.float32)
    for index, x in enumerate(features):
        d = local_detail(channel_norm(x.detach()))
        if index >= gap:
            old = history[0]
            valid = ((old.square().sum(1, keepdim=True) > eps**2)
                     & (d.square().sum(1, keepdim=True) > eps**2))
            score = F.cosine_similarity(old, d, dim=1, eps=eps)[:, None]
            support.add_(pair_weights[index-gap] * score.clamp(0, 1) * valid)
        history.append(d)
    return support.clamp(0, 1).detach()


class TaskGatedDepthContext(nn.Module):
    """beta=bound*tanh(raw_beta); per-layer shared, semantic-conditioned gate.

    Initial beta=0 and initial gate=0.5. Gate hidden weights are NOT zero.
    Gated details are NOT renormalized across depth: all candidates may be
    attenuated simultaneously. Norm bounds concern aggregation only.
    """
    def __init__(self, channels: int, gate_hidden: int = 32,
                 beta_init: float = 0., beta_bound: float = 4., eps: float = 1e-6):
        super().__init__()
        if type(channels) is not int or channels < 2:
            raise ValueError('channels must be an integer >= 2')
        if type(gate_hidden) is not int or gate_hidden < 1:
            raise ValueError('gate_hidden must be a positive integer')
        if (isinstance(beta_init, bool) or isinstance(beta_bound, bool)
                or not math.isfinite(beta_init) or not math.isfinite(beta_bound)
                or not 0 < beta_bound or not abs(beta_init) < beta_bound
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError('Invalid beta initialization/bound or epsilon')
        self.channels, self.beta_bound = channels, float(beta_bound)
        self.beta_init, self.eps = float(beta_init), eps
        self.raw_beta = nn.Parameter(torch.tensor(math.atanh(beta_init / beta_bound)))
        self.gate = nn.Sequential(nn.Conv2d(3*channels+1, gate_hidden, 1),
                                  nn.GELU(), nn.Conv2d(gate_hidden, 1, 1))
        self.reset_context()

    @property
    def beta(self) -> Tensor:
        return self.beta_bound * self.raw_beta.tanh()

    @torch.no_grad()
    def reset_context(self) -> None:
        self.raw_beta.fill_(math.atanh(self.beta_init / self.beta_bound))
        self.gate[-1].weight.zero_()
        self.gate[-1].bias.zero_()

    def prepare_shared(self, features):
        """Reuse only detached support; learned paths keep their original order."""
        first = validate_features(features,self.channels)
        with torch.no_grad(),torch.autocast(device_type=first.device.type,enabled=False):
            support = cross_depth_support(features,depth_weights(len(features),self.beta),self.eps)
        return features,support

    def forward_prepared(self, prepared, semantic_index, *, return_diagnostics=False):
        features,support = prepared
        return TaskGatedDepthContext.forward(self,features,features[semantic_index],
            return_diagnostics=return_diagnostics,shared_support=support)

    def forward(self, features: Sequence[Tensor], semantic: Tensor,
                *, return_diagnostics: bool = False, shared_support=None):
        first = validate_features(features, self.channels)
        if (semantic.shape != first.shape or semantic.device != first.device
                or not semantic.is_floating_point()):
            raise ValueError('Semantic map must share BCHW shape and device')
        if self.raw_beta.device != first.device:
            raise ValueError('Move the context module and features to the same device')
        with torch.autocast(device_type=first.device.type, enabled=False):
            weights = depth_weights(len(features), self.beta)
            support = (cross_depth_support(features, weights, self.eps)
                       if shared_support is None else shared_support)
            semantic_norm = channel_norm(semantic)
            anchor = torch.zeros_like(first, dtype=torch.float32)
            detail = torch.zeros_like(anchor)
            gate_means, contributions = [], []
            for index, x in enumerate(features):
                z = channel_norm(x)
                d = local_detail(z)
                gate = self.gate(torch.cat((z, d, semantic_norm, support), dim=1)).sigmoid()
                contribution = weights[index] * gate * d
                anchor = anchor + weights[index] * z
                detail = detail + contribution
                if return_diagnostics:
                    gate_means.append(gate.detach().mean())
                    contributions.append(contribution.detach().square().mean().sqrt())
        if return_diagnostics:
            diagnostics = dict(beta=self.beta.detach(), weights=weights.detach(),
                               gate_mean=torch.stack(gate_means),
                               contribution_rms=torch.stack(contributions))
            return anchor, detail, support, diagnostics
        return anchor, detail, support
