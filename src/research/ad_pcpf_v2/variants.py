"""Explicit V2 interventions; default core remains the original task-gated V2."""
from __future__ import annotations
import math
import torch
from .core import TaskGatedDepthContext, validate_features, depth_weights, channel_norm, local_detail, cross_depth_support


class VariantDepthContext(TaskGatedDepthContext):
    def __init__(self, channels: int, gate_hidden: int = 32, beta_init: float = 0.,
                 beta_bound: float = 4., fixed_beta=None, gate_mode: str = 'learned',
                 support_mode: str = 'measured'):
        if gate_mode not in ('learned', 'half', 'one') or support_mode not in ('measured', 'one'):
            raise ValueError('Unknown gate/support intervention')
        if fixed_beta is not None:
            beta_init = float(fixed_beta)
        super().__init__(channels, gate_hidden, beta_init, beta_bound)
        self.gate_mode, self.support_mode = gate_mode, support_mode
        self.raw_beta.requires_grad_(fixed_beta is None)
        if gate_mode != 'learned':
            self.gate = None  # No unused trainable parameters in DDP.

    @torch.no_grad()
    def reset_context(self):
        self.raw_beta.fill_(math.atanh(self.beta_init / self.beta_bound))
        if self.gate is not None:
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.zero_()

    def forward(self, features, semantic, *, return_diagnostics=False):
        if self.gate_mode == 'learned' and self.support_mode == 'measured':
            return super().forward(features, semantic, return_diagnostics=return_diagnostics)
        first = validate_features(features, self.channels)
        if semantic.shape != first.shape or semantic.device != first.device:
            raise ValueError('Semantic map and layer maps must align')
        with torch.autocast(device_type=first.device.type, enabled=False):
            weights = depth_weights(len(features), self.beta)
            support = (cross_depth_support(features, weights, self.eps)
                       if self.support_mode == 'measured' else torch.ones_like(first[:, :1], dtype=torch.float32))
            semantic_norm = channel_norm(semantic)
            anchor = torch.zeros_like(first, dtype=torch.float32)
            detail = torch.zeros_like(anchor)
            gm, cr = [], []
            for i, feature in enumerate(features):
                z = channel_norm(feature)
                d = local_detail(z)
                g = (self.gate(torch.cat((z, d, semantic_norm, support), 1)).sigmoid()
                     if self.gate is not None else torch.full_like(support, .5 if self.gate_mode == 'half' else 1.))
                contribution = weights[i] * g * d
                anchor = anchor + weights[i] * z
                detail = detail + contribution
                if return_diagnostics:
                    gm.append(g.detach().mean())
                    cr.append(contribution.detach().square().mean().sqrt())
        if return_diagnostics:
            return anchor, detail, support.detach(), dict(beta=self.beta.detach(), weights=weights.detach(),
                                                         gate_mean=torch.stack(gm), contribution_rms=torch.stack(cr))
        return anchor, detail, support.detach()


class CachedDepthContext(VariantDepthContext):
    """Staged candidates; identical parameters, cached all-layer local maps.

    shared_context uses the mean of the four raw semantic taps as a single
    condition. Retain the existing sigmoid and beta weights so the screening
    changes sharing alone, not the aggregation's amplitude normalization.
    """
    def __init__(self, *args, shared_context=False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.gate_mode != 'learned' or self.support_mode != 'measured':
            raise ValueError('Cached context requires learned gates and measured support')
        self.shared_context = bool(shared_context)

    def prepare_shared(self, features):
        first = validate_features(features, self.channels)
        with torch.autocast(device_type=first.device.type, enabled=False):
            weights = depth_weights(len(features), self.beta)
            normalized = tuple(channel_norm(x) for x in features)
            details = tuple(local_detail(x) for x in normalized)
            anchor = torch.zeros_like(first, dtype=torch.float32)
            for w, z in zip(weights, normalized):
                anchor = anchor + w * z
            with torch.no_grad():
                gap = max(1, len(features) // 4)
                pair_weights = weights[:-gap].detach()
                pair_weights = pair_weights / pair_weights.sum()
                support = torch.zeros_like(first[:, :1], dtype=torch.float32)
                for i, (old, new) in enumerate(zip(details[:-gap], details[gap:])):
                    old, new = old.detach(), new.detach()
                    valid = ((old.square().sum(1, keepdim=True) > self.eps**2)
                             & (new.square().sum(1, keepdim=True) > self.eps**2))
                    score = torch.nn.functional.cosine_similarity(old, new, dim=1, eps=self.eps)[:, None]
                    support.add_(pair_weights[i] * score.clamp(0, 1) * valid)
                support = support.clamp(0, 1).detach()
            prepared = dict(features=features, normalized=normalized, details=details,
                            weights=weights, anchor=anchor, support=support)
            if self.shared_context:
                taps = [math.ceil(len(features)*q/4)-1 for q in (1,2,3,4)]
                semantic = channel_norm(sum(features[k] for k in taps)/4)
                prepared['shared_result'] = self._select(prepared, semantic, True)
        return prepared

    def _select(self, prepared, semantic, diagnostics):
        first = prepared['features'][0]
        with torch.autocast(device_type=first.device.type, enabled=False):
            detail = torch.zeros_like(first, dtype=torch.float32)
            gm, cr = [], []
            for w, z, d in zip(prepared['weights'], prepared['normalized'], prepared['details']):
                gate = self.gate(torch.cat((z, d, semantic, prepared['support']), 1)).sigmoid()
                contribution = w * gate * d
                detail = detail + contribution
                if diagnostics:
                    gm.append(gate.detach().mean())
                    cr.append(contribution.detach().square().mean().sqrt())
        out = (prepared['anchor'], detail, prepared['support'])
        if diagnostics:
            return (*out, dict(beta=self.beta.detach(), weights=prepared['weights'].detach(),
                              gate_mean=torch.stack(gm), contribution_rms=torch.stack(cr)))
        return out

    def forward_prepared(self, prepared, semantic_index, *, return_diagnostics=False):
        if not 0 <= semantic_index < len(prepared['features']):
            raise ValueError('Invalid semantic tap')
        if self.shared_context:
            return prepared['shared_result'] if return_diagnostics else prepared['shared_result'][:3]
        return self._select(prepared, prepared['normalized'][semantic_index], return_diagnostics)

    def forward(self, features, semantic, *, return_diagnostics=False):
        prepared = self.prepare_shared(features)
        if self.shared_context:
            return prepared['shared_result'] if return_diagnostics else prepared['shared_result'][:3]
        return self._select(prepared, channel_norm(semantic), return_diagnostics)
