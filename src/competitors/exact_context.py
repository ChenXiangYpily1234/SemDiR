"""Algebraically factor the first gate convolution; retain every depth/tap gate.

No parameters or buffers are added. Convert an existing VariantDepthContext by
assigning its __class__ to FactorizedDepthContext and shared_context=False.
Floating point reduction regrouping can produce small roundoff differences.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F
from research.ad_pcpf_v2.variants import CachedDepthContext


class FactorizedDepthContext(CachedDepthContext):
    """Cache depth/support projections while preserving independent semantics."""

    def prepare_shared(self, features):
        # The parent computes the same normalized/detail tensors as the original
        # core, in the same pair and depth order, with detached measured support.
        if self.shared_context:
            raise ValueError('Factorized context requires independent semantic conditions')
        prepared = super().prepare_shared(features)
        first = features[0]
        with torch.autocast(device_type=first.device.type, enabled=False):
            convolution = self.gate[0]
            c = self.channels
            if convolution.kernel_size != (1, 1) or convolution.groups != 1:
                raise ValueError('Factorization requires the original dense 1x1 gate convolution')
            # Slicing the original parameter preserves its name, shape and full
            # gradient. A single 2C projection computes Wz*z + Wd*d per depth.
            weight = convolution.weight
            prepared['projected_depth'] = tuple(
                F.conv2d(torch.cat((z, d), dim=1), weight[:, :2*c])
                for z, d in zip(prepared['normalized'], prepared['details']))
            prepared['projected_support'] = F.conv2d(
                prepared['support'], weight[:, 3*c:], convolution.bias)
        return prepared

    def _select(self, prepared, semantic, diagnostics):
        first = prepared['features'][0]
        with torch.autocast(device_type=first.device.type, enabled=False):
            c = self.channels
            projected_semantic = F.conv2d(semantic, self.gate[0].weight[:, 2*c:3*c])
            detail = torch.zeros_like(first, dtype=torch.float32)
            gm, cr = [], []
            for w, projected, d in zip(prepared['weights'], prepared['projected_depth'], prepared['details']):
                hidden = projected + projected_semantic + prepared['projected_support']
                gate = self.gate[2](self.gate[1](hidden)).sigmoid()
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
