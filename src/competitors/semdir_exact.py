"""Active SemDiR-Exact: parameter-preserving algebraic computation sharing."""
import torch
from torch.nn import functional as F
from mmseg.registry import MODELS
from .models import CapacityMatchedPlainFusion,CompetitorM23Backbone
from research.ad_pcpf_v2.relay import SemanticPyramidRelay,pyramid_indices
from research.ad_pcpf_v2.core import validate_features
from research.pcpf_v6.core import contrast_correction
from .exact_context import FactorizedDepthContext
from .exact_rgb import refine_rgb_exact

class ExactPlainFusion(CapacityMatchedPlainFusion):
    """Shared spatial fusion, consuming the factorized RGB query when supplied."""


class ExactRelay(SemanticPyramidRelay):
    def route_residuals(self, base_features, actual_updates, refined):
        return (tuple(refined) if self.inject_pyramid else base_features), actual_updates

    def forward(self, image, maps, indices, neck):
        validate_features(maps,self.channels)
        if tuple(indices) != pyramid_indices(len(maps)):
            raise ValueError('Semantic taps must match the actual encoder depth')
        base_features = tuple(maps[index] for index in indices)
        shared = (self.context.prepare_shared(maps)
                  if self.context.gate_mode == 'learned' and self.context.support_mode == 'measured' else None)
        refined, diagnostics, supports = [], [], []
        ratios, clipped_fractions, actual_updates = [], [], []
        shared_proposal_input = None
        for branch, feature in enumerate(base_features):
            # Shared selector parameters, branch-specific semantic conditioning.
            # Every branch consults ALL L blocks, with no shallow-first prior.
            if shared is not None:
                anchor,detail,support,diagnostic = self.context.forward_prepared(
                    shared,indices[branch],return_diagnostics=True)
            else:
                anchor,detail,support,diagnostic = self.context(maps,feature,return_diagnostics=True)
            if self.novelty:
                detail = self.remove_redundant_direction(detail, feature)
            if self.shared_context:
                if shared_proposal_input is None:
                    shared_proposal_input = F.gelu(self.coarse_dw(detail.to(feature.dtype)))
                proposal = self.coarse_out[branch](shared_proposal_input)
            else:
                proposal = self.coarse_out[branch](F.gelu(self.coarse_dw(detail.to(feature.dtype))))
            proposal = proposal.to(feature.dtype)
            proposal = proposal-proposal.mean((-2,-1),keepdim=True)
            if self.use_projection:
                correction = contrast_correction(feature.movedim(1,-1).contiguous(),
                    proposal.movedim(1,-1).contiguous(), anchor.movedim(1,-1).contiguous(),
                    mode='persistent',offset=(0,0))
                proposal = proposal+correction.movedim(-1,1)
            # A positive scale preserves the corrected direction. This is a
            # separate projection/correction for each original semantic input.
            if self.branch_budget is None and self.residual_scale == 1.:
                update = self.alpha[branch].to(feature.dtype)*proposal
                refined.append(feature+update)
            else:
                update = self.alpha[branch].float() * proposal.float() * self.residual_scale
                if self.branch_budget is not None:
                    update, ratio, clipped = self.bound_update(update, feature, self.branch_budget[branch])
                    ratios.append(ratio)
                    clipped_fractions.append(clipped)
                refined.append((feature.float() + update).to(feature.dtype))
            actual_updates.append(update)
            supports.append(support)
            diagnostics.append(diagnostic)
        self.last_diagnostics = dict(beta=diagnostics[0]['beta'],weights=diagnostics[0]['weights'],
            gate_mean=torch.stack([d['gate_mean'] for d in diagnostics]),
            contribution_rms=torch.stack([d['contribution_rms'] for d in diagnostics]),
            alpha=self.alpha.detach(),semantic_indices=torch.tensor(indices,device=maps[0].device))
        if ratios:
            self.last_diagnostics.update(update_ratio_max=torch.stack(ratios),
                budget_active_fraction=torch.stack(clipped_fractions))
        pyramid_inputs, condition_updates = self.route_residuals(base_features, actual_updates, refined)
        pyramid = neck(pyramid_inputs)
        if len(pyramid) != 4:
            raise RuntimeError('Expected four pyramid outputs')
        if hasattr(self, "transport"):
            high, rgb, query = refine_rgb_exact(self, image, pyramid[0], supports[0])
            extra = self.transport(condition_updates, rgb, q=query)
            self.last_diagnostics.update(self.transport.last_diagnostics)
            return (high + extra, *pyramid[1:])
        return (self.refine_highres(image,pyramid[0],supports[0]),*pyramid[1:])


def supports_exact_relay(relay):
    """Identify the unchanged Full contract; retain other experiment semantics."""
    return (type(relay) is SemanticPyramidRelay
        and not relay.novelty and relay.branch_budget is None and relay.rgb_budget is None
        and relay.residual_scale == 1. and not relay.shared_context and relay.inject_pyramid
        and relay.use_projection and relay.use_attenuation and relay.use_rgb
        and not relay.plain_cnn
        and type(getattr(relay, 'transport', None)) is CapacityMatchedPlainFusion
        and getattr(relay.transport, 'conditioning', 'both') == 'both'
        and relay.context.gate_mode == 'learned' and relay.context.support_mode == 'measured')


def convert_relay(relay):
    """Promote in place without constructing tensors or changing checkpoint keys."""
    if not supports_exact_relay(relay):
        raise ValueError('Exact requires every unchanged Full branch and condition')
    relay.context.__class__ = FactorizedDepthContext
    relay.context.shared_context = False
    relay.context_cache = True
    relay.transport.__class__ = ExactPlainFusion
    relay.__class__ = ExactRelay
    return relay


@MODELS.register_module()
class SemDiRExactBackbone(CompetitorM23Backbone):
    """Explicit active entrypoint; legacy Full configs auto-promote identically."""
    def __init__(self, **kwargs):
        kwargs.setdefault('fusion', 'plain')
        super().__init__(**kwargs)
        if not isinstance(self.relay, ExactRelay):
            raise ValueError('SemDiR-Exact requires unchanged plain fusion with both conditions')
