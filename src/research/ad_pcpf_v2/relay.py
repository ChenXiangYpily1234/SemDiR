"""Primary R9 semantic-pyramid relay and legacy R8 compatibility implementation."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from research.pcpf_v6.core import contrast_correction
from research.pcpf_v7.core import impulse_indicator, local_detail
from .variants import VariantDepthContext, CachedDepthContext
from .core import validate_features, channel_norm


class TaskGatedAllDepthRelay(nn.Module):
    """Legacy CLS-fusion relay, retained only for earlier experiment IDs."""
    def __init__(self, channels=384, stem_channels=48, gate_hidden=32, beta_init=0., beta_bound=4.,
                 fixed_beta=None, gate_mode='learned', support_mode='measured',
                 use_projection=True, use_rgb=True, use_attenuation=True, plain_cnn=False,
                 readout_only=False):
        super().__init__()
        self.readout_only = bool(readout_only)
        self.last_diagnostics = {}
        if self.readout_only:
            self.context = None
            return
        self.use_projection, self.use_rgb = bool(use_projection), bool(use_rgb)
        self.use_attenuation, self.plain_cnn = bool(use_attenuation), bool(plain_cnn)
        self.coarse_dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.coarse_out = nn.Conv2d(channels, channels, 1, bias=False)
        # Construct in the same order even for no-RGB so remaining new modules
        # receive the same initial random tensors. Remove unused modules below.
        self.stem = nn.Sequential(nn.Conv2d(3, stem_channels, 3, 2, 1), nn.GELU(),
                                  nn.Conv2d(stem_channels, stem_channels, 3, 2, 1), nn.GELU())
        self.detail_dw = nn.Conv2d(stem_channels, stem_channels, 3, padding=1,
                                   groups=stem_channels, bias=False)
        self.semantic_proj = nn.Conv2d(channels, stem_channels, 1, bias=False)
        self.detail_gate = nn.Conv2d(2 * stem_channels + 2, 1, 1)
        self.detail_out = nn.Conv2d(stem_channels, channels, 1, bias=False)
        self.context = VariantDepthContext(channels, gate_hidden, beta_init, beta_bound,
                                           fixed_beta, gate_mode, support_mode)
        self.reset_relay()
        if not self.use_rgb:
            for name in ('stem', 'detail_dw', 'semantic_proj', 'detail_gate', 'detail_out'):
                delattr(self, name)

    @torch.no_grad()
    def reset_relay(self):
        if self.readout_only:
            return
        nn.init.dirac_(self.coarse_dw.weight, groups=self.coarse_dw.groups)
        self.coarse_out.weight.zero_()
        if self.use_rgb:
            nn.init.dirac_(self.detail_dw.weight, groups=self.detail_dw.groups)
            self.detail_out.weight.zero_()
            self.detail_gate.weight.zero_()
            self.detail_gate.bias.zero_()
        self.context.reset_context()

    def refine_highres(self, image, high, support, return_update=False):
        if not self.use_rgb:
            return high
        detail = local_detail(self.detail_dw(self.stem(image)))
        if detail.shape[-2:] != high.shape[-2:]:
            detail = F.interpolate(detail, high.shape[-2:], mode='bilinear', align_corners=False)
        semantic = self.semantic_proj(high)
        if self.plain_cnn:
            # Clamp before sqrt: exact zero-detail locations otherwise have an
            # infinite derivative and can poison the plain-CNN ablation's
            # backward pass even while its losses remain finite.
            magnitude = detail.float().square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
            average = F.avg_pool2d(magnitude, 3, 1, 1, count_include_pad=False)
            filtered = detail
            signals = (magnitude.to(detail.dtype), average.to(detail.dtype))
        else:
            confidence = F.interpolate(support.float(), detail.shape[-2:], mode='nearest').to(detail.dtype)
            outlier = impulse_indicator(detail).to(detail.dtype)
            filtered = detail * (1. - .75 * (1. - confidence) * outlier) if self.use_attenuation else detail
            signals = (confidence, outlier)
        gate = self.detail_gate(torch.cat((filtered, semantic, *signals), 1)).sigmoid()
        update = self.detail_out(filtered) * gate
        result = high + update - update.mean((-2, -1), keepdim=True)
        return (result, update - update.mean((-2, -1), keepdim=True)) if return_update else result

    def forward(self, image, maps, fused, neck):
        if self.readout_only:
            pyramid = neck((fused,) * 4)
            if len(pyramid) != 4:
                raise RuntimeError('Expected four pyramid outputs')
            return pyramid
        anchor, detail, support, diagnostics = self.context(maps, fused, return_diagnostics=True)
        self.last_diagnostics = diagnostics
        proposal = self.coarse_out(F.gelu(self.coarse_dw(detail.to(fused.dtype)))).to(fused.dtype)
        proposal = proposal - proposal.mean((-2, -1), keepdim=True)
        if self.use_projection:
            correction = contrast_correction(fused.movedim(1, -1).contiguous(),
                proposal.movedim(1, -1).contiguous(), anchor.movedim(1, -1).contiguous(),
                mode='persistent', offset=(0, 0))
            proposal = proposal + correction.movedim(-1, 1)
        pyramid = neck((fused + proposal,) * 4)
        if len(pyramid) != 4:
            raise RuntimeError('Expected four pyramid outputs')
        return (self.refine_highres(image, pyramid[0], support), *pyramid[1:])


def pyramid_indices(depth):
    """Zero-based quarter-depth taps; require four distinct semantic inputs."""
    if type(depth) is not int or depth < 4:
        raise ValueError('A four-branch semantic pyramid requires at least four blocks')
    return tuple(math.ceil(depth*q/4)-1 for q in (1,2,3,4))


class SemanticPyramidRelay(nn.Module):
    def __init__(self, channels=384, stem_channels=48, gate_hidden=32,
                 beta_init=0., beta_bound=4., fixed_beta=None, gate_mode='learned',
                 support_mode='measured', use_projection=True, use_rgb=True,
                 use_attenuation=True, plain_cnn=False, novelty=False,
                 branch_budget=None, rgb_budget=None, residual_scale=1., fusion_eps=1e-6,
                 transport=False, transport_hidden=32, context_cache=False,
                 shared_context=False, inject_pyramid=True):
        super().__init__()
        self.novelty = bool(novelty)
        self.fusion_eps = float(fusion_eps)
        self.residual_scale = float(residual_scale)
        if not math.isfinite(self.fusion_eps) or self.fusion_eps <= 0:
            raise ValueError('fusion_eps must be finite and positive')
        if not math.isfinite(self.residual_scale) or not 0 <= self.residual_scale <= 1:
            raise ValueError('residual_scale must be in [0, 1]')
        if branch_budget is not None:
            if isinstance(branch_budget, (float, int)):
                branch_budget = (float(branch_budget),) * 4
            branch_budget = tuple(float(x) for x in branch_budget)
            if len(branch_budget) != 4 or any(not math.isfinite(x) or x <= 0 for x in branch_budget):
                raise ValueError('branch_budget requires four finite positive values')
        if rgb_budget is not None:
            rgb_budget = float(rgb_budget)
            if not math.isfinite(rgb_budget) or rgb_budget <= 0:
                raise ValueError('rgb_budget must be finite and positive')
        self.branch_budget, self.rgb_budget = branch_budget, rgb_budget
        self.channels = channels
        if any(type(v) is not bool for v in (context_cache, shared_context, inject_pyramid)):
            raise ValueError('Context and pyramid switches must be boolean')
        if shared_context and not context_cache:
            raise ValueError('Shared aggregation requires the cached context')
        self.context_cache, self.shared_context = context_cache, shared_context
        self.inject_pyramid = inject_pyramid
        self.use_projection, self.use_rgb = bool(use_projection), bool(use_rgb)
        self.use_attenuation, self.plain_cnn = bool(use_attenuation), bool(plain_cnn)
        self.coarse_dw = nn.Conv2d(channels,channels,3,padding=1,groups=channels,bias=False)
        self.coarse_out = nn.ModuleList(nn.Conv2d(channels,channels,1,bias=False) for _ in range(4))
        # Positive branch scales, initialized to 1. Residual projections alone
        # start at zero: zeroing both factors would kill their learning path.
        self.raw_alpha = nn.Parameter(torch.zeros(4))
        context_type = CachedDepthContext if context_cache else VariantDepthContext
        context_args = dict(shared_context=shared_context) if context_cache else {}
        self.context = context_type(channels,gate_hidden,beta_init,beta_bound,
                                    fixed_beta,gate_mode,support_mode,**context_args)
        if self.use_rgb:
            self.stem = nn.Sequential(nn.Conv2d(3,stem_channels,3,2,1),nn.GELU(),
                                      nn.Conv2d(stem_channels,stem_channels,3,2,1),nn.GELU())
            self.detail_dw = nn.Conv2d(stem_channels,stem_channels,3,padding=1,
                                       groups=stem_channels,bias=False)
            self.semantic_proj = nn.Conv2d(channels,stem_channels,1,bias=False)
            self.detail_gate = nn.Conv2d(2*stem_channels+2,1,1)
            self.detail_out = nn.Conv2d(stem_channels,channels,1,bias=False)
        self.last_diagnostics = {}
        self.reset_relay()
        if transport:
            if not use_rgb or novelty or branch_budget is not None or rgb_budget is not None or residual_scale != 1.:
                raise ValueError("CFRT requires the unchanged M21 residual paths")
            from .transport import CoarseFineResidualTransport
            self.transport = CoarseFineResidualTransport(channels, transport_hidden)

    @property
    def alpha(self):
        return 2*self.raw_alpha.sigmoid()

    @torch.no_grad()
    def reset_relay(self):
        nn.init.dirac_(self.coarse_dw.weight,groups=self.coarse_dw.groups)
        for projection in self.coarse_out:
            projection.weight.zero_()
        self.raw_alpha.zero_()
        self.context.reset_context()
        if self.use_rgb:
            nn.init.dirac_(self.detail_dw.weight,groups=self.detail_dw.groups)
            self.detail_out.weight.zero_()
            self.detail_gate.weight.zero_()
            self.detail_gate.bias.zero_()

    def remove_redundant_direction(self, candidate, feature):
        """Per-pixel channel projection, before the learned output projection.

        Both inputs remain differentiable. Orthogonality is approximate because
        of epsilon, and does not survive arbitrary subsequent convolutions.
        """
        with torch.autocast(device_type=feature.device.type, enabled=False):
            base = local_detail(channel_norm(feature))
            candidate = candidate.float()
            coefficient = (candidate * base).sum(1, keepdim=True)
            coefficient = coefficient / (base.square().sum(1, keepdim=True) + self.fusion_eps)
            return candidate - coefficient * base

    def bound_update(self, update, feature, budget):
        """FP32 channel-norm budget per pixel, after all learned scales.

        Quantization of the final feature addition is outside the real-valued
        bound. Spatially varying clipping need not preserve Haar-mode guarantees.
        """
        with torch.autocast(device_type=feature.device.type, enabled=False):
            update = update.float()
            base_norm = torch.linalg.vector_norm(feature.float(), dim=1, keepdim=True)
            update_norm = torch.linalg.vector_norm(update, dim=1, keepdim=True)
            scale = (budget * (base_norm + self.fusion_eps)
                     / (update_norm + self.fusion_eps)).clamp(max=1.)
            bounded = scale * update
            ratio = torch.linalg.vector_norm(bounded, dim=1) / (base_norm[:, 0] + self.fusion_eps)
            return bounded, ratio.detach().amax(), (scale.detach() < 1).float().mean()

    def refine_highres(self, image, high, support):
        if not self.use_rgb:
            return high
        if self.rgb_budget is None and self.residual_scale == 1.:
            # Preserve the exact arithmetic order of historical R9 checkpoints.
            return TaskGatedAllDepthRelay.refine_highres(self, image, high, support)
        detail = local_detail(self.detail_dw(self.stem(image)))
        if detail.shape[-2:] != high.shape[-2:]:
            detail = F.interpolate(detail, high.shape[-2:], mode='bilinear', align_corners=False)
        semantic = self.semantic_proj(high)
        if self.plain_cnn:
            magnitude = detail.float().square().mean(1, keepdim=True).clamp_min(1e-12).sqrt()
            average = F.avg_pool2d(magnitude, 3, 1, 1, count_include_pad=False)
            filtered, signals = detail, (magnitude.to(detail.dtype), average.to(detail.dtype))
        else:
            confidence = F.interpolate(support.float(), detail.shape[-2:], mode='nearest').to(detail.dtype)
            outlier = impulse_indicator(detail).to(detail.dtype)
            filtered = detail * (1. - .75 * (1. - confidence) * outlier) if self.use_attenuation else detail
            signals = (confidence, outlier)
        gate = self.detail_gate(torch.cat((filtered, semantic, *signals), 1)).sigmoid()
        update = self.detail_out(filtered) * gate
        update = (update.float() - update.float().mean((-2, -1), keepdim=True)) * self.residual_scale
        if self.rgb_budget is not None:
            update, ratio, clipped = self.bound_update(update, high, self.rgb_budget)
            self.last_diagnostics.update(rgb_update_ratio_max=ratio, rgb_budget_active_fraction=clipped)
        return (high.float() + update).to(high.dtype)

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
        pyramid = neck(tuple(refined) if self.inject_pyramid else base_features)
        if len(pyramid) != 4:
            raise RuntimeError('Expected four pyramid outputs')
        if hasattr(self, "transport"):
            high, rgb = TaskGatedAllDepthRelay.refine_highres(
                self, image, pyramid[0], supports[0], return_update=True)
            extra = self.transport(actual_updates, rgb)
            self.last_diagnostics.update(self.transport.last_diagnostics)
            return (high + extra, *pyramid[1:])
        return (self.refine_highres(image,pyramid[0],supports[0]),*pyramid[1:])
