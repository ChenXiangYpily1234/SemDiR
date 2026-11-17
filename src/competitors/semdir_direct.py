"""Independent DeiT-S candidate: direct multilevel conditioning + SAG + RSP.

Historical Simple/SAG weights and results belong to a different architecture.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.dist import get_rank
from mmengine.registry import HOOKS
from mmseg.registry import MODELS

from research.ad_pcpf_v2 import config, experiments
from research.ad_pcpf_v2.adapter import (
    ADPCPFPaperBackbone, ADPCPFPaperSegmentor, ADPCPFRegionPreservingSegmentor)
from research.ad_pcpf_v2.artifacts import write_once
from research.ad_pcpf_v2.engine import PaperAuditHook, unwrap
from research.ad_pcpf_v2.transport import CoarseFineResidualTransport
from . import registration
from .registry import EXPERIMENTS, CompetitorExperiment


def channel_norm(x):
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.layer_norm(x.float().movedim(1, -1), (x.shape[1],),
                            eps=1e-6).movedim(-1, 1)


class DirectSemanticRelay(nn.Module):
    """Only the layers in the new definition; no dormant legacy subnetwork."""
    def __init__(self, channels=384, hidden=32, semantic_channels=None):
        super().__init__()
        self.channels, self.hidden = channels, hidden
        self.semantic_proj = nn.Conv2d(semantic_channels or 4 * channels, hidden, 1)
        self.semantic_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.rgb = nn.Conv2d(3, hidden, 1)
        self.score = nn.Conv2d(2 * hidden, 1, 1)
        self.semantic_align = nn.Conv2d(hidden, hidden, 1, bias=False)
        self.output = nn.Conv2d(3 * hidden, channels, 1, bias=False)
        # Each block retains the fan-in of an independent d -> D projection.
        with torch.no_grad():
            for block in self.output.weight.split(hidden, dim=1):
                nn.init.kaiming_uniform_(block, a=math.sqrt(5))
        self.gamma = nn.Parameter(torch.zeros(()))
        self.reset_relay()

    def reset_relay(self):
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        nn.init.zeros_(self.semantic_align.weight)
        with torch.no_grad():
            self.output.weight[:, :self.hidden].zero_()
            self.gamma.zero_()

    @staticmethod
    def local_detail(q, c):
        """Disjoint floor-index bins, including non-divisible spatial sizes."""
        b, d, height, width = q.shape
        h, w = c.shape[-2:]
        ids, _ = CoarseFineResidualTransport.geometry(height, width, h, w, q.device)
        with torch.autocast(device_type=q.device.type, enabled=False):
            flat = q.float().flatten(2)
            index = ids.view(1, 1, -1).expand(b, d, -1)
            sums = torch.zeros(b, d, h*w, device=q.device, dtype=torch.float32)
            sums.scatter_add_(2, index, flat)
            counts = torch.bincount(ids, minlength=h*w).float()
            means = (sums / counts.view(1, 1, -1)).gather(2, index)
            local = (flat - means).reshape_as(q)
            cell = c.float().flatten(2).gather(2, index).reshape_as(q)
        return local, cell

    def forward(self, image, features, neck):
        if len(features) != 4 or any(f.shape != features[0].shape for f in features):
            raise ValueError('Four aligned DeiT semantic taps are required')
        if features[0].shape[1] != self.channels:
            raise ValueError('Semantic channel mismatch')
        # Original taps, without any residual injection, form the pyramid.
        pyramid = neck(tuple(features))
        c = self.semantic_proj(torch.cat([channel_norm(f) for f in features], 1))
        with torch.autocast(device_type=c.device.type, enabled=False):
            c = self.semantic_norm(c.float().movedim(1, -1)).movedim(-1, 1)
        return self.refine(image, pyramid, c)

    def refine(self, image, pyramid, c):
        """Shared SAG/local-detail update for aligned or hierarchical inputs."""
        q = self.rgb(F.interpolate(image, pyramid[0].shape[-2:], mode='bilinear', align_corners=False))
        c = c.to(q.dtype)
        gsem = F.interpolate(c, q.shape[-2:], mode='bilinear', align_corners=False)
        aligned = F.interpolate(self.semantic_align(c), q.shape[-2:], mode='bilinear', align_corners=False)
        s0 = self.score(torch.cat([gsem, q], 1))
        with torch.autocast(device_type=q.device.type, enabled=False):
            sa = (aligned.float() * q.float()).sum(1, keepdim=True) / math.sqrt(self.hidden)
            gate = (s0.float() + sa).tanh()
            local, cell = self.local_detail(q, c)
            e = local * cell.tanh()
            fused = torch.cat([q.float(), gate * gsem.float(), self.gamma.float() * gate * e], 1)
        delta = self.output(fused.to(q.dtype))
        self.last_diagnostics = dict(
            gate_mean=gate.detach().mean(), gate_abs_mean=gate.detach().abs().mean(),
            sag_score_rms=sa.detach().square().mean().sqrt(),
            update_rms=delta.detach().float().square().mean().sqrt(),
            fine_gamma=self.gamma.detach().float())
        return (pyramid[0] + delta, *pyramid[1:])


@MODELS.register_module()
class SemDiRDirectBackbone(ADPCPFPaperBackbone):
    def __init__(self, direct_hidden=32, **kwargs):
        super().__init__(method='vanilla', **kwargs)
        with torch.random.fork_rng(devices=[]):
            self.relay = DirectSemanticRelay(self.encoder.embed_dim, direct_hidden)
        self.method = 'semdir_direct'

    def _direct_features(self, image):
        return self.encoder.forward_intermediates(image, indices=self.indices, norm=False,
            output_fmt='NCHW', intermediates_only=True)

    def forward(self, image):
        return self.relay(image, self._direct_features(image), self.feature2pyramid)

    @torch.no_grad()
    def check_zero_initialization(self, image):
        training = self.training
        self.eval()
        try:
            features = self._direct_features(image)
            actual = self.relay(image, features, self.feature2pyramid)
            expected = self.feature2pyramid(tuple(features))
            for x, y in zip(actual, expected):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        finally:
            self.train(training)


@MODELS.register_module()
class DirectRegionPreservingSegmentor(ADPCPFRegionPreservingSegmentor):
    backbone_type = SemDiRDirectBackbone

    def __init__(self, region_preservation, **kwargs):
        ADPCPFPaperSegmentor.__init__(self, **kwargs)
        if type(self.backbone) is not self.backbone_type:
            raise ValueError('Direct RSP requires the new beta-free DeiT backbone')
        self.region_preservation = deepcopy(region_preservation)
        if self.region_preservation != registration.RSP:
            raise ValueError('Original frozen RSP definition must remain unchanged')
        self._rsp_iteration = 0
        object.__setattr__(self, '_rsp_mask_pool', None)


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@HOOKS.register_module()
class DirectAuditHook(PaperAuditHook):
    def before_train(self, runner):
        model = unwrap(runner.model)
        if not runner.cfg.resume:
            digest = state_digest(model)
            proof_path = Path(runner.work_dir).parents[1] / 'direct_smoke_initialization.json'
            if not runner.cfg.paper.smoke:
                proof = json.loads(proof_path.read_text())
                if proof.get('direct_state_sha256') != digest:
                    raise RuntimeError('Direct smoke/formal initialization mismatch')
        super().before_train(runner)
        if not runner.cfg.resume and get_rank() == 0:
            # Native initialization is immutable; keep the full-state proof separate.
            write_once(Path(runner.work_dir) / 'direct_initialization.json',
                       dict(direct_state_sha256=digest, zero_update_check='PASS',
                            module_parameters=sum(p.numel() for p in model.backbone.relay.parameters())))
            if runner.cfg.paper.smoke:
                write_once(proof_path, dict(experiment=runner.cfg.paper.experiment,
                    smoke_config_sha256=runner.cfg.paper_config_sha256,
                    direct_state_sha256=digest, zero_update_check='PASS'))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter + 1 == 200:
            relay = unwrap(runner.model).backbone.relay
            for value in (relay.output.weight[:, :relay.hidden], relay.semantic_align.weight,
                          relay.score.weight, relay.gamma):
                if not bool(torch.isfinite(value).all()) or not bool(value.detach().abs().sum() > 0):
                    raise RuntimeError('A zero-initialized Direct path failed to learn')
        super().after_train_iter(runner, batch_idx, data_batch, outputs)


ID = 'SemDiR-Direct-DeiT-S-S1'
if ID in EXPERIMENTS:
    raise RuntimeError('Direct experiment identity already registered')
EXPERIMENTS[ID] = CompetitorExperiment(ID, 'm23_plain', backbone='deit_s',
                                     comparison='', seed_required=1)
experiments.EXPERIMENTS[ID] = EXPERIMENTS[ID]
_parent_build = config._build


def build(*args, **kwargs):
    cfg = _parent_build(*args, **kwargs)
    if cfg['paper']['experiment'] != ID:
        return cfg
    if cfg['train_cfg']['max_iters'] != 160000:
        raise ValueError('Direct uses the unchanged 160K schedule')
    backbone = cfg['model']['backbone']
    for key in ('fusion', 'conditioning', 'relay_options'):
        backbone.pop(key, None)
    backbone.update(type='SemDiRDirectBackbone', direct_hidden=32)
    cfg['model']['type'] = 'DirectRegionPreservingSegmentor'
    cfg['custom_imports']['imports'].append('competitors.semdir_direct')
    cfg['custom_hooks'] = [dict(type='DirectAuditHook')]
    cfg['paper']['direct_semantics'] = dict(
        version='DeiT-only direct normalized four-tap conditioning; independent new candidate',
        D=384, d=32, K=4, norm_eps=1e-6, semantic_projection=[1536, 32],
        output_projection=[96, 384], beta=False, direct_injection=False,
        cross_layer_difference=False, complementary_routing=False,
        SAG=True, original_RSP=True, boundary_RSP=False,
        bins='disjoint floor-index groups; FP32 mean; bilinear guidance versus cell gather',
        identity_initialization=True, training_schedule_end=160000)
    cfg['paper']['provenance_label'] = 'Unmeasured independent beta-free Direct candidate; not historical G1 scores'
    return cfg


config._build = build
