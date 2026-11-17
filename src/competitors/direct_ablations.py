"""Independent A1/A3: first20K of the exact Direct160K schedule.

The running Full source snapshot is never modified or reused for training.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.dist import get_rank
from mmengine.registry import HOOKS
from mmseg.registry import MODELS

from research.ad_pcpf_v2 import config, experiments
from research.ad_pcpf_v2.artifacts import sha256, write_once
from research.ad_pcpf_v2.engine import PaperAuditHook, unwrap
from . import semdir_direct as direct
from .registry import EXPERIMENTS, CompetitorExperiment

FULL_STUDY = Path(__file__).resolve().parents[2] / 'trained_models/SemDiR-Direct-DeiT-S-160K-w4-b4-g16-s1-20261114'
FULL_CONFIG = FULL_STUDY / 'configs' / (direct.ID + '.py')
FULL_INITIALIZATION = FULL_STUDY / 'runs' / direct.ID / 'direct_initialization.json'
A1_ID = 'SemDiR-Direct-A1-NoSAG-20Kof160K-S1'
A3_ID = 'SemDiR-Direct-A3-NoRSP-20Kof160K-S1'
VARIANTS = {A1_ID: 'A1_NoSAG', A3_ID: 'A3_NoRSP'}


class DirectNoSAGRelay(direct.DirectSemanticRelay):
    def __init__(self, channels=384, hidden=32):
        # Consume exactly the Full constructor RNG, then remove only Wa.
        super().__init__(channels, hidden)
        del self.semantic_align

    def reset_relay(self):
        if hasattr(self, 'semantic_align'):
            return super().reset_relay()
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        with torch.no_grad():
            self.output.weight[:, :self.hidden].zero_()
            self.gamma.zero_()

    def refine(self, image, pyramid, c):
        # Same operations/order as Full, excluding only alignment and its dot score.
        q = self.rgb(F.interpolate(image, pyramid[0].shape[-2:], mode='bilinear', align_corners=False))
        c = c.to(q.dtype)
        gsem = F.interpolate(c, q.shape[-2:], mode='bilinear', align_corners=False)
        s0 = self.score(torch.cat([gsem, q], 1))
        with torch.autocast(device_type=q.device.type, enabled=False):
            gate = s0.float().tanh()
            local, cell = self.local_detail(q, c)
            e = local * cell.tanh()
            fused = torch.cat([q.float(), gate * gsem.float(), self.gamma.float() * gate * e], 1)
        delta = self.output(fused.to(q.dtype))
        self.last_diagnostics = dict(gate_mean=gate.detach().mean(),
            gate_abs_mean=gate.detach().abs().mean(), sag_score_rms=gate.new_zeros(()),
            update_rms=delta.detach().float().square().mean().sqrt(), fine_gamma=self.gamma.detach().float())
        return (pyramid[0] + delta, *pyramid[1:])


@MODELS.register_module()
class DirectNoSAGBackbone(direct.SemDiRDirectBackbone):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.relay.__class__ = DirectNoSAGRelay
        del self.relay.semantic_align
        self.method = 'semdir_direct_no_sag'


@MODELS.register_module()
class DirectNoSAGSegmentor(direct.DirectRegionPreservingSegmentor):
    backbone_type = DirectNoSAGBackbone


@MODELS.register_module()
class DirectNoRSPSegmentor(direct.DirectRegionPreservingSegmentor):
    def __init__(self, region_preservation, **kwargs):
        expected = dict(direct.registration.RSP, enabled=False, max_weight=0.)
        if dict(region_preservation) != expected:
            raise ValueError('A3 changes only RSP enabled/max_weight; other region settings are fixed')
        super().__init__(region_preservation=deepcopy(direct.registration.RSP), **kwargs)
        self.region_preservation = deepcopy(expected)
        # The exact inherited loss takes its CE/aux-only branch when disabled.


def canonical_full_digest(model):
    """Audit retained initial tensors; only A1's removed zero Wa is reconstructed.

This is a comparison hash, not the ablation's actual state hash/checkpoint.
"""
    state = dict(model.state_dict())
    relay = model.backbone.relay
    if isinstance(relay, DirectNoSAGRelay):
        key = 'backbone.relay.semantic_align.weight'
        if key in state:
            raise RuntimeError('A1 must not retain alignment parameters')
        d = relay.hidden
        state[key] = relay.semantic_proj.weight.new_zeros(d, d, 1, 1)
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@HOOKS.register_module()
class DirectAblationAuditHook(direct.DirectAuditHook):
    def before_train(self, runner):
        model = unwrap(runner.model)
        if not runner.cfg.resume:
            reference = runner.cfg.paper.direct_ablation
            if sha256(FULL_INITIALIZATION) != reference.full_initialization_file_sha256:
                raise RuntimeError('Full initialization proof changed')
            common = canonical_full_digest(model)
            if common != reference.full_initialization_sha256:
                raise RuntimeError('Ablation retained initialization differs from the actual running Full')
        super().before_train(runner)
        if not runner.cfg.resume and get_rank() == 0:
            write_once(Path(runner.work_dir) / 'ablation_initialization.json', dict(
                variant=runner.cfg.paper.direct_ablation.variant,
                actual_state_sha256=direct.state_digest(model),
                retained_parameter_comparison_sha256=common,
                comparison_scope='All Full tensors; A1 adds only a zero Wa for comparison, not a trainable parameter',
                full_config_sha256=runner.cfg.paper.direct_ablation.full_config_sha256))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter + 1 == 200:
            relay = unwrap(runner.model).backbone.relay
            values = dict(H=relay.output.weight[:, :relay.hidden], score=relay.score.weight, gamma=relay.gamma)
            if hasattr(relay, 'semantic_align'):
                values['alignment'] = relay.semantic_align.weight
            learned = {k: float(v.detach().float().abs().sum()) for k,v in values.items()}
            if any(not bool(torch.isfinite(v).all()) for v in values.values()) or any(v <= 0 for v in learned.values()):
                raise RuntimeError('An active zero-initialized ablation path did not learn')
        # Skip the Full hook's mandatory Wa assertion for physically pruned A1.
        PaperAuditHook.after_train_iter(self, runner, batch_idx, data_batch, outputs)
        if runner.iter + 1 == 200 and get_rank() == 0:
            rsp = 'loss_rsp' in (outputs or {})
            expected_rsp = runner.cfg.paper.direct_ablation.variant != 'A3_NoRSP'
            if rsp != expected_rsp:
                raise RuntimeError('Ablation RSP execution does not match its definition')
            root = Path(runner.work_dir).parents[1] if runner.cfg.paper.smoke else Path(runner.work_dir)
            write_once(root / 'ablation_learning_probe.json', dict(iteration=200,
                variant=runner.cfg.paper.direct_ablation.variant, active_projection_l1=learned,
                RSP_loss_present=rsp, all_active_paths_learned=True,
                measurement='shared-GPU engineering smoke' if runner.cfg.paper.smoke else 'formal training'))


class DirectScreenExperiment(CompetitorExperiment):
    @property
    def steps(self):
        return 20000


for name in VARIANTS:
    if name in EXPERIMENTS:
        raise RuntimeError('Direct ablation identity already registered')
    EXPERIMENTS[name] = DirectScreenExperiment(name, 'm23_plain', backbone='deit_s', comparison='', seed_required=1)
    experiments.EXPERIMENTS[name] = EXPERIMENTS[name]

_parent_build = config._build


def build(*args, **kwargs):
    cfg = _parent_build(*args, **kwargs)
    name = cfg['paper']['experiment']
    if name not in VARIANTS:
        return cfg
    if cfg['train_cfg']['max_iters'] != 20000:
        raise ValueError('A1/A3 are independent first20K screenings')
    reference = Config.fromfile(str(FULL_CONFIG), import_custom_modules=False).to_dict()
    cfg['param_scheduler'] = deepcopy(reference['param_scheduler'])
    for key in ('optim_wrapper', 'train_dataloader', 'val_dataloader', 'test_dataloader', 'val_evaluator', 'test_evaluator'):
        if cfg[key] != reference[key]:
            raise ValueError('Ablation recipe differs from the actual Full: ' + key)
    for key in ('decode_head', 'auxiliary_head', 'data_preprocessor', 'test_cfg'):
        if cfg['model'][key] != reference['model'][key]:
            raise ValueError('Ablation changes a shared model component: ' + key)
    backbone = cfg['model']['backbone']
    for key in ('fusion', 'conditioning', 'relay_options'):
        backbone.pop(key, None)
    backbone.update(type='DirectNoSAGBackbone' if name==A1_ID else 'SemDiRDirectBackbone', direct_hidden=32)
    cfg['model']['type'] = 'DirectNoSAGSegmentor' if name==A1_ID else 'DirectNoRSPSegmentor'
    cfg['model']['region_preservation'] = deepcopy(direct.registration.RSP)
    if name == A3_ID:
        cfg['model']['region_preservation'].update(enabled=False, max_weight=0.)
    cfg['custom_imports']['imports'].append('competitors.direct_ablations')
    cfg['custom_hooks'] = [dict(type='DirectAblationAuditHook')]
    initial = json.loads(FULL_INITIALIZATION.read_text())
    cfg['paper']['direct_semantics'] = deepcopy(reference['paper']['direct_semantics'])
    cfg['paper']['direct_semantics'].update(SAG=name!=A1_ID, original_RSP=name!=A3_ID)
    cfg['paper']['direct_ablation'] = dict(variant=VARIANTS[name], reference=direct.ID,
        full_config_sha256=sha256(FULL_CONFIG), full_initialization_sha256=initial['direct_state_sha256'],
        full_initialization_file_sha256=sha256(FULL_INITIALIZATION),
        independent_ImageNet_start=True, schedule_end=160000, training_stop_step=20000,
        only_change='remove Wa and sa; retain tanh(s0)' if name==A1_ID else 'disable RSP only; inference identical')
    cfg['paper']['provenance_label'] = 'New beta-free Direct single-factor ablation, not historical Exact/G1; first20K of unchanged160K schedule'
    return cfg


config._build = build
