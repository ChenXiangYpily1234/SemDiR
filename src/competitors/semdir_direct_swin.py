"""Native Swin-T adaptation of the beta-free Direct candidate.

Raw hierarchical stages stay unchanged. Only semantic-condition construction
aligns their grids; the shared SAG update operates on native 96-channel stage1.
"""
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from mmseg.registry import MODELS

from research.ad_pcpf_v2 import config, experiments
from . import semdir_direct as direct
from .models import CompetitorHierarchical
from .registry import EXPERIMENTS, CompetitorExperiment


class DirectSwinRelay(direct.DirectSemanticRelay):
    def __init__(self, hidden=32):
        self.stage_channels = (96, 192, 384, 768)
        super().__init__(96, hidden, semantic_channels=sum(self.stage_channels))

    def forward(self, image, features):
        if len(features) != 4 or any(f.shape[1] != c for f, c in zip(features, self.stage_channels)):
            raise ValueError('Four native Swin-T stage outputs required')
        grid = features[2].shape[-2:]
        normalized = [F.interpolate(direct.channel_norm(f), grid, mode='bilinear', align_corners=False)
                      for f in features]
        c = self.semantic_proj(torch.cat(normalized, 1))
        with torch.autocast(device_type=c.device.type, enabled=False):
            c = self.semantic_norm(c.float().movedim(1, -1)).movedim(-1, 1)
        return self.refine(image, features, c)


@MODELS.register_module()
class SemDiRDirectSwinBackbone(CompetitorHierarchical):
    def __init__(self, pretrained_checkpoint, pretrained_sha256, direct_hidden=32):
        super().__init__('swin_t', pretrained_checkpoint, pretrained_sha256, local_relay=False)
        with torch.random.fork_rng(devices=[]):
            self.relay = DirectSwinRelay(direct_hidden)
        self.method = 'semdir_direct_swin'

    def forward(self, image):
        return self.relay(image, self.encoder(image))

    @torch.no_grad()
    def check_zero_initialization(self, image):
        training = self.training
        self.eval()
        try:
            features = self.encoder(image)
            actual = self.relay(image, features)
            for x, y in zip(actual, features):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
        finally:
            self.train(training)


@MODELS.register_module()
class DirectSwinRegionPreservingSegmentor(direct.DirectRegionPreservingSegmentor):
    backbone_type = SemDiRDirectSwinBackbone


ID = 'SemDiR-Direct-Swin-T-S1'
if ID in EXPERIMENTS:
    raise RuntimeError('Direct Swin identity already registered')
EXPERIMENTS[ID] = CompetitorExperiment(ID, 'semdir_direct_swin', backbone='swin_t',
                                     comparison='', seed_required=1)
experiments.EXPERIMENTS[ID] = EXPERIMENTS[ID]
_parent_build = config._build


def build(*args, **kwargs):
    cfg = _parent_build(*args, **kwargs)
    if cfg['paper']['experiment'] != ID:
        return cfg
    if cfg['train_cfg']['max_iters'] != 160000:
        raise ValueError('Direct Swin uses the unchanged 160K schedule')
    old = cfg['model']['backbone']
    cfg['model']['backbone'] = dict(type='SemDiRDirectSwinBackbone', direct_hidden=32,
        pretrained_checkpoint=old['pretrained_checkpoint'], pretrained_sha256=old['pretrained_sha256'])
    cfg['model']['type'] = 'DirectSwinRegionPreservingSegmentor'
    cfg['model']['region_preservation'] = deepcopy(direct.registration.RSP)
    cfg['custom_imports']['imports'].append('competitors.semdir_direct_swin')
    cfg['custom_hooks'] = [dict(type='DirectAuditHook')]
    cfg['paper']['direct_semantics'] = dict(
        version='Native Swin-T beta-free Direct adaptation; independent new candidate',
        stage_channels=[96, 192, 384, 768], semantic_grid='stage3 native stride16',
        semantic_normalization='FP32 nonaffine channel LN before bilinear stage alignment',
        semantic_projection=[1440, 32], output_projection=[96, 96],
        native_pyramid=True, beta=False, direct_injection=False,
        cross_layer_difference=False, complementary_routing=False,
        SAG=True, original_RSP=True, boundary_RSP=False,
        identity_initialization=True, training_schedule_end=160000)
    cfg['paper']['provenance_label'] = 'Independent native Swin-T Direct adaptation; no historical G1 scores'
    return cfg


config._build = build
