"""P1/P2: beta-free SemDiR-Direct, native MiT-B0 and Context59.

The condition is constructed directly from four normalized raw outputs. No
cross-layer subtraction, beta, back-projection, legacy relay or Boundary-RSP.
"""
from copy import deepcopy
import torch
from mmseg.registry import MODELS
from research.ad_pcpf_v2 import config,experiments
from . import semdir_direct as direct
from .semdir_direct_swin import DirectSwinRelay
from .models import CompetitorHierarchical
from .registry import EXPERIMENTS,CompetitorExperiment


class DirectMiTRelay(DirectSwinRelay):
    def __init__(self,hidden=32):
        self.stage_channels=(32,64,160,256)
        direct.DirectSemanticRelay.__init__(self,32,hidden,semantic_channels=sum(self.stage_channels))
    # Inherited forward normalizes raw stage channels before grid alignment,
    # concatenates all four stages and reuses DirectSemanticRelay.refine.


@MODELS.register_module()
class SemDiRDirectMiTBackbone(CompetitorHierarchical):
    def __init__(self,pretrained_checkpoint,pretrained_sha256,direct_hidden=32):
        super().__init__('mit_b0',pretrained_checkpoint,pretrained_sha256,local_relay=False)
        with torch.random.fork_rng(devices=[]):self.relay=DirectMiTRelay(direct_hidden)
        self.method='semdir_direct_mit'
    def forward(self,image):return self.relay(image,self.encoder(image))
    @torch.no_grad()
    def check_zero_initialization(self,image):
        training=self.training;self.eval()
        try:
            features=self.encoder(image);actual=self.relay(image,features)
            for x,y in zip(actual,features):torch.testing.assert_close(x,y,atol=0,rtol=0)
        finally:self.train(training)


@MODELS.register_module()
class DirectMiTRegionPreservingSegmentor(direct.DirectRegionPreservingSegmentor):
    backbone_type=SemDiRDirectMiTBackbone


MIT_ID='SemDiR-Direct-MiT-B0-S1'
CONTEXT_ID='SemDiR-Direct-Context59-S1'
EXPERIMENTS[MIT_ID]=CompetitorExperiment(MIT_ID,'semdir_direct_mit',backbone='mit_b0',comparison='M29',seed_required=1)
EXPERIMENTS[CONTEXT_ID]=CompetitorExperiment(CONTEXT_ID,'m23_plain',dataset='context59',backbone='deit_s',comparison='Context59-Vanilla',seed_required=1)
for name in [MIT_ID,CONTEXT_ID]:experiments.EXPERIMENTS[name]=EXPERIMENTS[name]
_parent_build=config._build

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs);name=cfg['paper']['experiment']
    if name not in [MIT_ID,CONTEXT_ID]:return cfg
    if name==MIT_ID:
        if cfg['train_cfg']['max_iters']!=160000:raise ValueError('MiT Direct requires independent160K')
        old=cfg['model']['backbone']
        cfg['model']['backbone']=dict(type='SemDiRDirectMiTBackbone',direct_hidden=32,
            pretrained_checkpoint=old['pretrained_checkpoint'],pretrained_sha256=old['pretrained_sha256'])
        cfg['model']['type']='DirectMiTRegionPreservingSegmentor'
        details=dict(version='Native MiT-B0 beta-free Direct',stage_channels=[32,64,160,256],
            semantic_grid='stage3 native stride16',semantic_projection=[512,32],output_projection=[96,32],
            semantic_normalization='FP32 nonaffine channel LN before bilinear alignment',native_pyramid=True)
    else:
        if cfg['train_cfg']['max_iters']!=80000:raise ValueError('Context59 Direct requires independent80K')
        backbone=cfg['model']['backbone']
        for key in ['fusion','conditioning','relay_options']:backbone.pop(key,None)
        backbone.update(type='SemDiRDirectBackbone',direct_hidden=32)
        cfg['model']['type']='DirectRegionPreservingSegmentor'
        details=dict(version='Original DeiT-S beta-free Direct; independent Context59',D=384,d=32,K=4,
            semantic_projection=[1536,32],output_projection=[96,384],norm_eps=1e-6)
        cfg['paper']['method']='semdir_direct'
        cfg['paper']['label_mapping']=dict(dataset='PASCAL-Context-59',png_ids='0 background;1..59 classes',
            model_ids='0..58',reduce_zero_label=True,ignore_index=255)
    cfg['model']['region_preservation']=deepcopy(direct.registration.RSP)
    cfg['custom_imports']['imports'].append('competitors.semdir_direct_generalization')
    cfg['custom_hooks']=[dict(type='DirectAuditHook')]
    details.update(beta=False,cross_layer_difference=False,complementary_routing=False,direct_injection=False,
        SAG=True,original_RSP=True,boundary_RSP=False,identity_initialization=True,
        bins='FP32 disjoint floor-index mean; bilinear guidance versus cell gather',
        training_schedule_end=cfg['train_cfg']['max_iters'])
    cfg['paper']['direct_semantics']=details
    cfg['paper']['budget']=cfg['train_cfg']['max_iters']
    cfg['paper']['architecture_reference']='SemDiR-Direct'
    cfg['paper']['provenance_label']='New beta-free Direct; independent ImageNet initialization; not old Simple+SAG or Exact results'
    return cfg
config._build=build
