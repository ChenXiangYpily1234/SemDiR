"""COCO-Stuff-10K official v1.1, independent80K, NEW beta-free Direct."""
from copy import deepcopy
from research.ad_pcpf_v2 import config,experiments
from . import semdir_direct as direct
from . import generalization_datasets as target
from .registry import EXPERIMENTS

ID='SemDiR-Direct-COCO10K-S1'
spec=target.TargetDatasetExperiment(ID,'semdir_direct',dataset='cocostuff10k',backbone='deit_s',comparison='COCOStuff10K-Vanilla',seed_required=1)
EXPERIMENTS[ID]=spec;experiments.EXPERIMENTS[ID]=spec
# The native target-dataset preparer fingerprints official9K/1K splits and
# exact image/label pairing. Its generic build supplies171-class COCO loaders.
target.EXPERIMENTS[ID]=spec
_parent_build=config._build

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs)
    if cfg['paper']['experiment']!=ID:return cfg
    if cfg['train_cfg']['max_iters']!=80000 or cfg['model']['decode_head']['num_classes']!=171:
        raise ValueError('COCO Direct requires independent80K and171 classes')
    backbone=cfg['model']['backbone']
    for key in ['fusion','conditioning','relay_options','method']:backbone.pop(key,None)
    backbone.update(type='SemDiRDirectBackbone',direct_hidden=32,image_size=512)
    cfg['model']['type']='DirectRegionPreservingSegmentor'
    cfg['model']['region_preservation']=deepcopy(direct.registration.RSP)
    cfg['custom_imports']['imports'].append('competitors.semdir_direct_coco')
    cfg['custom_hooks']=[dict(type='DirectAuditHook')]
    cfg['paper'].update(method='semdir_direct',budget=80000,architecture_reference='SemDiR-Direct',
        provenance_label='NEW beta-free Direct; official COCO-Stuff10K v1.1; independent ImageNet80K; original RSP',
        direct_semantics=dict(version='Original DeiT-S beta-free Direct on COCO-Stuff10K',D=384,d=32,K=4,
            semantic_projection=[1536,32],output_projection=[96,384],norm_eps=1e-6,
            beta=False,cross_layer_difference=False,complementary_routing=False,direct_injection=False,
            SAG=True,original_RSP=True,boundary_RSP=False,identity_initialization=True,
            training_schedule_end=80000,bins='FP32 disjoint floor-index mean; bilinear guidance versus cell gather'))
    return cfg
config._build=build
