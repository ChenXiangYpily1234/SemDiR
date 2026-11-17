"""Independent ADE20K160K main A2 on unchanged DeiT-S/Swin-T/MiT-B0."""
from copy import deepcopy
from pathlib import Path
import json

from mmengine.config import Config
from mmengine.dist import get_rank
from mmengine.registry import HOOKS
from mmseg.registry import MODELS
from research.ad_pcpf_v2 import config,experiments
from research.ad_pcpf_v2.artifacts import sha256,write_once
from research.ad_pcpf_v2.engine import unwrap
from . import direct_second as a2
from . import semdir_direct as direct
from . import semdir_direct_swin as swin
from . import semdir_direct_generalization as generalization
from .registry import EXPERIMENTS,CompetitorExperiment,ROOT

# Resolve the existing asset link for both canonical and frozen source roots.
ASSET_ROOT=(ROOT.parent/'models').resolve().parent
IDS={
    'deit_s':'SemDiR-A2-DeiT-S-160K-S1',
    'swin_t':'SemDiR-A2-Swin-T-160K-S1',
    'mit_b0':'SemDiR-A2-MiT-B0-160K-S1',
}
REFERENCES={
    'deit_s':('SemDiR-Direct-DeiT-S-160K-w4-b4-g16-s1-20261114',direct.ID),
    'swin_t':('SemDiR-Direct-Swin-T-160K-w4-b4-g16-s1-20261114',swin.ID),
    'mit_b0':('SemDir-Direct-MiT-B0-160K-w4-b4-g16-s1-20261114',generalization.MIT_ID),
}


def reference_config(arch):
    folder,name=REFERENCES[arch]
    return ASSET_ROOT/'trained_models'/folder/'configs'/f'{name}.py'


def reference_initialization(arch):
    folder,name=REFERENCES[arch]
    return ASSET_ROOT/'trained_models'/folder/'runs'/name/'direct_initialization.json'


class A2SwinRelay(a2.DirectNoFineRelay,swin.DirectSwinRelay):
    """Keep native stage conditioning; reuse the exact existing A2 refinement."""
    def __init__(self,hidden=32):
        swin.DirectSwinRelay.__init__(self,hidden)
        self.prune()

    def forward(self,image,features):
        return swin.DirectSwinRelay.forward(self,image,features)


class A2MiTRelay(a2.DirectNoFineRelay,generalization.DirectMiTRelay):
    def __init__(self,hidden=32):
        generalization.DirectMiTRelay.__init__(self,hidden)
        self.prune()

    def forward(self,image,features):
        return swin.DirectSwinRelay.forward(self,image,features)


@MODELS.register_module()
class A2SwinBackbone(swin.SemDiRDirectSwinBackbone):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.relay.__class__=A2SwinRelay
        self.relay.prune()
        self.method='semdir_a2_swin'


@MODELS.register_module()
class A2MiTBackbone(generalization.SemDiRDirectMiTBackbone):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.relay.__class__=A2MiTRelay
        self.relay.prune()
        self.method='semdir_a2_mit'


@MODELS.register_module()
class A2SwinSegmentor(direct.DirectRegionPreservingSegmentor):
    backbone_type=A2SwinBackbone


@MODELS.register_module()
class A2MiTSegmentor(direct.DirectRegionPreservingSegmentor):
    backbone_type=A2MiTBackbone


@HOOKS.register_module()
class A2MainAuditHook(direct.DirectAuditHook):
    def before_train(self,runner):
        model=unwrap(runner.model)
        if not runner.cfg.resume:
            proof=runner.cfg.paper.a2_main
            path=reference_initialization(proof.backbone)
            if sha256(path)!=proof.reference_initialization_file_sha256:
                raise RuntimeError('Corresponding Full initialization reference changed')
            comparison=a2.canonical_full_digest(model)
            if comparison!=proof.full_initialization_sha256:
                raise RuntimeError('A2 retained tensors differ from corresponding native Full initialization')
        super().before_train(runner)
        if not runner.cfg.resume and get_rank()==0:
            write_once(Path(runner.work_dir)/'a2_main_initialization.json',dict(
                architecture=proof.backbone,independent_ImageNet_start=True,
                full_comparison_sha256=comparison,actual_state_sha256=direct.state_digest(model),
                formula='P0+Wout[Q,gG]',fine_value=False,beta=False,
                module_parameters=sum(p.numel() for p in model.backbone.relay.parameters())))

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        # The A2 hook checks H/score/alignment learning and original RSP,
        # without requiring physically removed gamma/fine-value parameters.
        a2.DirectSecondAuditHook.after_train_iter(self,runner,batch_idx,data_batch,outputs)


for arch,name in IDS.items():
    if name in EXPERIMENTS:
        raise RuntimeError('A2 main identity already registered: '+name)
    spec=CompetitorExperiment(name,'m23_plain' if arch=='deit_s' else 'semdir_a2_'+arch,
        backbone=arch,comparison='',seed_required=1)
    EXPERIMENTS[name]=spec
    experiments.EXPERIMENTS[name]=spec

_parent_build=config._build


def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs)
    name=cfg['paper']['experiment']
    if name not in IDS.values():
        return cfg
    arch=next(key for key,value in IDS.items() if value==name)
    ref=Config.fromfile(str(reference_config(arch)),import_custom_modules=False).to_dict()
    for key in ['train_dataloader','val_dataloader','test_dataloader','optim_wrapper',
                'param_scheduler','train_cfg','val_evaluator','test_evaluator']:
        if cfg[key]!=ref[key]:
            raise ValueError('A2 main changes the corresponding matched recipe: '+key)
    for key in ['decode_head','auxiliary_head','data_preprocessor','test_cfg']:
        if cfg['model'][key]!=ref['model'][key]:
            raise ValueError('A2 main changes the corresponding decoder: '+key)
    if cfg['train_cfg']['max_iters']!=160000 or cfg['param_scheduler'][-1]['end']!=160000:
        raise ValueError('A2 main requires a complete160K schedule')
    if cfg['load_from'] is not None or cfg['resume']:
        raise ValueError('A2 main requires an independent ImageNet start')
    old=cfg['model']['backbone']
    if arch=='deit_s':
        for key in ['method','fusion','conditioning','relay_options']:
            old.pop(key,None)
        old.update(type='DirectNoFineBackbone',direct_hidden=32,image_size=512)
        model_type='DirectNoFineSegmentor'
    else:
        cfg['model']['backbone']=dict(type='A2SwinBackbone' if arch=='swin_t' else 'A2MiTBackbone',
            pretrained_checkpoint=old['pretrained_checkpoint'],pretrained_sha256=old['pretrained_sha256'],direct_hidden=32)
        model_type='A2SwinSegmentor' if arch=='swin_t' else 'A2MiTSegmentor'
    cfg['model'].update(type=model_type,region_preservation=deepcopy(direct.registration.RSP))
    cfg['custom_imports']['imports'].append('competitors.semdir_a2_main')
    cfg['custom_hooks']=[dict(type='A2MainAuditHook')]
    initial=json.loads(reference_initialization(arch).read_text())
    in_channels={'deit_s':1536,'swin_t':1440,'mit_b0':512}[arch]
    out_channels={'deit_s':384,'swin_t':96,'mit_b0':32}[arch]
    cfg['paper'].update(method='semdir_a2',budget=160000,architecture_reference='SemDiR-A2',
        provenance_label='Current A2, no beta/no Vf; native unchanged backbone; independent ImageNet160K',
        direct_semantics=dict(semantic_projection=[in_channels,32],output_projection=[64,out_channels],
            semantic_grid='four native stage outputs aligned to stage3' if arch!='deit_s' else 'four patch-grid taps3/6/9/12',
            beta=False,cross_layer_difference=False,complementary_routing=False,direct_injection=False,
            SAG=True,original_RSP=True,boundary_RSP=False,fine_value=False,gamma=False,local_deviation=False,
            ungated_RGB_update=True,formula='P0+Wout[Q,gG]',identity_initialization=True,training_schedule_end=160000),
        direct_ablation=dict(variant='A2_main_'+arch),
        a2_main=dict(backbone=arch,independent_ImageNet_start=True,training_stop_step=160000,schedule_end=160000,
            reference_config_sha256=sha256(reference_config(arch)),
            reference_initialization_file_sha256=sha256(reference_initialization(arch)),
            full_initialization_sha256=initial['direct_state_sha256']))
    return cfg


config._build=build
