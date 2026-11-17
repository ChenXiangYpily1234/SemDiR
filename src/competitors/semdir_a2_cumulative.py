"""C2 = current A2 without RSP; first20K of the unchanged160K schedule."""
from pathlib import Path
from copy import deepcopy
import json,os
import torch
from mmengine.config import Config
from mmengine.registry import HOOKS
from mmengine.dist import get_rank
from mmseg.registry import MODELS
from research.ad_pcpf_v2 import config,experiments
from research.ad_pcpf_v2.artifacts import sha256,write_once
from research.ad_pcpf_v2.engine import PaperAuditHook,unwrap
from . import direct_second as a2
from . import semdir_direct as direct
from .registry import EXPERIMENTS

BASE=Path(__file__).resolve().parents[2]
A2_STUDY=BASE/'trained_models/SemDiR-Direct-A2-20Kof160K-w4-b4-g16-s1-20261115'
A2_ID=a2.A2_ID
A2_CONFIG=A2_STUDY/'configs'/f'{A2_ID}.py'
A2_INIT=A2_STUDY/'runs'/A2_ID/'direct_initialization.json'
ID='SemDiR-A2-C2-NoRSP-20Kof160K-S1'
SHARED_KEYS=['optim_wrapper','param_scheduler','train_dataloader','val_dataloader','test_dataloader','train_cfg','val_evaluator','test_evaluator','randomness']

@MODELS.register_module()
class A2WithoutRSPSegmentor(a2.DirectNoFineSegmentor):
    def __init__(self,region_preservation,**kwargs):
        expected=dict(direct.registration.RSP,enabled=False,max_weight=0.)
        if dict(region_preservation)!=expected:raise ValueError('C2 changes only RSP enabled/max_weight')
        super().__init__(region_preservation=deepcopy(direct.registration.RSP),**kwargs)
        self.region_preservation=deepcopy(expected)

@HOOKS.register_module()
class A2CumulativeAuditHook(direct.DirectAuditHook):
    def before_train(self,runner):
        model=unwrap(runner.model)
        if not runner.cfg.resume:
            proof=runner.cfg.paper.a2_cumulative
            if sha256(A2_INIT)!=proof.reference_initialization_file_sha256:raise RuntimeError('A2 initialization reference changed')
            actual=direct.state_digest(model)
            if actual!=proof.reference_initialization_sha256:raise RuntimeError('C2 retained initialization differs from actual A2')
        super().before_train(runner)
        if not runner.cfg.resume and get_rank()==0:
            write_once(Path(runner.work_dir)/'a2_cumulative_initialization.json',dict(
                actual_state_sha256=actual,reference_A2_state_sha256=proof.reference_initialization_sha256,
                only_change='Disable original RSP; same A2 forward, parameter tensors and initialization',independent_ImageNet_start=True))

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        if runner.iter+1==200:
            r=unwrap(runner.model).backbone.relay
            values=dict(H=r.output.weight[:,:r.hidden],score=r.score.weight,alignment=r.semantic_align.weight)
            learned={k:float(v.detach().float().abs().sum()) for k,v in values.items()}
            if any(not bool(torch.isfinite(v).all()) for v in values.values()) or any(v<=0 for v in learned.values()):raise RuntimeError('An A2 path did not learn')
            if 'loss_rsp' in (outputs or {}):raise RuntimeError('C2 must never execute RSP')
        PaperAuditHook.after_train_iter(self,runner,batch_idx,data_batch,outputs)
        if runner.iter+1==200 and get_rank()==0:
            root=Path(runner.work_dir).parents[1] if runner.cfg.paper.smoke else Path(runner.work_dir)
            write_once(root/'ablation_learning_probe.json',dict(iteration=200,variant='C2_A2_without_RSP',
                active_projection_l1=learned,all_active_paths_learned=True,RSP_loss_present=False,
                measurement='engineering smoke' if runner.cfg.paper.smoke else 'formal training'))

if ID in EXPERIMENTS:raise RuntimeError('C2 already registered')
EXPERIMENTS[ID]=a2.first.DirectScreenExperiment(ID,'m23_plain',backbone='deit_s',comparison='',seed_required=1)
experiments.EXPERIMENTS[ID]=EXPERIMENTS[ID]
_parent_build=config._build

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs)
    if cfg['paper']['experiment']!=ID:return cfg
    ref=Config.fromfile(str(A2_CONFIG),import_custom_modules=False).to_dict()
    cfg['param_scheduler']=deepcopy(ref['param_scheduler'])
    for key in SHARED_KEYS:
        if cfg[key]!=ref[key]:raise ValueError('C2 changes A2 recipe: '+key)
    for key in ['decode_head','auxiliary_head','data_preprocessor','test_cfg']:
        if cfg['model'][key]!=ref['model'][key]:raise ValueError('C2 changes shared head: '+key)
    if cfg['load_from'] is not None or os.environ.get('DCPA_COMP_INIT'):raise ValueError('Independent ImageNet start required')
    cfg['model']['backbone']=deepcopy(ref['model']['backbone'])
    cfg['model'].update(type='A2WithoutRSPSegmentor',region_preservation=dict(direct.registration.RSP,enabled=False,max_weight=0.))
    cfg['custom_imports']['imports'].append('competitors.semdir_a2_cumulative');cfg['custom_hooks']=[dict(type='A2CumulativeAuditHook')]
    init=json.loads(A2_INIT.read_text())
    cfg['paper']['direct_semantics']=deepcopy(ref['paper']['direct_semantics'])
    cfg['paper']['direct_semantics'].update(original_RSP=False,formula='P0+Wout[Q,gG]')
    cfg['paper'].update(method='semdir_a2_no_rsp',budget=20000,architecture_reference='SemDiR-A2',
        provenance_label='C2 cumulative control: current A2 forward with RSP disabled; independent first20K of160K schedule',
        a2_cumulative=dict(variant='C2',reference=A2_ID,reference_config_sha256=sha256(A2_CONFIG),
            reference_initialization_file_sha256=sha256(A2_INIT),reference_initialization_sha256=init['direct_state_sha256'],
            independent_ImageNet_start=True,schedule_end=160000,training_stop_step=20000))
    return cfg

config._build=build
