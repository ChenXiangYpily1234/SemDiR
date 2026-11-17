"""Independent current-A2 RGB/fusion/gate/RSP controls on the fixed160K LR curve."""
from copy import deepcopy
from pathlib import Path
import json
import math
import os
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
from . import semdir_a2_cumulative as control
from . import direct_second as a2
from . import semdir_direct as direct
from .semdir_a2_additive import shared_digest, study_path as additive_study, IDS as ADDITIVE_IDS
from .registry import EXPERIMENTS, CompetitorExperiment

IDS={
    'RGB':'SemDiR-A2-A-RGBOnly-NoRSP-20Kof160K-S1',
    'GLOBAL':'SemDiR-A2-B-GlobalGate-NoRSP-20Kof160K-S1',
    'UNGATED':'SemDiR-A2-E2-UngatedFusion-20Kof160K-S1',
    'SIGMOID':'SemDiR-A2-E3-SigmoidG05-20Kof160K-S1',
    'TANH_MATCHED':'SemDiR-A2-E4-TanhG05-20Kof160K-S1',
    'RSP_ONLY':'SemDiR-A2-E5-VanillaRSP-20Kof160K-S1',
    'NO_RSP_160K':'SemDiR-A2-C2-NoRSP-160K-S1',
}
FORMULAS=dict(RGB='P0+WqQ',GLOBAL='P0+WqQ+tanh(beta)WgG',UNGATED='P0+Wout[Q,G]',
    SIGMOID='P0+Wout[Q,sigmoid(s)G]',TANH_MATCHED='P0+Wout[Q,tanh(s+atanh(0.5))G]',
    RSP_ONLY='P0; original RSP training loss',NO_RSP_160K='P0+Wout[Q,tanh(s)G]; no RSP')

def gate_value(score,variant):
    if variant=='SIGMOID':return score.sigmoid()
    if variant=='TANH_MATCHED':return (score+math.atanh(.5)).tanh()
    raise ValueError('No learned gate for '+variant)

class MechanismRelay(a2.DirectNoFineRelay):
    def configure(self,variant):
        if variant not in ['RGB','GLOBAL','UNGATED','SIGMOID','TANH_MATCHED']:raise ValueError(variant)
        self.variant=variant
        if variant=='RGB':
            with torch.random.fork_rng(devices=[]):out=nn.Conv2d(self.hidden,self.channels,1,bias=False)
            self.output=out
            for name in ['semantic_proj','semantic_norm','score','semantic_align']:delattr(self,name)
        elif variant in ['UNGATED','GLOBAL']:
            del self.score,self.semantic_align
            if variant=='GLOBAL':self.beta=nn.Parameter(torch.zeros(()))
        self.reset_relay()

    def reset_relay(self):
        if not hasattr(self,'variant'):return super().reset_relay()
        if self.variant=='GLOBAL':
            with torch.no_grad():self.output.weight[:,:self.hidden].zero_();self.beta.zero_()
        else:nn.init.zeros_(self.output.weight)
        if hasattr(self,'score'):
            nn.init.zeros_(self.score.weight);nn.init.zeros_(self.score.bias)
            nn.init.zeros_(self.semantic_align.weight)

    def forward(self,image,features,neck):
        if self.variant!='RGB':return super().forward(image,features,neck)
        pyramid=neck(tuple(features))
        q=self.rgb(F.interpolate(image,pyramid[0].shape[-2:],mode='bilinear',align_corners=False))
        delta=self.output(q)
        self.last_diagnostics=dict(update_rms=delta.detach().float().square().mean().sqrt())
        return (pyramid[0]+delta,*pyramid[1:])

    def refine(self,image,pyramid,c):
        q=self.rgb(F.interpolate(image,pyramid[0].shape[-2:],mode='bilinear',align_corners=False))
        c=c.to(q.dtype);gsem=F.interpolate(c,q.shape[-2:],mode='bilinear',align_corners=False)
        if self.variant=='UNGATED':
            fused=torch.cat([q,gsem],1)
        elif self.variant=='GLOBAL':
            with torch.autocast(device_type=q.device.type,enabled=False):
                gate=self.beta.float().tanh()
                fused=torch.cat([q.float(),gate*gsem.float()],1)
        else:
            aligned=F.interpolate(self.semantic_align(c),q.shape[-2:],mode='bilinear',align_corners=False)
            score=self.score(torch.cat([gsem,q],1))
            with torch.autocast(device_type=q.device.type,enabled=False):
                sa=(aligned.float()*q.float()).sum(1,keepdim=True)/math.sqrt(self.hidden)
                gate=gate_value(score.float()+sa,self.variant)
                fused=torch.cat([q.float(),gate*gsem.float()],1)
        delta=self.output(fused.to(q.dtype))
        self.last_diagnostics=dict(update_rms=delta.detach().float().square().mean().sqrt())
        if self.variant=='GLOBAL':
            self.last_diagnostics.update(global_gate=gate.detach(),global_beta=self.beta.detach().float())
        elif self.variant!='UNGATED':
            self.last_diagnostics.update(gate_mean=gate.detach().mean(),gate_abs_mean=gate.detach().abs().mean(),
                gate_negative_fraction=(gate.detach()<0).float().mean(),sag_score_rms=sa.detach().square().mean().sqrt())
        return (pyramid[0]+delta,*pyramid[1:])

@MODELS.register_module()
class MechanismBackbone(a2.DirectNoFineBackbone):
    def __init__(self,mechanism_variant,**kwargs):
        super().__init__(**kwargs)
        self.relay.__class__=MechanismRelay;self.relay.configure(mechanism_variant)
        self.method='semdir_mechanism_'+mechanism_variant.lower()

@MODELS.register_module()
class MechanismSegmentor(direct.DirectRegionPreservingSegmentor):
    backbone_type=MechanismBackbone

@MODELS.register_module()
class MechanismWithoutRSPSegmentor(MechanismSegmentor):
    def __init__(self,region_preservation,**kwargs):
        expected=dict(direct.registration.RSP,enabled=False,max_weight=0.)
        if dict(region_preservation)!=expected:raise ValueError('A/B must disable original RSP only')
        super().__init__(region_preservation=deepcopy(direct.registration.RSP),**kwargs)
        self.region_preservation=deepcopy(expected)

@MODELS.register_module()
class VanillaRSPSegmentor(direct.DirectRegionPreservingSegmentor):
    backbone_type=direct.ADPCPFPaperBackbone

@HOOKS.register_module()
class MechanismAuditHook(PaperAuditHook):
    def before_train(self,runner):
        model=unwrap(runner.model)
        if not runner.cfg.resume:
            actual=direct.state_digest(model);shared=shared_digest(model)
            proof=runner.cfg.paper.mechanism_control
            reference=Path(proof.shared_initialization_file)
            if sha256(reference)!=proof.shared_initialization_file_sha256:raise RuntimeError('C0 initialization evidence changed')
            if shared!=json.loads(reference.read_text())['shared_state_sha256']:raise RuntimeError('Shared backbone/pyramid/head initialization differs from C0')
            receipt=Path(runner.work_dir).parents[1]/('mechanism_smoke_initialization-'+runner.cfg.paper.experiment+'.json')
            if not runner.cfg.paper.smoke and actual!=json.loads(receipt.read_text())['state_sha256']:raise RuntimeError('Own smoke/formal initialization mismatch')
        super().before_train(runner)
        if not runner.cfg.resume and get_rank()==0:
            value=dict(variant=proof.variant,state_sha256=actual,shared_state_sha256=shared,
                independent_ImageNet_start=True,zero_update_check='PASS',formula=proof.formula)
            write_once(Path(runner.work_dir)/'mechanism_initialization.json',value)
            if runner.cfg.paper.smoke:write_once(receipt,value)

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        variant=runner.cfg.paper.mechanism_control.variant
        if runner.iter+1==200:
            model=unwrap(runner.model)
            rsp=bool(runner.cfg.paper.mechanism_control.original_RSP)
            if ('loss_rsp' in (outputs or {}))!=rsp:raise RuntimeError('RSP execution does not match control')
            learned={}
            relay=getattr(model.backbone,'relay',None)
            if relay is not None:
                for name in ['output','score','semantic_align']:
                    if hasattr(relay,name):
                        value=getattr(relay,name).weight.detach().float()
                        if not torch.isfinite(value).all() or value.abs().sum()<=0:raise RuntimeError('Active path did not learn: '+name)
                        learned[name]=float(value.abs().sum())
                if hasattr(relay,'beta'):
                    beta=relay.beta.detach().float()
                    if not torch.isfinite(beta) or beta.abs()<=0:raise RuntimeError('Global beta did not learn')
                    learned['global_beta']=float(beta);learned['global_gate']=float(beta.tanh())
            if get_rank()==0:
                root=Path(runner.work_dir).parents[1] if runner.cfg.paper.smoke else Path(runner.work_dir)
                write_once(root/('mechanism_learning_probe-'+runner.cfg.paper.experiment+'.json'),dict(iteration=200,variant=variant,
                    active_projection_l1=learned,RSP_loss_present=rsp,all_active_paths_learned=True))
        super().after_train_iter(runner,batch_idx,data_batch,outputs)

for variant,ID in IDS.items():
    if ID in EXPERIMENTS:raise RuntimeError('Duplicate mechanism experiment')
    cls=CompetitorExperiment if variant=='NO_RSP_160K' else a2.first.DirectScreenExperiment
    EXPERIMENTS[ID]=cls(ID,'m23_plain',backbone='deit_s',comparison='',seed_required=1)
    experiments.EXPERIMENTS[ID]=EXPERIMENTS[ID]
_parent_build=config._build

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs);ID=cfg['paper']['experiment']
    if ID not in IDS.values():return cfg
    variant=next(k for k,v in IDS.items() if v==ID)
    ref=Config.fromfile(str(control.A2_CONFIG),import_custom_modules=False).to_dict()
    cfg['param_scheduler']=deepcopy(ref['param_scheduler'])
    for key in control.SHARED_KEYS:
        if key!='train_cfg' and cfg[key]!=ref[key]:raise ValueError('Control changes recipe: '+key)
    for key in ['decode_head','auxiliary_head','data_preprocessor','test_cfg']:
        if cfg['model'][key]!=ref['model'][key]:raise ValueError('Control changes shared head: '+key)
    if cfg['load_from'] is not None or os.environ.get('DCPA_COMP_INIT'):raise ValueError('Independent ImageNet start required')
    cfg['model']=deepcopy(ref['model']);bb=cfg['model']['backbone']
    if variant=='RSP_ONLY':
        bb.pop('direct_hidden');bb.update(type='ADPCPFPaperBackbone',method='vanilla')
        cfg['model']['type']='VanillaRSPSegmentor'
    elif variant=='NO_RSP_160K':
        cfg['model'].update(type='A2WithoutRSPSegmentor',region_preservation=dict(direct.registration.RSP,enabled=False,max_weight=0.))
    else:
        bb.update(type='MechanismBackbone',mechanism_variant=variant)
        cfg['model']['type']='MechanismSegmentor'
        if variant in ['RGB','GLOBAL']:
            cfg['model'].update(type='MechanismWithoutRSPSegmentor',region_preservation=dict(direct.registration.RSP,enabled=False,max_weight=0.))
    stop=160000 if variant=='NO_RSP_160K' else 20000
    cfg['train_cfg']=dict(ref['train_cfg'],max_iters=stop)
    cfg['custom_imports']['imports'].append('competitors.semdir_mechanism')
    cfg['custom_hooks']=[dict(type='MechanismAuditHook')]
    initial=additive_study('C0')/'runs'/ADDITIVE_IDS['C0']/'additive_initialization.json'
    cfg['paper'].update(method='semdir_mechanism_'+variant.lower(),budget=stop,architecture_reference='SemDiR-A2',
        provenance_label='Independent current A2 mechanism control; RSP as specified; no Vf/cross-layer beta; no checkpoint warm start',
        mechanism_control=dict(variant=variant,formula=FORMULAS[variant],reference=control.A2_ID,
            reference_config_sha256=sha256(control.A2_CONFIG),shared_initialization_file=str(initial),
            shared_initialization_file_sha256=sha256(initial),training_stop_step=stop,schedule_end=160000,
            gate_initial=.5 if variant in ['SIGMOID','TANH_MATCHED'] else 0. if variant=='GLOBAL' else None,
            output_initialization='original A2; Q block zero' if variant in ['NO_RSP_160K','GLOBAL'] else 'all zero' if variant!='RSP_ONLY' else 'no refinement',
            pairwise_activation_comparator=IDS['TANH_MATCHED'] if variant=='SIGMOID' else IDS['SIGMOID'] if variant=='TANH_MATCHED' else None,
            original_RSP=variant not in ['RGB','GLOBAL','NO_RSP_160K']))
    cfg['paper'].pop('direct_semantics',None)
    return cfg

config._build=build
