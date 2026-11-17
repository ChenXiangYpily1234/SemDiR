"""C0 Vanilla -> C1 MSC; independent matched first20K of160K."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,os
import torch
from torch import nn
from mmengine.config import Config
from mmengine.registry import HOOKS
from mmengine.dist import get_rank
from mmseg.registry import MODELS
from research.ad_pcpf_v2 import config,experiments
from research.ad_pcpf_v2.adapter import ADPCPFPaperSegmentor
from research.ad_pcpf_v2.engine import PaperAuditHook,unwrap
from research.ad_pcpf_v2.artifacts import write_once,sha256
from . import semdir_a2_cumulative as control
from . import semdir_direct as direct
from .registry import EXPERIMENTS

BASE=Path(__file__).resolve().parents[2]
IDS={'C0':'SemDiR-A2-C0-Vanilla-20Kof160K-S1','C1':'SemDiR-A2-C1-MSC-20Kof160K-S1'}

def study_path(variant):return BASE/f'trained_models/SemDiR-A2-{variant}-20Kof160K-w4-b4-g16-s1-20261116-R2'

def shared_digest(model):
    digest=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        if k.startswith('backbone.relay.'):continue
        digest.update(k.encode());digest.update(str((tuple(v.shape),v.dtype)).encode())
        digest.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

class MSCOnlyRelay(nn.Module):
    def prune(self):
        with torch.random.fork_rng(devices=[]):output=nn.Conv2d(self.hidden,self.channels,1,bias=False)
        self.output=output
        for key in ['rgb','score','semantic_align','gamma']:delattr(self,key)
        self.reset_relay()

    def reset_relay(self):nn.init.zeros_(self.output.weight)

    def forward(self,image,features,neck):
        pyramid=neck(tuple(features))
        c=self.semantic_proj(torch.cat([direct.channel_norm(f) for f in features],1))
        with torch.autocast(device_type=c.device.type,enabled=False):
            c=self.semantic_norm(c.float().movedim(1,-1)).movedim(-1,1)
        g=torch.nn.functional.interpolate(c.to(pyramid[0].dtype),pyramid[0].shape[-2:],mode='bilinear',align_corners=False)
        delta=self.output(g)
        self.last_diagnostics=dict(update_rms=delta.detach().float().square().mean().sqrt(),
            semantic_condition_rms=c.detach().float().square().mean().sqrt())
        return (pyramid[0]+delta,*pyramid[1:])

@MODELS.register_module()
class MSCOnlyBackbone(direct.SemDiRDirectBackbone):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.relay.__class__=MSCOnlyRelay;self.relay.prune();self.method='semdir_a2_msc_only'

@HOOKS.register_module()
class AdditiveAuditHook(PaperAuditHook):
    def before_train(self,runner):
        model=unwrap(runner.model);variant=runner.cfg.paper.a2_additive.variant
        if not runner.cfg.resume:
            raw=direct.state_digest(model);shared=shared_digest(model);root=Path(runner.work_dir).parents[1]
            own=root/'additive_smoke_initialization.json'
            if not runner.cfg.paper.smoke and json.loads(own.read_text())['state_sha256']!=raw:raise RuntimeError('Additive smoke/formal initialization mismatch')
            if variant=='C1':
                ref=study_path('C0')/'runs'/IDS['C0']/'additive_initialization.json'
                if shared!=json.loads(ref.read_text())['shared_state_sha256']:raise RuntimeError('C1 shared native initialization differs from C0')
            self.initial_head=model.decode_head.conv_seg.weight.detach().clone()
            if variant=='C1':self.initial_semantic=model.backbone.relay.semantic_proj.weight.detach().clone()
        super().before_train(runner)
        if not runner.cfg.resume and get_rank()==0:
            write_once(Path(runner.work_dir)/'additive_initialization.json',dict(variant=variant,state_sha256=raw,shared_state_sha256=shared,
                independent_ImageNet_start=True,scope='CPU retained initialization matches A2; native C0/C1 shared tensors equal; own smoke/formal entire state equal'))
            if runner.cfg.paper.smoke:write_once(own,dict(state_sha256=raw,shared_state_sha256=shared))

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        if runner.iter+1==200:
            model=unwrap(runner.model);variant=runner.cfg.paper.a2_additive.variant
            learned=dict(decode_head_change=float((model.decode_head.conv_seg.weight.detach()-self.initial_head).abs().sum()))
            if variant=='C1':
                r=model.backbone.relay
                learned.update(semantic_projection_change=float((r.semantic_proj.weight.detach()-self.initial_semantic).abs().sum()),MSC_output_l1=float(r.output.weight.detach().abs().sum()))
            if any(not torch.isfinite(torch.tensor(v)) or v<=0 for v in learned.values()):raise RuntimeError('Additive active path did not learn')
            if 'loss_rsp' in (outputs or {}):raise RuntimeError('C0/C1 must not execute RSP')
        super().after_train_iter(runner,batch_idx,data_batch,outputs)
        if runner.iter+1==200 and get_rank()==0:
            root=Path(runner.work_dir).parents[1] if runner.cfg.paper.smoke else Path(runner.work_dir)
            write_once(root/'ablation_learning_probe.json',dict(iteration=200,variant=variant,active_projection_l1=learned,
                all_active_paths_learned=True,RSP_loss_present=False,measurement='engineering smoke' if runner.cfg.paper.smoke else 'formal training'))

for variant,ID in IDS.items():
    if ID in EXPERIMENTS:raise RuntimeError('Additive ID already registered')
    spec=control.a2.first.DirectScreenExperiment(ID,'m23_plain',backbone='deit_s',comparison='',seed_required=1)
    EXPERIMENTS[ID]=spec;experiments.EXPERIMENTS[ID]=spec
_parent_build=config._build

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs);ID=cfg['paper']['experiment']
    if ID not in IDS.values():return cfg
    variant=next(k for k,v in IDS.items() if v==ID);ref=Config.fromfile(str(control.A2_CONFIG),import_custom_modules=False).to_dict()
    cfg['param_scheduler']=deepcopy(ref['param_scheduler'])
    for key in control.SHARED_KEYS:
        if cfg[key]!=ref[key]:raise ValueError('Additive changes A2 recipe: '+key)
    for key in ['decode_head','auxiliary_head','data_preprocessor','test_cfg']:
        if cfg['model'][key]!=ref['model'][key]:raise ValueError('Additive changes shared head: '+key)
    if cfg['load_from'] is not None or os.environ.get('DCPA_COMP_INIT'):raise ValueError('Independent ImageNet start required')
    bb=deepcopy(ref['model']['backbone']);bb.pop('direct_hidden',None)
    if variant=='C0':bb.update(type='ADPCPFPaperBackbone',method='vanilla')
    else:bb.update(type='MSCOnlyBackbone',direct_hidden=32)
    cfg['model']['backbone']=bb;cfg['model']['type']='ADPCPFPaperSegmentor';cfg['model'].pop('region_preservation',None)
    cfg['custom_imports']['imports'].append('competitors.semdir_a2_additive');cfg['custom_hooks']=[dict(type='AdditiveAuditHook')]
    cfg['paper'].update(method='semdir_a2_'+variant.lower(),budget=20000,architecture_reference='SemDiR-A2-additive',
        provenance_label='Independent matched additive20K: '+variant+'; no SAG/RSP/Vf; no old M03 or independent20K M24 substitution',
        a2_additive=dict(variant=variant,MSC=variant=='C1',SAGR=False,RSP=False,reference=control.A2_ID,
            reference_config_sha256=sha256(control.A2_CONFIG),training_stop_step=20000,schedule_end=160000,
            formula='P0' if variant=='C0' else 'P0+Wc*Up(C)',MSC_output_zero_initialized=variant=='C1'))
    return cfg

config._build=build
