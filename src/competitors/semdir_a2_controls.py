"""D1/D2: matched current A2 controls, independent first20K of160K."""
from copy import deepcopy
from pathlib import Path
import hashlib,json,os
import torch
from torch import nn
from torch.nn import functional as F
from mmengine.config import Config
from mmengine.dist import get_rank
from mmengine.registry import HOOKS
from mmseg.registry import MODELS
from research.ad_pcpf_v2 import config,experiments
from research.ad_pcpf_v2.artifacts import sha256,write_once
from research.ad_pcpf_v2.engine import PaperAuditHook,unwrap
from . import semdir_a2_cumulative as control
from . import direct_second as a2
from . import semdir_direct as direct
from .registry import EXPERIMENTS

IDS={'D1':'SemDiR-A2-D1-NoSAGDot-20Kof160K-S1',
     'D2':'SemDiR-A2-D2-F12Only-20Kof160K-S1'}

class A2ControlRelay(a2.DirectNoFineRelay):
    def prune_control(self,variant):
        self.control_variant=variant
        if variant=='D1':
            del self.semantic_align
        elif variant=='D2':
            object.__setattr__(self,'_multitap_initial_weight',self.semantic_proj.weight.detach().cpu().clone())
            with torch.random.fork_rng(devices=[]):new=nn.Conv2d(self.channels,self.hidden,1)
            with torch.no_grad():
                new.weight.copy_(self.semantic_proj.weight[:,-self.channels:]);new.bias.copy_(self.semantic_proj.bias)
            self.semantic_proj=new
        else:raise ValueError('Unknown current A2 control')

    def reset_relay(self):
        if hasattr(self,'semantic_align'):return super().reset_relay()
        nn.init.zeros_(self.score.weight);nn.init.zeros_(self.score.bias)
        with torch.no_grad():self.output.weight[:,:self.hidden].zero_()

    def forward(self,image,features,neck):
        if self.control_variant=='D1':return super().forward(image,features,neck)
        if len(features)!=4 or any(f.shape!=features[0].shape for f in features):raise ValueError('Four original decoder taps required')
        pyramid=neck(tuple(features))
        c=self.semantic_proj(direct.channel_norm(features[-1]))
        with torch.autocast(device_type=c.device.type,enabled=False):
            c=self.semantic_norm(c.float().movedim(1,-1)).movedim(-1,1)
        return self.refine(image,pyramid,c)

    def refine(self,image,pyramid,c):
        if self.control_variant=='D2':return super().refine(image,pyramid,c)
        q=self.rgb(F.interpolate(image,pyramid[0].shape[-2:],mode='bilinear',align_corners=False))
        c=c.to(q.dtype);gsem=F.interpolate(c,q.shape[-2:],mode='bilinear',align_corners=False)
        s0=self.score(torch.cat([gsem,q],1))
        with torch.autocast(device_type=q.device.type,enabled=False):
            gate=s0.float().tanh();fused=torch.cat([q.float(),gate*gsem.float()],1)
        delta=self.output(fused.to(q.dtype))
        self.last_diagnostics=dict(gate_mean=gate.detach().mean(),gate_abs_mean=gate.detach().abs().mean(),
            sag_score_rms=gate.new_zeros(()),update_rms=delta.detach().float().square().mean().sqrt())
        return (pyramid[0]+delta,*pyramid[1:])

@MODELS.register_module()
class A2ControlBackbone(a2.DirectNoFineBackbone):
    def __init__(self,control_variant,**kwargs):
        super().__init__(**kwargs)
        self.relay.__class__=A2ControlRelay;self.relay.prune_control(control_variant)
        self.method='semdir_a2_'+control_variant.lower()

@MODELS.register_module()
class A2ControlSegmentor(direct.DirectRegionPreservingSegmentor):
    backbone_type=A2ControlBackbone


def canonical_a2_digest(model):
    """Retained-initialization proof only; restored values are not model state."""
    state=dict(model.state_dict());relay=model.backbone.relay
    if relay.control_variant=='D1':
        key='backbone.relay.semantic_align.weight'
        if key in state:raise RuntimeError('D1 retained Wa')
        state[key]=torch.zeros(relay.hidden,relay.hidden,1,1,dtype=relay.score.weight.dtype)
    else:
        key='backbone.relay.semantic_proj.weight';full=relay._multitap_initial_weight
        if not torch.equal(state[key].detach().cpu(),full[:,-relay.channels:]):raise RuntimeError('Retained F12 weights changed before native initialization audit')
        state[key]=full
    digest=hashlib.sha256()
    for k,v in sorted(state.items()):
        digest.update(k.encode());digest.update(str((tuple(v.shape),v.dtype)).encode());digest.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

@HOOKS.register_module()
class A2ControlAuditHook(direct.DirectAuditHook):
    def before_train(self,runner):
        model=unwrap(runner.model)
        if not runner.cfg.resume:
            proof=runner.cfg.paper.a2_control
            if sha256(control.A2_INIT)!=proof.reference_initialization_file_sha256:raise RuntimeError('Reference A2 initialization changed')
            canonical=canonical_a2_digest(model)
            if canonical!=proof.reference_initialization_sha256:raise RuntimeError('Retained native initialization differs from A2')
        super().before_train(runner)
        if not runner.cfg.resume and get_rank()==0:
            write_once(Path(runner.work_dir)/'a2_control_initialization.json',dict(variant=proof.variant,
                actual_state_sha256=direct.state_digest(model),canonical_A2_initialization_sha256=canonical,
                scope='All retained A2 initial tensors; removed initial slices restored for comparison only',independent_ImageNet_start=True))

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        if runner.iter+1==200:
            r=unwrap(runner.model).backbone.relay;values=dict(H=r.output.weight[:,:r.hidden],score=r.score.weight)
            if hasattr(r,'semantic_align'):values['alignment']=r.semantic_align.weight
            learned={k:float(v.detach().float().abs().sum()) for k,v in values.items()}
            if any(not bool(torch.isfinite(v).all()) for v in values.values()) or any(v<=0 for v in learned.values()):raise RuntimeError('Active A2 control path failed to learn')
            if 'loss_rsp' not in (outputs or {}):raise RuntimeError('D1/D2 must preserve original RSP')
        PaperAuditHook.after_train_iter(self,runner,batch_idx,data_batch,outputs)
        if runner.iter+1==200 and get_rank()==0:
            root=Path(runner.work_dir).parents[1] if runner.cfg.paper.smoke else Path(runner.work_dir)
            write_once(root/'ablation_learning_probe.json',dict(iteration=200,variant=runner.cfg.paper.a2_control.variant,
                active_projection_l1=learned,all_active_paths_learned=True,RSP_loss_present=True,
                measurement='engineering smoke' if runner.cfg.paper.smoke else 'formal training'))

for variant,ID in IDS.items():
    if ID in EXPERIMENTS:raise RuntimeError('A2 control ID already registered')
    spec=a2.first.DirectScreenExperiment(ID,'m23_plain',backbone='deit_s',comparison='',seed_required=1)
    EXPERIMENTS[ID]=spec;experiments.EXPERIMENTS[ID]=spec
_parent_build=config._build

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs);ID=cfg['paper']['experiment']
    if ID not in IDS.values():return cfg
    variant=next(k for k,v in IDS.items() if v==ID);ref=Config.fromfile(str(control.A2_CONFIG),import_custom_modules=False).to_dict()
    cfg['param_scheduler']=deepcopy(ref['param_scheduler'])
    for key in control.SHARED_KEYS:
        if cfg[key]!=ref[key]:raise ValueError('A2 control changes recipe: '+key)
    for key in ['decode_head','auxiliary_head','data_preprocessor','test_cfg']:
        if cfg['model'][key]!=ref['model'][key]:raise ValueError('A2 control changes shared head: '+key)
    if cfg['load_from'] is not None or os.environ.get('DCPA_COMP_INIT'):raise ValueError('Independent ImageNet initialization required')
    cfg['model']=deepcopy(ref['model']);cfg['model']['backbone'].update(type='A2ControlBackbone',control_variant=variant)
    cfg['model']['type']='A2ControlSegmentor'
    cfg['custom_imports']['imports'].append('competitors.semdir_a2_controls');cfg['custom_hooks']=[dict(type='A2ControlAuditHook')]
    init=json.loads(control.A2_INIT.read_text())
    cfg['paper']['direct_semantics']=deepcopy(ref['paper']['direct_semantics'])
    cfg['paper']['direct_semantics'].update(SAG=variant=='D2',semantic_taps=[12] if variant=='D2' else [3,6,9,12],
        K=1 if variant=='D2' else 4,semantic_projection=[384 if variant=='D2' else 1536,32],fine_value=False,original_RSP=True)
    cfg['paper'].update(method='semdir_a2_'+variant.lower(),budget=20000,architecture_reference='SemDiR-A2',
        provenance_label='Matched current no-Vf A2 control '+variant+'; not historical Direct+Vf A1/A4',
        a2_control=dict(variant=variant,reference=control.A2_ID,reference_config_sha256=sha256(control.A2_CONFIG),
            reference_initialization_file_sha256=sha256(control.A2_INIT),reference_initialization_sha256=init['direct_state_sha256'],
            independent_ImageNet_start=True,training_stop_step=20000,schedule_end=160000,
            only_change='Remove Wa and its dot score; gate=tanh(s0)' if variant=='D1' else 'F12-only condition; retain four decoder taps',
            formula='P0+Wout[Q,gG]',decoder_taps=[3,6,9,12],RSP=True,Vf=False))
    return cfg

config._build=build
