"""Independent Selective E1/E2: retain all SemDiR-Exact paths and RSP."""
from __future__ import annotations
import weakref
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from mmseg.registry import MODELS
from mmengine.registry import HOOKS
from mmengine.dist import get_rank,get_world_size
from .registration import CompetitorAuditHook
from research.ad_pcpf_v2.artifacts import shared_digest,write_once,identity
from research.ad_pcpf_v2.runtime import runtime_from_config
from research.ad_pcpf_v2.engine import unwrap
from pathlib import Path
from . import registration
from .semdir_exact import ExactPlainFusion,ExactRelay,SemDiRExactBackbone
from .exact_context import FactorizedDepthContext
from .registry import EXPERIMENTS,CompetitorExperiment
from research.ad_pcpf_v2 import config,experiments
from research.ad_pcpf_v2.transport import CoarseFineResidualTransport


def cross_depth_disagreement(queries,eps=1e-6):
    if len(queries)!=4 or any(x.shape!=queries[0].shape for x in queries):
        raise ValueError('Four aligned existing semantic query maps required')
    with torch.autocast(device_type=queries[0].device.type,enabled=False):
        unit=[x.float()/(x.float().norm(dim=1,keepdim=True)+eps) for x in queries]
        return (1.-torch.stack(unit).mean(0).square().sum(1,keepdim=True)).clamp(0.,1.)


class FineValue(nn.Module):
    """Disjoint-bin local deviation, semantic modulation, bias-free direction."""
    geometry=staticmethod(CoarseFineResidualTransport.geometry)
    def __init__(self,hidden,output_channels):
        super().__init__();self.project=nn.Conv2d(hidden,output_channels,1,bias=False)

    def forward(self,q,c):
        b,d,H,W=q.shape;h,w=c.shape[-2:]
        if c.shape[:2]!=(b,d):raise ValueError('Coarse and fine query channels/batch must agree')
        ids,_=self.geometry(H,W,h,w,q.device)
        with torch.autocast(device_type=q.device.type,enabled=False):
            index=ids[None,None].expand(b,d,-1);fine=q.float().flatten(2)
            sums=fine.new_zeros(b,d,h*w).scatter_add(2,index,fine)
            counts=torch.bincount(ids,minlength=h*w).to(fine.dtype)[None,None]
            mean=(sums/counts).gather(2,index)
            semantic=c.float().flatten(2).gather(2,index).tanh()
            e=((fine-mean)*semantic).reshape(b,d,H,W)
        return self.project(e.to(q.dtype))


class SelectiveDepthContext(FactorizedDepthContext):
    def prepare_shared(self,features):
        self.selective_queries=[]
        return super().prepare_shared(features)

    def _select(self,prepared,semantic,diagnostics):
        first=prepared['features'][0]
        with torch.autocast(device_type=first.device.type,enabled=False):
            c=self.channels
            projected_semantic=F.conv2d(semantic,self.gate[0].weight[:,2*c:3*c])
            if self.selective_capture:self.selective_queries.append(projected_semantic)
            detail=torch.zeros_like(first,dtype=torch.float32);gm,cr=[],[]
            for w,projected,d in zip(prepared['weights'],prepared['projected_depth'],prepared['details']):
                hidden=projected+projected_semantic+prepared['projected_support']
                gate=self.gate[2](self.gate[1](hidden)).sigmoid()
                contribution=w*gate*d;detail=detail+contribution
                if diagnostics:
                    gm.append(gate.detach().mean());cr.append(contribution.detach().square().mean().sqrt())
        out=(prepared['anchor'],detail,prepared['support'])
        if diagnostics:
            return (*out,dict(beta=self.beta.detach(),weights=prepared['weights'].detach(),gate_mean=torch.stack(gm),contribution_rms=torch.stack(cr)))
        return out


class SelectiveFusion(ExactPlainFusion):
    def forward(self,residuals,rgb,q=None):
        c=self.coarse(torch.cat(residuals,1));c=self.norm(c.movedim(1,-1)).movedim(-1,1)
        H,W=rgb.shape[-2:];_,pos=self.geometry(H,W,*c.shape[-2:],rgb.device)
        expanded=F.interpolate(c,(H,W),mode='bilinear',align_corners=False)
        if q is None:q=self.rgb(rgb)
        score=self.score(torch.cat((expanded*q,(expanded-q).abs(),q,pos[None].expand(rgb.shape[0],-1,-1,-1).to(q.dtype)),1))
        diagnostics={}
        logits=score.float()
        if self.selective_variant in ('e1','e3','e4','e6'):
            queries=self._selective_context_ref().selective_queries
            D=cross_depth_disagreement(queries)
            D=F.interpolate(D,(H,W),mode='bilinear',align_corners=False)
            logits=logits+self.disagreement_eta.float()*(D-D.mean((-2,-1),keepdim=True))
            diagnostics.update(selective_eta=self.disagreement_eta.detach().float(),selective_disagreement_mean=D.detach().mean())
        coarse_value=F.interpolate(self.value(c),(H,W),mode='bilinear',align_corners=False)
        value=coarse_value.float()
        if self.selective_variant in ('e2','e3','e4','e7','e8','e9','e10','e11'):
            vf=self.fine_value(q,c)
            value=value+self.fine_gamma.float()*vf.float()
            diagnostics.update(selective_gamma=self.fine_gamma.detach().float(),selective_fine_value_rms=vf.detach().float().square().mean().sqrt())
        gate=logits.tanh();extra=(gate*value).to(rgb.dtype)
        self.last_diagnostics=dict(transport_update_rms=extra.detach().float().square().mean().sqrt(),transport_value_rms=coarse_value.detach().float().square().mean().sqrt(),transport_allocation_rms=gate.detach().square().mean().sqrt(),**diagnostics)
        return extra


class SelectiveRelay(ExactRelay):
    def forward(self,*args,**kwargs):
        try:return super().forward(*args,**kwargs)
        finally:self.context.selective_queries=[]



class AllocatedSelectiveRelay(SelectiveRelay):
    """Route actual R with independent or shared sigmoid injection coefficients."""
    @property
    def routing_beta(self):
        return self.routing_theta.float().sigmoid().expand(4)

    def route_residuals(self, base_features, actual_updates, refined):
        beta = self.routing_beta
        direct, condition = [], []
        for index, (feature, update) in enumerate(zip(base_features, actual_updates)):
            direct_update = (beta[index] * update.float()).to(feature.dtype)
            condition_update = (update if getattr(self,'routing_full_condition',False)
                                else ((1.-beta[index]) * update.float()).to(update.dtype))
            direct.append(feature + direct_update)
            condition.append(condition_update)
        with torch.no_grad():
            self.last_routing_ratio=torch.stack([
                (beta[k]*r.float()).norm()/(f.float().norm()+1e-6)
                for k,(f,r) in enumerate(zip(base_features,actual_updates))])
        return tuple(direct), condition

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        self.last_diagnostics.update(routing_beta=self.routing_beta.detach(),
                                     routing_theta=self.routing_theta.detach(),
                                     routing_effective_ratio=self.last_routing_ratio)
        return output


def upgrade(relay,variant):
    if type(relay) is not ExactRelay or type(relay.transport) is not ExactPlainFusion:
        raise ValueError('Selective requires unchanged SemDiR-Exact Full relay')
    if variant not in ('e1','e2','e3','e4','e5','e6','e7','e8','e9','e10','e11'):raise ValueError('Expected one of the registered E1-E11 variants')
    if variant=='e5':
        # ConditionOnly: retain the exact E0 operators, parameters and RNG.
        # Residuals still feed C/g/V; only their direct pyramid use is removed.
        relay.inject_pyramid=False
        return relay
    f=relay.transport;device=f.value.weight.device;dtype=f.value.weight.dtype
    with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
        if variant in ('e1','e3','e4','e6'):f.disagreement_eta=nn.Parameter(torch.zeros((),device=device,dtype=dtype))
        if variant in ('e2','e3','e4','e7','e8','e9','e10','e11'):
            f.fine_value=FineValue(f.value.in_channels,f.output_channels).to(device=device,dtype=dtype)
            f.fine_gamma=nn.Parameter(torch.zeros((),device=device,dtype=dtype))
    relay.context.__class__=SelectiveDepthContext;relay.context.selective_capture=variant in ('e1','e3','e4','e6');relay.context.selective_queries=[]
    f.__class__=SelectiveFusion;f.selective_variant=variant
    object.__setattr__(f,'_selective_context_ref',weakref.ref(relay.context))
    relay.__class__=SelectiveRelay
    if variant in ('e4','e6','e7','e8','e9','e10','e11'):relay.inject_pyramid=False
    if variant in ('e8','e9','e10','e11'):
        import math
        initial=.02 if variant=='e8' else .25
        relay.routing_theta=nn.Parameter(torch.full((1 if variant=='e11' else 4,),math.log(initial/(1-initial)),device=device,dtype=torch.float32))
        relay.routing_full_condition=variant in ('e10','e11')
        relay.__class__=AllocatedSelectiveRelay
    return relay


@MODELS.register_module()
class SemDiRSelectiveBackbone(SemDiRExactBackbone):
    def __init__(self,selective_variant,**kwargs):
        super().__init__(**kwargs);upgrade(self.relay,selective_variant)


def selective_common_digest(model):
    """Hash all shared tensors, including relay; exclude only new E1/E2 params."""
    excluded={'backbone.relay.transport.disagreement_eta',
              'backbone.relay.transport.fine_gamma',
              'backbone.relay.transport.fine_value.project.weight',
              'backbone.relay.routing_theta'}
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        if key in excluded:continue
        digest.update(key.encode());digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def require_smoke_common_initialization(actual,receipt):
    expected=receipt.get('initialization',{}).get('selective_common_state_sha256')
    if receipt.get('status')!='PASS' or expected is None or actual!=expected:
        raise RuntimeError('Formal Selective initialization does not match full-scope smoke lineage')


@HOOKS.register_module()
class SelectiveAuditHook(CompetitorAuditHook):
    """Native gradient audit plus complete common-parameter init lineage."""
    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        super().after_train_iter(runner,batch_idx,data_batch,outputs)
        iteration=runner.iter+1
        relay=unwrap(runner.model).backbone.relay
        if hasattr(relay,'routing_theta') and (iteration==1 or iteration%50==0):
            ratio=relay.last_routing_ratio.detach().clone()
            if get_world_size()>1:
                torch.distributed.all_reduce(ratio)
                ratio/=get_world_size()
            self.routing_ratio_global=ratio.cpu()
        if hasattr(relay,'routing_theta') and get_rank()==0 and (iteration==1 or iteration%50==0):
            beta=relay.routing_beta.detach().cpu()
            if not torch.isfinite(beta).all():raise RuntimeError('Nonfinite residual allocation')
            row=dict(iteration=iteration,routing_beta=beta.tolist(),routing_theta=relay.routing_theta.detach().float().cpu().tolist(),fine_gamma=float(relay.transport.fine_gamma.detach().float().cpu()),timing='after_optimizer_step')
            # Forward diagnostics predate this optimizer step; average ratios
            # across all ranks so rank0 is not mistaken for the global batch.
            row.update(routing_parameter_count=relay.routing_theta.numel(),
                       condition_path='full_R' if relay.routing_full_condition else 'complementary_R',
                       effective_ratio_timing='last_train_forward_before_optimizer_step')
            row['effective_ratio']=self.routing_ratio_global.tolist()
            row['effective_ratio_reduction']='mean of per-rank batch L2 ratios'
            with (Path(runner.work_dir)/'residual_allocation.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

    def before_train(self,runner):
        runtime=runtime_from_config(runner.cfg)
        if get_world_size()!=runtime['world_size'] or runner.cfg.train_dataloader.batch_size!=runtime['per_gpu']:
            raise RuntimeError('Selective DDP layout mismatch')
        model=unwrap(runner.model);self.handles=[];self.seen=set()
        self.finite=torch.ones((),dtype=torch.bool,device=next(model.parameters()).device)
        if not runner.cfg.resume:
            shared=shared_digest(model);common=selective_common_digest(model)
            if not runner.cfg.paper.smoke:
                receipt=Path(runner.work_dir).parents[1]/'smoke_receipts'/(runner.cfg.paper.experiment+'.json')
                require_smoke_common_initialization(common,json.loads(receipt.read_text()))
            model.backbone.check_zero_initialization(torch.zeros(1,3,65,79,device=self.finite.device))
            if get_rank()==0:
                write_once(Path(runner.work_dir)/'initialization.json',dict(
                    **identity(runner.cfg,runner.cfg.paper_config_sha256),shared_state_sha256=shared,
                    selective_common_state_sha256=common,
                    selective_common_scope='All state tensors including relay; excludes only eta, gamma, fine_value.project.weight and routing_theta',
                    zero_residual_check='applicable_only_to_M21_extensions',pretraining_sha256=runner.cfg.paper.pretrained_sha256))
        if runner.iter<200:
            self.expected={n for n,p in model.named_parameters() if p.requires_grad}
            def record(name):
                def hook(g):self.seen.add(name);self.finite.logical_and_(torch.isfinite(g).all());return g
                return hook
            for name,parameter in model.named_parameters():
                if parameter.requires_grad:self.handles.append(parameter.register_hook(record(name)))


REFERENCE='M23-PlainFusion'  # Existing sealed independent20K; no new E0 implementation/training.
CANDIDATES={f'SemDiR-Selective-E{i}-S1':f'e{i}' for i in range(1,12)}
for name in CANDIDATES:
    if name in EXPERIMENTS:raise RuntimeError('Selective experiment identity already exists')
    EXPERIMENTS[name]=CompetitorExperiment(name,'m23_plain',backbone='deit_s',comparison=REFERENCE,seed_required=1)
    experiments.EXPERIMENTS[name]=EXPERIMENTS[name]
_parent_build=config._build

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs);name=cfg['paper']['experiment']
    if name not in CANDIDATES:return cfg
    if cfg['train_cfg']['max_iters']!=20000:raise ValueError('Selective screening is20K only; no implicit160K')
    variant=CANDIDATES[name];cfg['custom_imports']['imports'].append('competitors.selective');cfg['custom_hooks']=[dict(type='SelectiveAuditHook')]
    if variant:cfg['model']['backbone'].update(type='SemDiRSelectiveBackbone',selective_variant=variant)
    cfg['paper']['selective_screening']=dict(variant=variant or 'e0',budget=20000,reference=REFERENCE,independent_imagenet_initialization=True,semantic_queries='existing first-gate semantic projection B, differentiable',fine_geometry='disjoint bins, nearest semantic expansion, bias-free fine projection',zero_scalar_initialization=True,formal_method_unchanged=True,disagreement=variant in ('e1','e3','e4','e6'),fine_value=variant in ('e2','e3','e4','e7','e8','e9','e10','e11'),inject_pyramid=variant not in ('e4','e5','e6','e7'),E4_parent='E3; removes only direct pyramid injection, retains R generators for C and V',E5_parent='E0 unchanged fusion; removes only direct pyramid injection; no D/Vf/new parameters')
    if variant in ('e6','e7'):
        cfg['paper']['selective_screening']['parent']=('E1' if variant=='e6' else 'E2')+'; only disable direct R-to-pyramid injection; retain all residual generators and H/RSP'
    if variant=='e8':
        cfg['paper']['selective_screening'].update(inject_pyramid='adaptive_beta',parent='E7 + complementary residual allocation',routing_beta_initial=.02,routing_theta_initial='log(.02/.98)',direct_path='beta_k*R_k',condition_path='(1-beta_k)*R_k',routing_parameters=4,routing_optimizer='unchanged default LR6e-5/WD.01',routing_log_interval=50)
    if variant in ('e9','e10','e11'):
        shared=variant=='e11';full=variant in ('e10','e11')
        cfg['paper']['selective_screening'].update(
            inject_pyramid='adaptive_shared_alpha' if shared else 'adaptive_beta',
            parent={'e9':'E8; only initialization .02 to .25','e10':'E9; only use full R for condition','e11':'E10; only share four coefficients as one'}[variant],
            routing_beta_initial=.25,routing_theta_initial='log(.25/.75)',
            direct_path='alpha*R_k' if shared else 'beta_k*R_k',
            condition_path='R_k' if full else '(1-beta_k)*R_k',
            routing_parameters=1 if shared else 4,
            routing_optimizer='unchanged default LR6e-5/WD.01',routing_log_interval=50,
            routing_effective_ratio='mean across ranks of ||coefficient*R_k||_2/(||F_k||_2+1e-6)',
            zero_scalar_initialization=False,zero_fine_gamma_initialization=True,
            method_name='SemDiR-SR' if shared else 'SemDiR-SR controlled ablation')
    return cfg
config._build=build
