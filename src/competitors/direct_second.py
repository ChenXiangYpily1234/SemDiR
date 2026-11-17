"""Independent A2/A4: physical Vf removal / F12-only semantic conditioning."""
from copy import deepcopy
import hashlib,json,math
from pathlib import Path
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
from . import direct_ablations as first
from . import semdir_direct as direct
from .registry import EXPERIMENTS
FULL_CONFIG=first.FULL_CONFIG
FULL_INITIALIZATION=first.FULL_INITIALIZATION
A2_ID='SemDiR-Direct-A2-NoVf-20Kof160K-S1'
A4_ID='SemDiR-Direct-A4-F12Only-20Kof160K-S1'
VARIANTS={A2_ID:'A2_NoVf',A4_ID:'A4_F12Only'}

class DirectNoFineRelay(direct.DirectSemanticRelay):
 def __init__(self,channels=384,hidden=32):
  super().__init__(channels,hidden)
  self.prune()
 def prune(self):
  object.__setattr__(self,'_full_output',self.output.weight.detach().cpu().clone())
  with torch.random.fork_rng(devices=[]):new=nn.Conv2d(2*self.hidden,self.channels,1,bias=False)
  with torch.no_grad():new.weight.copy_(self.output.weight[:,:2*self.hidden])
  self.output=new
  del self.gamma
 def reset_relay(self):
  if hasattr(self,'gamma'):return super().reset_relay()
  nn.init.zeros_(self.score.weight);nn.init.zeros_(self.score.bias);nn.init.zeros_(self.semantic_align.weight)
  with torch.no_grad():self.output.weight[:,:self.hidden].zero_()
 def refine(self,image,pyramid,c):
  q=self.rgb(F.interpolate(image,pyramid[0].shape[-2:],mode='bilinear',align_corners=False))
  c=c.to(q.dtype)
  gsem=F.interpolate(c,q.shape[-2:],mode='bilinear',align_corners=False)
  aligned=F.interpolate(self.semantic_align(c),q.shape[-2:],mode='bilinear',align_corners=False)
  s0=self.score(torch.cat([gsem,q],1))
  with torch.autocast(device_type=q.device.type,enabled=False):
   sa=(aligned.float()*q.float()).sum(1,keepdim=True)/math.sqrt(self.hidden)
   gate=(s0.float()+sa).tanh()
   fused=torch.cat([q.float(),gate*gsem.float()],1)
  delta=self.output(fused.to(q.dtype))
  self.last_diagnostics=dict(gate_mean=gate.detach().mean(),gate_abs_mean=gate.detach().abs().mean(),sag_score_rms=sa.detach().square().mean().sqrt(),update_rms=delta.detach().float().square().mean().sqrt())
  return (pyramid[0]+delta,*pyramid[1:])

class DirectF12Relay(direct.DirectSemanticRelay):
 def __init__(self,channels=384,hidden=32):
  super().__init__(channels,hidden)
  self.prune()
 def prune(self):
  object.__setattr__(self,'_full_semantic',self.semantic_proj.weight.detach().cpu().clone())
  with torch.random.fork_rng(devices=[]):new=nn.Conv2d(self.channels,self.hidden,1)
  with torch.no_grad():
   new.weight.copy_(self.semantic_proj.weight[:,-self.channels:]);new.bias.copy_(self.semantic_proj.bias)
  self.semantic_proj=new
 def forward(self,image,features,neck):
  if len(features)!=4 or any(f.shape!=features[0].shape for f in features):raise ValueError('Four aligned decoder taps required')
  pyramid=neck(tuple(features))
  c=self.semantic_proj(direct.channel_norm(features[-1]))
  with torch.autocast(device_type=c.device.type,enabled=False):c=self.semantic_norm(c.float().movedim(1,-1)).movedim(-1,1)
  return self.refine(image,pyramid,c)

@MODELS.register_module()
class DirectNoFineBackbone(direct.SemDiRDirectBackbone):
 def __init__(self,**kwargs):
  super().__init__(**kwargs)
  self.relay.__class__=DirectNoFineRelay;self.relay.prune();self.method='semdir_direct_no_fine'

@MODELS.register_module()
class DirectF12Backbone(direct.SemDiRDirectBackbone):
 def __init__(self,**kwargs):
  super().__init__(**kwargs)
  self.relay.__class__=DirectF12Relay;self.relay.prune();self.method='semdir_direct_f12'

@MODELS.register_module()
class DirectNoFineSegmentor(direct.DirectRegionPreservingSegmentor):
 backbone_type=DirectNoFineBackbone

@MODELS.register_module()
class DirectF12Segmentor(direct.DirectRegionPreservingSegmentor):
 backbone_type=DirectF12Backbone

def canonical_full_digest(model):
 state=dict(model.state_dict());r=model.backbone.relay
 if isinstance(r,DirectNoFineRelay):
  key='backbone.relay.output.weight'
  if not torch.equal(state[key].detach().cpu(),r._full_output[:,:2*r.hidden]):raise RuntimeError('Retained output initialization changed')
  state[key]=r._full_output;state['backbone.relay.gamma']=torch.zeros(())
 elif isinstance(r,DirectF12Relay):
  key='backbone.relay.semantic_proj.weight'
  if not torch.equal(state[key].detach().cpu(),r._full_semantic[:,-r.channels:]):raise RuntimeError('Retained F12 initialization changed')
  state[key]=r._full_semantic
 digest=hashlib.sha256()
 for key,value in sorted(state.items()):
  digest.update(key.encode());digest.update(str((tuple(value.shape),value.dtype)).encode());digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
 return digest.hexdigest()

@HOOKS.register_module()
class DirectSecondAuditHook(direct.DirectAuditHook):
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
                comparison_scope='All retained tensors match Full; removed projection slices are restored for comparison only',
                full_config_sha256=runner.cfg.paper.direct_ablation.full_config_sha256))

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if runner.iter + 1 == 200:
            relay = unwrap(runner.model).backbone.relay
            values = dict(H=relay.output.weight[:, :relay.hidden], score=relay.score.weight)
            if hasattr(relay, 'gamma'):
                values['gamma'] = relay.gamma
            if hasattr(relay, 'semantic_align'):
                values['alignment'] = relay.semantic_align.weight
            learned = {k: float(v.detach().float().abs().sum()) for k,v in values.items()}
            if any(not bool(torch.isfinite(v).all()) for v in values.values()) or any(v <= 0 for v in learned.values()):
                raise RuntimeError('An active zero-initialized ablation path did not learn')
        # Skip the Full hook's mandatory Wa assertion for physically pruned A1.
        PaperAuditHook.after_train_iter(self, runner, batch_idx, data_batch, outputs)
        if runner.iter + 1 == 200 and get_rank() == 0:
            rsp = 'loss_rsp' in (outputs or {})
            expected_rsp = True
            if rsp != expected_rsp:
                raise RuntimeError('Ablation RSP execution does not match its definition')
            root = Path(runner.work_dir).parents[1] if runner.cfg.paper.smoke else Path(runner.work_dir)
            write_once(root / 'ablation_learning_probe.json', dict(iteration=200,
                variant=runner.cfg.paper.direct_ablation.variant, active_projection_l1=learned,
                RSP_loss_present=rsp, all_active_paths_learned=True,
                measurement='engineering smoke' if runner.cfg.paper.smoke else 'formal training'))


for name in VARIANTS:
 if name in EXPERIMENTS:raise RuntimeError('Duplicate second ablation')
 EXPERIMENTS[name]=first.DirectScreenExperiment(name,'m23_plain',backbone='deit_s',comparison='',seed_required=1)
 experiments.EXPERIMENTS[name]=EXPERIMENTS[name]

_parent_build = config._build


def build(*args, **kwargs):
    cfg = _parent_build(*args, **kwargs)
    name = cfg['paper']['experiment']
    if name not in VARIANTS:
        return cfg
    if cfg['train_cfg']['max_iters'] != 20000:
        raise ValueError('A2/A4 are independent first20K screenings')
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
    backbone.update(type='DirectNoFineBackbone' if name==A2_ID else 'DirectF12Backbone', direct_hidden=32)
    cfg['model']['type'] = 'DirectNoFineSegmentor' if name==A2_ID else 'DirectF12Segmentor'
    cfg['model']['region_preservation'] = deepcopy(direct.registration.RSP)
    cfg['custom_imports']['imports'].append('competitors.direct_second')
    cfg['custom_hooks'] = [dict(type='DirectSecondAuditHook')]
    initial = json.loads(FULL_INITIALIZATION.read_text())
    cfg['paper']['direct_semantics'] = deepcopy(reference['paper']['direct_semantics'])
    cfg['paper']['direct_semantics'].update(SAG=True, original_RSP=True, fine_value=name!=A2_ID, semantic_taps=[12] if name==A4_ID else [3,6,9,12])
    cfg['paper']['direct_semantics'].update(K=1 if name==A4_ID else 4, semantic_projection=[384 if name==A4_ID else 1536,32], output_projection=[64 if name==A2_ID else 96,384])
    cfg['paper']['direct_ablation'] = dict(variant=VARIANTS[name], reference=direct.ID,
        full_config_sha256=sha256(FULL_CONFIG), full_initialization_sha256=initial['direct_state_sha256'],
        full_initialization_file_sha256=sha256(FULL_INITIALIZATION),
        independent_ImageNet_start=True, schedule_end=160000, training_stop_step=20000,
        only_change='remove Vf projection and gamma only' if name==A2_ID else 'F12-only semantic conditioning; preserve all four decoder taps')
    cfg['paper']['provenance_label'] = 'New beta-free Direct single-factor ablation, not historical Exact/G1; first20K of unchanged160K schedule'
    return cfg


config._build = build
