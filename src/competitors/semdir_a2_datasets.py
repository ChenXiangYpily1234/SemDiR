"""Current main A2: unchanged Q+gG refinement; independent target80K."""
from copy import deepcopy
from pathlib import Path
import json,os
from mmengine.config import Config
from mmengine.registry import HOOKS
from mmengine.dist import get_rank
from research.ad_pcpf_v2 import config,experiments
from research.ad_pcpf_v2.artifacts import sha256,write_once
from research.ad_pcpf_v2.engine import unwrap
from . import direct_second as a2
from . import semdir_direct as direct
from . import semdir_direct_generalization
from . import semdir_direct_coco
from . import generalization_datasets as target
from .registry import EXPERIMENTS

BASE=Path(__file__).resolve().parents[2]
IDS=dict(context59='SemDiR-A2-Context59-S1',cocostuff10k='SemDiR-A2-COCO10K-S1')
REFERENCES={
 'context59':('SemDir-Direct-Context59-80K-w4-b4-g16-s1-20261114','SemDiR-Direct-Context59-S1'),
 'cocostuff10k':('SemDir-Direct-COCO10K-80K-w4-b4-g16-s1-20261114','SemDiR-Direct-COCO10K-S1')}
def reference_config(which):
 root,name=REFERENCES[which];return BASE/'trained_models'/root/'configs'/f'{name}.py'
def reference_initialization(which):
 root,name=REFERENCES[which];return BASE/'trained_models'/root/'runs'/name/'direct_initialization.json'

def which_dataset(name):return next(k for k,v in IDS.items() if v==name)

@HOOKS.register_module()
class A2DatasetAuditHook(direct.DirectAuditHook):
    def before_train(self,runner):
        model=unwrap(runner.model)
        if not runner.cfg.resume:
            which=which_dataset(runner.cfg.paper.experiment);reference=runner.cfg.paper.a2_dataset
            if sha256(reference_initialization(which))!=reference.reference_initialization_file_sha256:raise RuntimeError('Target Full initialization proof changed')
            common=a2.canonical_full_digest(model)
            if common!=reference.full_initialization_sha256:raise RuntimeError('Target A2 retained initialization differs from corresponding Full')
        super().before_train(runner)
        if not runner.cfg.resume and get_rank()==0:
            write_once(Path(runner.work_dir)/'a2_dataset_initialization.json',dict(
                full_comparison_sha256=common,actual_state_sha256=direct.state_digest(model),
                dataset=which,architecture='A2 unchanged: P0+Wout[Q,gG]',independent_ImageNet_start=True))

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        # This existing A2-compatible check requires H, score, alignment and RSP,
        # but never requires the physically removed gamma/fine-value parameters.
        a2.DirectSecondAuditHook.after_train_iter(self,runner,batch_idx,data_batch,outputs)

for dataset,name in IDS.items():
 if name in EXPERIMENTS:raise RuntimeError('A2 target experiment already registered')
 spec=target.TargetDatasetExperiment(name,'m23_plain',dataset=dataset,backbone='deit_s',comparison='',seed_required=1)
 EXPERIMENTS[name]=spec;experiments.EXPERIMENTS[name]=spec;target.EXPERIMENTS[name]=spec
_parent_build=config._build

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs);name=cfg['paper']['experiment']
    if name not in IDS.values():return cfg
    which=which_dataset(name);ref=Config.fromfile(str(reference_config(which)),import_custom_modules=False).to_dict()
    for key in ['optim_wrapper','param_scheduler','train_dataloader','val_dataloader','test_dataloader','train_cfg','val_evaluator','test_evaluator']:
        if cfg[key]!=ref[key]:raise ValueError('A2 target changes locked recipe: '+key)
    for key in ['decode_head','auxiliary_head','data_preprocessor','test_cfg']:
        if cfg['model'][key]!=ref['model'][key]:raise ValueError('A2 target changes shared decoder: '+key)
    if cfg['train_cfg']['max_iters']!=80000 or cfg['param_scheduler'][-1]['end']!=80000:raise ValueError('Targets require independent full80K')
    if cfg['load_from'] is not None or os.environ.get('DCPA_COMP_INIT'):raise ValueError('Target training must start from ImageNet, never ADE segmentation')
    bb=cfg['model']['backbone']
    for k in ['method','fusion','conditioning','relay_options']:bb.pop(k,None)
    bb.update(type='DirectNoFineBackbone',direct_hidden=32)
    cfg['model'].update(type='DirectNoFineSegmentor',region_preservation=deepcopy(direct.registration.RSP))
    cfg['custom_imports']['imports'].append('competitors.semdir_a2_datasets');cfg['custom_hooks']=[dict(type='A2DatasetAuditHook')]
    init=json.loads(reference_initialization(which).read_text())
    cfg['paper']['direct_semantics']=dict(D=384,d=32,K=4,semantic_projection=[1536,32],output_projection=[64,384],
        beta=False,cross_layer_difference=False,complementary_routing=False,direct_injection=False,
        SAG=True,semantic_alignment_bias=False,original_RSP=True,boundary_RSP=False,
        fine_value=False,gamma=False,local_deviation=False,ungated_RGB_update=True,
        formula='P0+Wout[Q,gG]',identity_initialization=True,training_schedule_end=80000)
    cfg['paper']['a2_dataset']=dict(dataset=which,reference=REFERENCES[which][1],
        reference_config_sha256=sha256(reference_config(which)),reference_initialization_file_sha256=sha256(reference_initialization(which)),
        full_initialization_sha256=init['direct_state_sha256'],independent_ImageNet_start=True,stop_step=80000,schedule_end=80000,module_parameters=75041)
    cfg['paper']['direct_ablation']=dict(variant='A2_'+which)
    cfg['paper'].update(method='semdir_a2',budget=80000,architecture_reference='SemDiR-A2',
        provenance_label='Current A2 main, unchanged64-channelQ+gG fusion; independent target80K, not old Full/Exact/SAGR scores')
    return cfg
config._build=build
