"""Current A2 on CamVid11; independent ImageNet80K, held-out test233."""
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
import json,os
from mmengine.config import Config
from mmengine.registry import HOOKS
from mmengine.dist import get_rank
from research.ad_pcpf_v2 import config,experiments,run
from research.ad_pcpf_v2.artifacts import sha256,write_once
from research.ad_pcpf_v2.runtime import resolve_runtime
from research.ad_pcpf_v2.engine import unwrap
from . import direct_second as a2
from . import semdir_direct as direct
from . import semdir_direct_camvid as camvid
from .registry import EXPERIMENTS,ROOT,PROTOCOL,default_weight

ID='SemDiR-A2-CamVid-S1'
REFERENCE_STUDY=ROOT.parent/'trained_models/SemDir-Direct-CamVid-80K-w4-b4-g16-s1-20261115'
REFERENCE_CONFIG=REFERENCE_STUDY/'configs/SemDiR-Direct-CamVid-S1.py'
REFERENCE_INIT=REFERENCE_STUDY/'runs/SemDiR-Direct-CamVid-S1/direct_initialization.json'
RECIPE_KEYS=['optim_wrapper','param_scheduler','train_dataloader','val_dataloader','test_dataloader','train_cfg','val_evaluator','test_evaluator']
if ID in EXPERIMENTS:raise RuntimeError('A2 CamVid already registered')
EXPERIMENTS[ID]=camvid.CamVidExperiment(ID,'semdir_direct',dataset='camvid11',backbone='deit_s',comparison='',seed_required=1)
experiments.EXPERIMENTS[ID]=EXPERIMENTS[ID]
_parent_build=config._build;_parent_prepare=run.prepare

@HOOKS.register_module()
class A2CamVidAuditHook(direct.DirectAuditHook):
    def before_train(self,runner):
        model=unwrap(runner.model);camvid.assert_test_protocol(runner.cfg)
        if not runner.cfg.resume:
            ref=runner.cfg.paper.a2_camvid
            if sha256(REFERENCE_INIT)!=ref.reference_initialization_file_sha256:raise RuntimeError('CamVid initialization reference changed')
            common=a2.canonical_full_digest(model)
            if common!=ref.full_initialization_sha256:raise RuntimeError('A2 CamVid retained initialization differs from Full')
        super().before_train(runner)
        if not runner.cfg.resume and get_rank()==0:
            write_once(Path(runner.work_dir)/'a2_camvid_initialization.json',dict(full_comparison_sha256=common,
                actual_state_sha256=direct.state_digest(model),independent_ImageNet_start=True,
                architecture='A2 unchanged: P0+Wout[Q,gG]',final_split='test233; distinct from val101',Sky0_valid=True))

    def after_train_iter(self,runner,batch_idx,data_batch=None,outputs=None):
        a2.DirectSecondAuditHook.after_train_iter(self,runner,batch_idx,data_batch,outputs)

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs)
    if cfg['paper']['experiment']!=ID:return cfg
    ref=Config.fromfile(str(REFERENCE_CONFIG),import_custom_modules=False).to_dict()
    # Reuse exactly the verified CamVid dataset transformation (Sky=0).
    for key in ['train_dataloader','val_dataloader','test_dataloader']:
        old=cfg[key]['dataset'];pipe=old['pipeline']
        for item in pipe:
            if item['type']=='LoadAnnotations':item['reduce_zero_label']=False
            if item['type'] in ['RandomResize','Resize']:item['scale']=(2048,512)
        split={'train_dataloader':'train','val_dataloader':'val','test_dataloader':'test'}[key]
        cfg[key]['dataset']=dict(type='SemDirCamVid11Dataset',data_root=old['data_root'],
            img_suffix='.png',seg_map_suffix='_L.png',reduce_zero_label=False,ignore_index=255,
            data_prefix=dict(img_path='CamVid_RGB',seg_map_path='CamVidGray'),ann_file=f'splits/{split}.txt',pipeline=pipe)
    for key in RECIPE_KEYS:
        if cfg[key]!=ref[key]:raise ValueError('A2 CamVid changes locked recipe: '+key)
    for key in ['decode_head','auxiliary_head','data_preprocessor','test_cfg']:
        if cfg['model'][key]!=ref['model'][key]:raise ValueError('A2 CamVid changes decoder: '+key)
    if cfg['load_from'] is not None or os.environ.get('DCPA_COMP_INIT'):raise ValueError('Independent ImageNet start only')
    if cfg['train_cfg']['max_iters']!=80000 or cfg['param_scheduler'][-1]['end']!=80000:raise ValueError('Independent80K required')
    bb=cfg['model']['backbone']
    for key in ['method','fusion','conditioning','relay_options']:bb.pop(key,None)
    bb.update(type='DirectNoFineBackbone',direct_hidden=32,image_size=512)
    cfg['model'].update(type='DirectNoFineSegmentor',region_preservation=deepcopy(direct.registration.RSP))
    cfg['custom_imports']['imports'].append('competitors.semdir_a2_camvid');cfg['custom_hooks']=[dict(type='A2CamVidAuditHook')]
    init=json.loads(REFERENCE_INIT.read_text())
    cfg['paper'].update(method='semdir_a2',budget=80000,architecture_reference='SemDiR-A2',
        provenance_label='Current A2, no Vf; independent CamVid11 ImageNet80K; final test233 distinct from val101',
        label_mapping=deepcopy(ref['paper']['label_mapping']),evaluation_protocol=deepcopy(ref['paper']['evaluation_protocol']),
        direct_semantics=dict(D=384,d=32,K=4,semantic_projection=[1536,32],output_projection=[64,384],
            beta=False,cross_layer_difference=False,complementary_routing=False,direct_injection=False,
            SAG=True,semantic_alignment_bias=False,original_RSP=True,boundary_RSP=False,
            fine_value=False,gamma=False,local_deviation=False,ungated_RGB_update=True,
            formula='P0+Wout[Q,gG]',identity_initialization=True,training_schedule_end=80000),
        a2_camvid=dict(reference_config_sha256=sha256(REFERENCE_CONFIG),reference_initialization_file_sha256=sha256(REFERENCE_INIT),
            full_initialization_sha256=init['direct_state_sha256'],independent_ImageNet_start=True,module_parameters=75041),
        direct_ablation=dict(variant='A2_camvid11'))
    camvid.assert_test_protocol(cfg)
    return cfg

def prepare(args,output):
    names=experiments.expand(args.experiments or [])
    if ID not in names or (output/'study.json').exists():return _parent_prepare(args,output)
    if names!=[ID]:raise ValueError('A2 CamVid requires an independent study')
    if any(p.name!='.lock' for p in output.iterdir()):raise FileExistsError('Nonempty unregistered output')
    if os.environ.get('DCPA_COMP_INIT'):raise ValueError('Independent ImageNet start only')
    runtime=resolve_runtime(args.world_size,args.per_gpu,args.global_batch,args.seed);spec=EXPERIMENTS[ID];weight,digest=default_weight(spec)
    if not weight.is_file() or sha256(weight)!=digest:raise ValueError('ImageNet weight missing or changed')
    root=Path(os.environ.get('DCPA_CAMVID_ROOT',str(camvid.DEFAULT_ROOT))).resolve();index=camvid.camvid_index(root)
    path=output/'configs'/f'{ID}.py';cfg=config.make_config(ID,root,weight,output/'runs'/ID,digest,**runtime);config.validate_config(cfg)
    path.parent.mkdir(parents=True,exist_ok=True);Config(cfg).dump(str(path))
    manifest=dict(protocol=PROTOCOL,**runtime,small_ce_normalization='global_eligible_images',
        experiments=[dict(id=ID,spec=asdict(spec),steps=80000,config=str(path),sha256=sha256(path))],
        datasets=dict(camvid11=dict(root=str(root),index=index)),pretrained=dict(deit_s=dict(path=str(weight),sha256=digest)),
        sources=run.sources(),environment=run.environment())
    write_once(output/'study.json',manifest)
    for p in [path,output/'study.json']:p.chmod(0o444)
    return manifest

config._build=build;run.prepare=prepare
