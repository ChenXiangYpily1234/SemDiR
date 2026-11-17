"""Process-local registry extension: legacy studies and active jobs are untouched."""
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import json,os
import torch
from mmengine.registry import HOOKS
from mmengine.dist import get_rank,get_world_size
from research.ad_pcpf_v2 import config,experiments,run,paper_matrix
from research.ad_pcpf_v2.artifacts import sha256,write_once,shared_digest,identity,load
from research.ad_pcpf_v2.engine import PaperAuditHook,unwrap
from research.ad_pcpf_v2.runtime import resolve_runtime,runtime_from_config
from .registry import EXPERIMENTS,CompetitorExperiment,PROTOCOL,ROOT,BUDGET,BLOCKERS,default_weight
from . import models,losses,semdir_exact
from . import hierarchical_competitors

_original_build=config._build
_original_sources=run.sources
_installed=False
RSP=dict(enabled=True,margin=.5,max_weight=.05,warmup_iters=10000,boundary_radius=3,small_area_fraction=.005)


def build(*args,**kwargs):
    cfg=_original_build(*args,**kwargs)
    spec=experiments.get_experiment(cfg['paper']['experiment'])
    if not isinstance(spec,CompetitorExperiment):return cfg
    if spec.id=='M31':raise ValueError(BLOCKERS[spec.id])
    if cfg['paper']['runtime']['seed']!=spec.seed_required:
        raise ValueError(f'{spec.id} requires seed {spec.seed_required}; use a new registered ID for another seed')
    checkpoint=cfg['model']['backbone']['pretrained_checkpoint'];digest=cfg['paper']['pretrained_sha256']
    cfg['custom_imports']['imports'].append('competitors.registration')
    cfg['custom_hooks']=[dict(type='CompetitorAuditHook')]
    cfg['paper']['initialization_provenance']=next((v for v in json.loads((ROOT/'competitors/weights.json').read_text()).values() if v['sha256']==digest),dict(sha256=digest,origin='user supplied; pretraining recipe audit required'))
    cfg['paper'].update(competition=True,comparison_reference=spec.comparison,
        method=spec.method,recipe=spec.recipe,budget=BUDGET,
        provenance_label='author-architecture / DCPA adapted training' if spec.method in ('vit_comer','vitcontroller') else 'M23 CoFiRT-RSP stage-aligned adaptation' if spec.method=='m23_stage_aligned' else 'DCPA matched' if spec.recipe=='matched' else 'author frozen readout / DCPA batch budget')
    common=dict(pretrained_checkpoint=checkpoint,pretrained_sha256=digest)
    channels=[384]*4
    deit=dict(model_name='deit_small_patch16_224.fb_in1k',image_size=512,drop_path_rate=.1,**common)
    if spec.method.startswith('adapted_'):
        adaptation=spec.method.removeprefix('adapted_')
        if spec.backbone=='deit_s':
            backbone=dict(type='AdaptedLocAtDeiT',**deit)
        else:
            backbone=dict(type='AdaptedHierarchicalCompetitor',architecture=spec.backbone,
                adaptation=adaptation,**common)
            channels=[96,192,384,768] if spec.backbone=='swin_t' else [32,64,160,256]
        cfg['paper']['provenance_label']='DCPA local cross-backbone adaptation; not author-score reproduction'
        if adaptation == 'fdam':
            backbone['fdam_order'] = 'pre_drop'
            cfg['paper']['fdam_frequency_order'] = 'pre_drop'
            cfg['paper']['implementation_revision'] = 'FDAM-PREDROP-R2-20261111'
        if adaptation=='locat':
            cfg['paper']['pretraining_information']='Vanilla ImageNet backbone initialization + downstream GAug; no GAug/PRR classification pretraining; inference before PRR as in author segmentation.'
    elif spec.method in ('semdir_clean','semdir_shared','semdir_lite','m23','m23_no_rsp','m23_plain','m23_plain_no_rsp',
                          'm23_plain_no_coarse','m23_plain_no_rgb'):
        opts=experiments.relay_options(experiments.EXPERIMENTS['M23'])
        plain = spec.method.startswith('m23_plain') or spec.method.startswith('semdir_')
        if spec.method.startswith('semdir_'):
            opts.update(use_projection=False,use_attenuation=False,context_cache=True,
                        shared_context=spec.method!='semdir_clean',
                        inject_pyramid=spec.method!='semdir_lite')
            cfg['paper']['simplification']=dict(candidate=spec.id,reference=spec.comparison,
                purpose='independent staged screening; not established performance',
                shared_condition='four raw semantic tap mean',
                original_sigmoid_depth_weighting_and_four_projections_retained=True)
        conditioning = {'m23_plain_no_coarse':'no_coarse',
                        'm23_plain_no_rgb':'no_rgb'}.get(spec.method,'both')
        exact_full = plain and conditioning == 'both' and not spec.method.startswith('semdir_')
        backbone=dict(type='SemDiRExactBackbone' if exact_full else 'CompetitorM23Backbone',relay_options=opts,
            fusion='plain' if plain else 'cfrt',**deit)
        if exact_full:
            cfg['custom_imports']['imports'].append('competitors.semdir_exact')
        if conditioning != 'both':
            backbone['conditioning']=conditioning
        backbone['image_size']=spec.crop
        cfg['paper']['provenance_label']='M23 controlled variant; ImageNet initialization; no teacher; SI=0'
        if spec.id in ('M23-PlainFusion-NoRSP','M23-PlainFusion-NoCoarse','M23-PlainFusion-NoRGB'):
            cfg['paper']['plain_fusion_ablation']=dict(
                reference='M23-PlainFusion',conditioning=conditioning,
                rsp=spec.method!='m23_plain_no_rsp',
                original_r9_and_rgb_paths_preserved=True,
                parameter_matching='identical' if conditioning=='both' else
                    'removed conditioner parameters; report independently measured Params/FLOPs')
    elif spec.method=='fdam':
        # Reuse the measured M04 AttInv + attention/MLP frequency implementation.
        backbone=dict(type='ADPCPFPaperBackbone',method='fdam',relay_options=None,
                      **dict(deit,image_size=spec.crop))
        cfg['paper']['budget']=spec.steps
        cfg['paper']['architecture_reference']='M04'
    elif spec.method=='vitcontroller':backbone=dict(type='CompetitorViTController',**deit)
    elif spec.method=='vit_comer':backbone=dict(type='CompetitorCoMer',drop_path_rate=.1,**common)
    elif spec.backbone in ('swin_t','mit_b0'):
        local=spec.method in ('m21_adapted','m23_stage_aligned','m23_plain_stage_aligned')
        transport=spec.method in ('m23_stage_aligned','m23_plain_stage_aligned')
        backbone=dict(type='CompetitorHierarchical',architecture=spec.backbone,
                      local_relay=local,**common)
        if transport:
            backbone['transport']=True
        if spec.method=='m23_plain_stage_aligned':
            backbone['fusion']='plain'
            cfg['paper']['provenance_label']='PlainFusion stage-aligned hierarchical adaptation; native four-stage outputs; all block local conditions projected to common stride16 grid; no CFRT patch softmax'
            cfg['paper']['architecture_reference']='M23-PlainFusion'
        channels=[96,192,384,768] if spec.backbone=='swin_t' else [32,64,160,256]
        if spec.method=='m23_stage_aligned':
            cfg['paper']['provenance_label']='M23 CoFiRT-RSP stage-aligned cross-backbone adaptation'
        elif local:cfg['paper']['provenance_label']='new stage-aligned M21 adaptation (not original R9)'
    elif spec.method=='locatvit':
        arch='swin' if spec.backbone=='locat_swin' else 'vit'
        size='small' if spec.backbone=='locat_small' else 'tiny'
        backbone=dict(type='CompetitorLocAt',architecture=arch,size=size,
            frozen=spec.recipe=='native_probe',linear_probe=spec.recipe=='native_probe',**common)
        channels=[384]*4 if size=='small' else [192]*4
        if arch=='swin':channels=[96,192,384,768]
    elif spec.method=='vanilla_probe':backbone=dict(type='CompetitorPlainProbe',**common);channels=[192]*4
    else:
        method='adpcpf_r9' if spec.method in ('m21','r9_region_ce') else 'vanilla'
        opts=experiments.relay_options(experiments.EXPERIMENTS['M21']) if method=='adpcpf_r9' else None
        backbone=dict(type='ADPCPFPaperBackbone',method=method,relay_options=opts,
                      **dict(deit,image_size=spec.crop))
    cfg['model']['backbone']=backbone
    cfg['model']['decode_head']['in_channels']=channels
    cfg['model']['auxiliary_head']['in_channels']=channels[2]
    if spec.method in ('m23_plain_stage_aligned','semdir_clean','semdir_shared','semdir_lite','m21_adapted','m21','vanilla_rsp','r9_region_ce','m23_stage_aligned','m23','m23_plain',
                       'm23_plain_no_coarse','m23_plain_no_rgb'):
        cfg['model']['type']='CompetitorVanillaRSP' if spec.method=='vanilla_rsp' else 'CompetitorRegionCE' if spec.method=='r9_region_ce' else 'ADPCPFRegionPreservingSegmentor'
        cfg['model']['region_preservation']=deepcopy(RSP)
    if spec.recipe=='native_probe':
        cfg['model']['decode_head']=dict(type='CompetitorLinearHead',in_channels=channels[-1],
            channels=512,in_index=0,num_classes=150,dropout_ratio=0.,norm_cfg=dict(type='GN',num_groups=1),
            act_cfg=dict(type='GELU'),align_corners=False,
            loss_decode=dict(type='CrossEntropyLoss',loss_weight=1.,avg_non_ignore=True))
        cfg['model'].pop('auxiliary_head');cfg['model']['test_cfg']=dict(mode='whole')
        cfg['model']['data_preprocessor'].update(mean=[127.5]*3,std=[127.5]*3)
        cfg['optim_wrapper']['optimizer']['lr']=.001
        cfg['param_scheduler']=[dict(type='OneCycleLR',eta_max=.001,total_steps=20000,pct_start=.25,
            anneal_strategy='linear',div_factor=1e4,final_div_factor=1e4,by_epoch=False)]
        cfg['paper']['pretraining_information']='GAug+PRR belong to classification pretraining; segmentation extracts pre-PRR intermediates as in author code.'
    return cfg


def sources():
    data=_original_sources()
    for p in sorted((ROOT/'competitors').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.json','.md') or p.is_file() and p.name=='LICENSE':
            data['competitors/'+str(p.relative_to(ROOT/'competitors'))]=sha256(p)
    return data


def prepare(args,output):
    from mmengine.config import Config
    if (output/'study.json').exists():
        manifest=run.verify(output);run.check_runtime_args(args,manifest)
        if args.experiments and experiments.expand(args.experiments)!=[x['id'] for x in manifest['experiments']]:
            raise ValueError('Cannot extend a sealed study')
        return manifest
    if any(p.name!='.lock' for p in output.iterdir()):raise FileExistsError('Preserve nonempty unregistered directory')
    runtime=resolve_runtime(args.world_size,args.per_gpu,args.global_batch,args.seed)
    names=experiments.expand(args.experiments or ['competition_core'])
    datasets={};pretrained={};configs=[]
    for name in names:
        spec=EXPERIMENTS[name]
        if name=='M31':raise ValueError(BLOCKERS[name])
        if os.environ.get('DCPA_COMP_INIT'):
            if len(names)!=1:raise ValueError('Explicit initialization requires one experiment per study')
            path=Path(os.environ['DCPA_COMP_INIT']).resolve();digest=os.environ['DCPA_COMP_INIT_SHA']
        else:path,digest=default_weight(spec)
        if not path.is_file() or sha256(path)!=digest:raise ValueError('Missing or changed pretrained bytes')
        key=spec.backbone
        if key in pretrained and pretrained[key]['sha256']!=digest:raise ValueError('Pretraining identities differ in study')
        pretrained[key]=dict(path=str(path),sha256=digest)
        if spec.dataset not in datasets:
            root=args.ade_root.resolve() if spec.dataset=='ade20k' else args.context_root.resolve()
            datasets[spec.dataset]=dict(root=str(root),index=run.dataset_index(root,spec.dataset))
        file=output/'configs'/(name+'.py');work=output/'runs'/name
        cfg=config.make_config(name,datasets[spec.dataset]['root'],path,work,digest,**runtime)
        config.validate_config(cfg)
        configs.append((spec,file,cfg))
    rows=[]
    for spec,file,cfg in configs:
        file.parent.mkdir(parents=True,exist_ok=True);Config(cfg).dump(str(file))
        rows.append(dict(id=spec.id,spec=asdict(spec),steps=spec.steps,config=str(file),sha256=sha256(file)))
    manifest=dict(protocol=PROTOCOL,**runtime,small_ce_normalization='global_eligible_images',
        experiments=rows,datasets=datasets,pretrained=pretrained,sources=sources(),environment=run.environment())
    write_once(output/'study.json',manifest)
    for p in [output/'study.json',*(file for _,file,_ in configs)]:p.chmod(0o444)
    return manifest


@HOOKS.register_module()
class CompetitorAuditHook(PaperAuditHook):
    def before_train(self,runner):
        runtime=runtime_from_config(runner.cfg)
        if get_world_size()!=runtime['world_size'] or runner.cfg.train_dataloader.batch_size!=runtime['per_gpu']:
            raise RuntimeError('Competitor DDP layout mismatch')
        model=unwrap(runner.model);self.handles=[];self.seen=set()
        self.finite=torch.ones((),dtype=torch.bool,device=next(model.parameters()).device)
        if not runner.cfg.resume:
            digest=shared_digest(model)
            model.backbone.check_zero_initialization(torch.zeros(1,3,65,79,device=self.finite.device))
            if get_rank()==0:write_once(Path(runner.work_dir)/'initialization.json',
                dict(**identity(runner.cfg,runner.cfg.paper_config_sha256),shared_state_sha256=digest,
                     zero_residual_check='applicable_only_to_M21_extensions',
                     pretraining_sha256=runner.cfg.paper.pretrained_sha256))
        if runner.iter<200:
            self.expected={n for n,p in model.named_parameters() if p.requires_grad}
            def record(name):
                def hook(g):self.seen.add(name);self.finite.logical_and_(torch.isfinite(g).all());return g
                return hook
            for n,p in model.named_parameters():
                if p.requires_grad:self.handles.append(p.register_hook(record(n)))


def render_tables(manifest,scores,multi_scores,costs,evaluations):
    lines=['# Independent competitors','',f'Protocol: {PROTOCOL}',
        '|ID|Method|Backbone|Recipe|SS mIoU|MS mIoU|','|---|---|---|---|---:|---:|']
    for row in manifest['experiments']:
        e=EXPERIMENTS[row['id']]
        lines.append(f'|{e.id}|{e.method}|{e.backbone}|{e.recipe}|{scores.get(e.id, "pending")}|{multi_scores.get(e.id,"pending")}|')
    return '\n'.join(lines)+'\n'


def install():
    global _installed
    if _installed:return
    experiments.EXPERIMENTS.update(EXPERIMENTS)
    experiments.GROUPS.update(competition_core=('M24','M25','M26'),competition_hierarchical=('M27','M28','M29','M30'))
    from .registry import MAIN_MISSING_TRAIN,PLAIN_ABLATIONS
    experiments.GROUPS['main_missing']=MAIN_MISSING_TRAIN
    experiments.GROUPS.update(plain_ablations=PLAIN_ABLATIONS,
        plain_primary=('M23-PlainFusion',),plain_p0=('M23-PlainFusion-NoRSP',),
        semdir_screen=('M23-PlainFusion','SemDiR-Clean','SemDiR-Clean-Shared'))
    config._build=build;run.prepare=prepare;run.PROTOCOL=PROTOCOL;run.sources=sources
    run.DEFAULT_OUTPUT=ROOT.parent/'trained_models'/f'COMPETITORS-R1-{BUDGET}'
    paper_matrix.render_tables=render_tables
    _installed=True
install()
