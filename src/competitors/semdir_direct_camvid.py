"""CamVid11: NEW beta-free Direct; independent80K; val101/test233 separated."""
from pathlib import Path
from dataclasses import dataclass,asdict
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import argparse,os,json,hashlib
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from research.ad_pcpf_v2 import config,experiments,run
from research.ad_pcpf_v2.artifacts import sha256,write_once
from research.ad_pcpf_v2.runtime import resolve_runtime
from . import semdir_direct as direct
from .registry import EXPERIMENTS,CompetitorExperiment,ROOT,PROTOCOL,default_weight

DEFAULT_ROOT=ROOT.parent/'datasets/CamVid'
CLASSES=('Sky','Building','Pole','Road','Sidewalk','Tree','SignSymbol','Fence','Car','Pedestrian','Bicyclist')
PALETTE=[[128,128,128],[128,0,0],[192,192,128],[128,64,128],[0,0,192],[128,128,0],[192,128,128],[64,64,128],[64,0,128],[64,64,0],[0,128,192]]
COUNTS=dict(train=367,val=101,test=233)

@DATASETS.register_module()
class SemDirCamVid11Dataset(BaseSegDataset):
    METAINFO=dict(classes=CLASSES,palette=PALETTE)
    def __init__(self,**kwargs):
        if kwargs.get('reduce_zero_label',False) is not False:raise ValueError('CamVid0 is Sky; never reduce zero')
        kwargs.setdefault('img_suffix','.png');kwargs.setdefault('seg_map_suffix','_L.png')
        super().__init__(**kwargs)

@dataclass(frozen=True)
class CamVidExperiment(CompetitorExperiment):
    @property
    def steps(self):return 80000
    @property
    def crop(self):return 512
    @property
    def classes(self):return 11

ID='SemDiR-Direct-CamVid-S1'
EXPERIMENTS[ID]=CamVidExperiment(ID,'semdir_direct',dataset='camvid11',backbone='deit_s',comparison='',seed_required=1)
experiments.EXPERIMENTS[ID]=EXPERIMENTS[ID]
_parent_build=config._build;_parent_prepare=run.prepare;_parent_index=run.dataset_index

def build(*args,**kwargs):
    cfg=_parent_build(*args,**kwargs)
    if cfg['paper']['experiment']!=ID:return cfg
    if cfg['train_cfg']['max_iters']!=80000 or cfg['model']['decode_head']['num_classes']!=11:raise ValueError('CamVid requires11 classes/independent80K')
    bb=cfg['model']['backbone']
    for k in ['method','fusion','conditioning','relay_options']:bb.pop(k,None)
    bb.update(type='SemDiRDirectBackbone',direct_hidden=32,image_size=512)
    cfg['model'].update(type='DirectRegionPreservingSegmentor',region_preservation=deepcopy(direct.registration.RSP))
    for key,split in [('train_dataloader','train'),('val_dataloader','val'),('test_dataloader','test')]:
        old=cfg[key]['dataset'];pipe=old['pipeline']
        for item in pipe:
            if item['type']=='LoadAnnotations':item['reduce_zero_label']=False
            if item['type'] in ['RandomResize','Resize']:item['scale']=(2048,512)
        cfg[key]['dataset']=dict(type='SemDirCamVid11Dataset',data_root=old['data_root'],
            img_suffix='.png',seg_map_suffix='_L.png',reduce_zero_label=False,ignore_index=255,
            data_prefix=dict(img_path='CamVid_RGB',seg_map_path='CamVidGray'),
            ann_file=f'splits/{split}.txt',pipeline=pipe)
    cfg['custom_imports']['imports'].append('competitors.semdir_direct_camvid');cfg['custom_hooks']=[dict(type='DirectAuditHook')]
    cfg['paper'].update(method='semdir_direct',budget=80000,architecture_reference='SemDiR-Direct',
        provenance_label='NEW beta-free Direct; independent CamVid11 ImageNet80K; val101 distinct from test233',
        label_mapping=dict(classes=list(CLASSES),model_ids='0..10',ignore_index=255,reduce_zero_label=False),
        evaluation_protocol=dict(train_images=367,val_images=101,test_images=233,
            final_split='test',selection='fixed_iteration80000; validation does not select checkpoint',
            terminal_seal='training validation101 SS; final held-out test233 SS/MS reported separately'),
        direct_semantics=dict(D=384,d=32,K=4,semantic_projection=[1536,32],output_projection=[96,384],
            beta=False,cross_layer_difference=False,complementary_routing=False,direct_injection=False,
            SAG=True,original_RSP=True,boundary_RSP=False,identity_initialization=True,training_schedule_end=80000))
    return cfg

def camvid_index(root):
    root=Path(root);metadata=json.loads((root/'manifest.json').read_text());classes=json.loads((root/'classes.json').read_text())
    if metadata['total_pairs']!=701 or metadata['splits']!=COUNTS or classes['reduce_zero_label'] is not False or classes['ignore_index']!=255:
        raise ValueError('Expected verified original CamVid11 split/label encoding')
    if tuple(x['name'] for x in classes['classes'])!=CLASSES:raise ValueError('CamVid class order changed')
    files=[root/'manifest.json',root/'classes.json'];sets=[]
    for split,n in COUNTS.items():
        index=root/'splits'/f'{split}.txt';ids=[x.strip() for x in index.read_text().splitlines() if x.strip()]
        if len(ids)!=n or len(set(ids))!=n or any('/' in x or '\\' in x or x in ['.','..'] for x in ids):raise ValueError('Unexpected CamVid split: '+split)
        sets.append(set(ids));files.append(index)
        files.extend(p for x in ids for p in [root/'CamVid_RGB'/f'{x}.png',root/'CamVidGray'/f'{x}_L.png'])
    if len(set.union(*sets))!=701:raise ValueError('CamVid split overlap')
    def record(p):
        s=p.stat();return str(p.relative_to(root)),s.st_size,s.st_mtime_ns
    with ThreadPoolExecutor(max_workers=16) as pool:rows=list(pool.map(record,files))
    return dict(kind='names_sizes_mtime_not_content_hash',files=len(rows),
        sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),split_counts=COUNTS,
        dataset_manifest_sha256=sha256(root/'manifest.json'),label_mapping='0..10 including Sky0;255ignore;no zero reduction')

def dataset_index(root,dataset):return camvid_index(root) if dataset=='camvid11' else _parent_index(root,dataset)

def prepare(args,output):
    names=experiments.expand(args.experiments or [])
    if ID not in names or (output/'study.json').exists():return _parent_prepare(args,output)
    if names!=[ID]:raise ValueError('CamVid needs its own independent study')
    if any(p.name!='.lock' for p in output.iterdir()):raise FileExistsError('Preserve nonempty unregistered output')
    if os.environ.get('DCPA_COMP_INIT'):raise ValueError('CamVid independent ImageNet initialization only')
    from mmengine.config import Config
    runtime=resolve_runtime(args.world_size,args.per_gpu,args.global_batch,args.seed);spec=EXPERIMENTS[ID];weight,digest=default_weight(spec)
    if not weight.is_file() or sha256(weight)!=digest:raise ValueError('Missing or changed ImageNet weights')
    root=Path(os.environ.get('DCPA_CAMVID_ROOT',str(DEFAULT_ROOT))).resolve();index=camvid_index(root)
    path=output/'configs'/f'{ID}.py';cfg=config.make_config(ID,root,weight,output/'runs'/ID,digest,**runtime);config.validate_config(cfg)
    path.parent.mkdir(parents=True,exist_ok=True);Config(cfg).dump(str(path))
    manifest=dict(protocol=PROTOCOL,**runtime,small_ce_normalization='global_eligible_images',
        experiments=[dict(id=ID,spec=asdict(spec),steps=80000,config=str(path),sha256=sha256(path))],
        datasets=dict(camvid11=dict(root=str(root),index=index)),pretrained=dict(deit_s=dict(path=str(weight),sha256=digest)),
        sources=run.sources(),environment=run.environment())
    write_once(output/'study.json',manifest)
    for p in [path,output/'study.json']:p.chmod(0o444)
    return manifest

def configure_cli(argv):
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--camvid-root',type=Path);known,rest=parser.parse_known_args(argv)
    if known.camvid_root is not None:os.environ['DCPA_CAMVID_ROOT']=str(known.camvid_root.resolve())
    return rest
config._build=build;run.prepare=prepare;run.dataset_index=dataset_index


def assert_test_protocol(cfg):
    ds=cfg['test_dataloader']['dataset'];meta=cfg['paper']['evaluation_protocol']
    if (ds['type']!='SemDirCamVid11Dataset' or ds['ann_file']!='splits/test.txt'
        or ds['reduce_zero_label'] is not False or cfg['model']['decode_head']['num_classes']!=11
        or meta['final_split']!='test' or meta['test_images']!=233
        or cfg['val_dataloader']['dataset']['ann_file']!='splits/val.txt'):
        raise ValueError('CamVid final evaluation must retain Sky0 and use test233, distinct from val101')
    return dict(images=233,validation_images=101,split='test',reduce_zero_label=False)
