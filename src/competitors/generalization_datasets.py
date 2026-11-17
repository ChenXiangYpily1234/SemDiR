"""Process-local Exact target-dataset recipes using the native study engine.

Both arms start independently from ImageNet and run 80K steps. COCO-Stuff-10K
uses the official v1.1 9K/1K split and converted PNG IDs 1..171; MMSeg reduces
zero exactly once when loading labels. Existing runs and recipes are untouched.
"""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import hashlib
import json
import os

from research.ad_pcpf_v2.bootstrap import activate
activate()
from research.ad_pcpf_v2 import config, experiments, run, evaluate
from research.ad_pcpf_v2.artifacts import sha256, write_once
from research.ad_pcpf_v2.runtime import resolve_runtime
from . import registration, registry
from .registry import CompetitorExperiment, ROOT


COCO_SPLITS = {
    'train': (9000, '511b98d0ef2ae15b1a5e8957401e50af46430077550e3c580a4bdd1d252583c4'),
    'test': (1000, '0994fddc9b4fe2cc769ef235cf1905eb90d7dc21b528c9316e5387a2530786a5'),
}
LABEL_MAPPINGS = {
    'context59': dict(dataset='PASCAL-Context-59', classes=59, png_ids='0=background, 1..59=classes',
                      model_ids='0..58', reduce_zero_label=True, ignore_index=255),
    'cocostuff10k': dict(dataset='COCO-Stuff-10K official v1.1', classes=171,
        conversion='mmsegmentation/tools/dataset_converters/coco_stuff10k.py:clsID_to_trID',
        png_ids='0=background, 1..171=classes', model_ids='0..170',
        reduce_zero_label=True, ignore_index=255,
        split_sha256={key: value[1] for key, value in COCO_SPLITS.items()}),
}


@dataclass(frozen=True)
class TargetDatasetExperiment(CompetitorExperiment):
    @property
    def steps(self):
        return 80000

    @property
    def classes(self):
        return 171 if self.dataset == 'cocostuff10k' else 59


EXPERIMENTS = {
    'SemDiR-Exact-Context59': TargetDatasetExperiment('SemDiR-Exact-Context59', 'm23_plain',
        dataset='context59', backbone='deit_s', comparison='Context59-Vanilla'),
    'SemDiR-Exact-COCO10K': TargetDatasetExperiment('SemDiR-Exact-COCO10K', 'm23_plain',
        dataset='cocostuff10k', backbone='deit_s', comparison='COCOStuff10K-Vanilla'),
}
_original_build = config._build
_original_prepare = run.prepare
_original_dataset_index = run.dataset_index
_original_dataset_pairs = evaluate.dataset_pairs
_installed = False


def configure_cli(argv):
    """Consume our optional root flag before the unchanged native CLI parser."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--coco-root', type=Path)
    known, rest = parser.parse_known_args(argv)
    if known.coco_root is not None:
        os.environ['DCPA_COCO10K_ROOT'] = str(known.coco_root.resolve())
    return rest


def build(*args, **kwargs):
    cfg = _original_build(*args, **kwargs)
    name = cfg['paper']['experiment']
    if name not in EXPERIMENTS:
        return cfg
    spec = EXPERIMENTS[name]
    cfg['paper'].update(budget=spec.steps, architecture_reference='M23-PlainFusion',
        provenance_label='SemDiR-Exact unchanged Full + RSP; independent target-dataset ImageNet initialization',
        label_mapping=deepcopy(LABEL_MAPPINGS[spec.dataset]),
        target_dataset_protocol='independent_target_dataset_80000')
    cfg['custom_imports']['imports'].append('competitors.generalization_datasets')
    if spec.dataset == 'cocostuff10k':
        for key in ('train_dataloader', 'val_dataloader', 'test_dataloader'):
            train = key == 'train_dataloader'
            old = cfg[key]['dataset']
            split = 'train2014' if train else 'test2014'
            # COCO baseline uses the ADE resize geometry (2048,512).
            pipeline = old['pipeline']
            for item in pipeline:
                if item['type'] in ('RandomResize', 'Resize'):
                    item['scale'] = (2048, 512)
            cfg[key]['dataset'] = dict(type='COCOStuffDataset', data_root=old['data_root'],
                reduce_zero_label=True, seg_map_suffix='_labelTrainIds.png', data_prefix=dict(img_path='images/'+split,
                seg_map_path='annotations/'+split), pipeline=pipeline)
    return cfg


def coco_pairs(root):
    """Validate official split bytes and exact image/label pairing, without GPU."""
    root = Path(root)
    pairs = {}
    for split, (expected, expected_sha) in COCO_SPLITS.items():
        index = root / 'imageLists' / (split + '.txt')
        data = index.read_bytes()
        ids = [line.strip() for line in data.decode().splitlines() if line.strip()]
        if (hashlib.sha256(data).hexdigest() != expected_sha or len(ids) != expected
            or len(set(ids)) != expected or any('/' in i or '\\' in i or i in ('.', '..') for i in ids)):
            raise ValueError(f'Expected official COCO-Stuff-10K v1.1 {split} split ({expected})')
        images = root / 'images' / (split + '2014')
        labels = root / 'annotations' / (split + '2014')
        if ({p.stem for p in images.glob('*.jpg')} != set(ids)
            or {p.name for p in labels.glob('*.png')} != {i+'_labelTrainIds.png' for i in ids}):
            raise ValueError(f'Incomplete or unexpected COCO-Stuff-10K {split} pairs')
        pairs[split] = [(i, images / (i+'.jpg'), labels / (i+'_labelTrainIds.png')) for i in ids]
    if {p[0] for p in pairs['train']} & {p[0] for p in pairs['test']}:
        raise ValueError('COCO-Stuff-10K train/test leakage')
    return pairs


def dataset_index(root, dataset):
    if dataset != 'cocostuff10k':
        return _original_dataset_index(root, dataset)
    root = Path(root)
    pairs = coco_pairs(root)
    files = []
    for split in COCO_SPLITS:
        files.append(root/'imageLists'/(split+'.txt'))
        files.extend(path for _, image, label in pairs[split] for path in (image, label))
    def record(path):
        stat = path.stat()
        return (str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns)
    with ThreadPoolExecutor(max_workers=16) as pool:
        records = list(pool.map(record, files))
    return dict(kind='names_sizes_mtime_not_content_hash', files=len(files),
        sha256=hashlib.sha256(json.dumps(records).encode()).hexdigest(),
        split_counts={split: len(rows) for split, rows in pairs.items()},
        label_mapping=deepcopy(LABEL_MAPPINGS['cocostuff10k']))


def dataset_pairs(spec, root):
    if spec.dataset == 'cocostuff10k':
        return coco_pairs(root)['test']
    return _original_dataset_pairs(spec, root)


def prepare(args, output):
    names = experiments.expand(args.experiments or ['competition_core'])
    if not any(name in EXPERIMENTS for name in names) or (output/'study.json').exists():
        return _original_prepare(args, output)
    from mmengine.config import Config
    if any(p.name != '.lock' for p in output.iterdir()):
        raise FileExistsError('Preserve nonempty unregistered directory')
    runtime = resolve_runtime(args.world_size, args.per_gpu, args.global_batch, args.seed)
    datasets, pretrained, configs = {}, {}, []
    for name in names:
        spec = registry.EXPERIMENTS[name]
        if name in registry.BLOCKERS:
            raise ValueError(registry.BLOCKERS[name])
        if os.environ.get('DCPA_COMP_INIT'):
            if len(names) != 1:
                raise ValueError('Explicit initialization requires one experiment per study')
            path = Path(os.environ['DCPA_COMP_INIT']).resolve()
            digest = os.environ['DCPA_COMP_INIT_SHA']
        else:
            path, digest = registry.default_weight(spec)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError('Missing or changed pretrained bytes')
        if spec.backbone in pretrained and pretrained[spec.backbone]['sha256'] != digest:
            raise ValueError('Pretraining identities differ in study')
        pretrained[spec.backbone] = dict(path=str(path), sha256=digest)
        if spec.dataset not in datasets:
            if spec.dataset == 'cocostuff10k':
                root = Path(os.environ.get('DCPA_COCO10K_ROOT', str(ROOT.parent/'datasets/COCOStuff10K'))).resolve()
            else:
                candidate = args.ade_root if spec.dataset == 'ade20k' else args.context_root
                if candidate is None:
                    raise ValueError(f'Explicit --context-root required for {spec.dataset}')
                root = candidate.resolve()
            datasets[spec.dataset] = dict(root=str(root), index=run.dataset_index(root, spec.dataset))
        file = output/'configs'/(name+'.py')
        cfg = config.make_config(name, datasets[spec.dataset]['root'], path,
                                 output/'runs'/name, digest, **runtime)
        config.validate_config(cfg)
        configs.append((spec, file, cfg))
    rows = []
    for spec, file, cfg in configs:
        file.parent.mkdir(parents=True, exist_ok=True)
        Config(cfg).dump(str(file))
        rows.append(dict(id=spec.id, spec=asdict(spec), steps=spec.steps, config=str(file), sha256=sha256(file)))
    manifest = dict(protocol=registration.PROTOCOL, **runtime,
        small_ce_normalization='global_eligible_images', experiments=rows, datasets=datasets,
        pretrained=pretrained, sources=run.sources(), environment=run.environment())
    write_once(output/'study.json', manifest)
    for file in [output/'study.json', *(file for _, file, _ in configs)]:
        file.chmod(0o444)
    return manifest


def install():
    global _installed
    if _installed:
        return
    registry.EXPERIMENTS.update(EXPERIMENTS)
    experiments.EXPERIMENTS.update(EXPERIMENTS)
    experiments.GROUPS['semdir_exact_datasets'] = tuple(EXPERIMENTS)
    config._build = build
    run.prepare = prepare
    run.dataset_index = dataset_index
    evaluate.dataset_pairs = dataset_pairs
    _installed = True


install()
