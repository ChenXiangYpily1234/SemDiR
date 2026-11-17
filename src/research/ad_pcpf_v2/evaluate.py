"""Read-only sealed-checkpoint evaluation: clean diagnostics or COCO shared-47.

No fine-tuning, target-domain adaptation, checkpoint selection or GT in inference.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from PIL import Image
from .bootstrap import ROOT, activate
from .artifacts import sha256, load, write_once
from .experiments import get_experiment


MS_RATIOS = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)



def evaluation_world_size(requested=None):
    """Use the visible allocation; never silently fall back to single GPU."""
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        import torch
        available = torch.cuda.device_count()
    else:
        available = len([v for v in visible.split(',') if v.strip() and v.strip() != '-1'])
    world = available if requested is None else int(requested)
    if world < 2 or world > available:
        raise ValueError(f'Evaluation needs 2..{available} visible GPUs, requested {world}')
    return world


def evaluation_shard(pairs, rank, world):
    if not 0 <= rank < world:
        raise ValueError('Invalid evaluation shard rank')
    return pairs[rank::world]


def merge_evaluation_shards(parts, pairs):
    """Merge exact, unpadded image shards in original dataset order."""
    by_id = {}
    total = dict(transfer_cm=np.zeros((47,48), dtype=np.int64),
                 mapped_pixels=0, valid_target_pixels=0)
    for part in parts:
        if len(part['ids']) != len(part['counts']):
            raise ValueError('Shard IDs/counts are not aligned')
        for i, image_id in enumerate(part['ids']):
            if image_id in by_id:
                raise ValueError('Repeated evaluation image')
            by_id[image_id] = (part['rows'][i] if part['rows'] else None, part['counts'][i])
        for key in total:
            total[key] += part[key]
    ids = [p[0] for p in pairs]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError('Incomplete or unexpected evaluation images')
    total.update(ids=ids, counts=[by_id[i][1] for i in ids],
                 rows=[by_id[i][0] for i in ids] if any(p['rows'] for p in parts) else [])
    return total


def evaluate_image_shard(study, name, scope, pairs, rank, source_lut, target_lut, budget):
    """One spawned process and one model per GPU; no padded sampler."""
    os.environ['DCPA_COMPETITOR_BUDGET'] = budget
    activate()
    if name.startswith('SemDiR-Selective-'):
        from competitors import selective
    elif name not in __import__('research.ad_pcpf_v2.experiments', fromlist=['EXPERIMENTS']).EXPERIMENTS:
        from competitors import registration
    import torch
    from mmseg.apis import inference_model
    from .metrics import image_statistics
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    model, _, _, _ = load_sealed_model(Path(study), name, f'cuda:{rank}')
    rows, counts, ids = [], [], []
    transfer_cm = np.zeros((47,48), dtype=np.int64)
    mapped_pixels = valid_target_pixels = 0
    classes = get_experiment(name).classes
    for index, (image_id, image_path, label_path) in enumerate(pairs):
        with torch.inference_mode():
            pred = inference_model(model, str(image_path)).pred_sem_seg.data[0].cpu().numpy().astype(np.int64)
        gt_raw = np.asarray(Image.open(label_path), dtype=np.int64)
        if pred.shape != gt_raw.shape:
            raise ValueError('Prediction was not restored to original GT dimensions')
        if scope == 'clean':
            gt = gt_raw - 1
            gt[(gt_raw == 0) | (gt_raw == 255)] = 255
            stats = image_statistics(pred, gt, classes, diagnostics=True)
            rows.append(stats)
            cm = stats['confusion']
            counts.append(np.stack((np.diag(cm), cm.sum(1), cm.sum(0))))
        else:
            if gt_raw.min() < 0 or gt_raw.max() > 255 or pred.min() < 0 or pred.max() >= 150:
                raise ValueError('Raw COCO PNG or ADE prediction has unexpected IDs')
            gt, prediction = target_lut[gt_raw], source_lut[pred]
            valid = gt != 255
            cm = np.bincount(gt[valid]*48 + prediction[valid], minlength=47*48).reshape(47,48)
            transfer_cm += cm
            counts.append(np.stack((np.diag(cm[:,:47]), cm.sum(1), cm[:,:47].sum(0))))
            mapped_pixels += int(valid.sum())
            valid_target_pixels += int((gt_raw != 255).sum())
        ids.append(image_id)
        if (index+1) % 100 == 0:
            print(f'SS GPU{rank}: {index+1}/{len(pairs)}', flush=True)
    return dict(rows=rows, counts=counts, ids=ids, transfer_cm=transfer_cm,
                mapped_pixels=mapped_pixels, valid_target_pixels=valid_target_pixels)


def run_parallel_single_scale(study, name, scope, pairs, world=None, source_lut=None, target_lut=None):
    world = evaluation_world_size(world)
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    budget = os.environ.get('DCPA_COMPETITOR_BUDGET', '160000')
    with ProcessPoolExecutor(max_workers=world, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(evaluate_image_shard, str(study), name, scope,
                            evaluation_shard(pairs, rank, world), rank, source_lut, target_lut, budget)
                for rank in range(world)]
        parts = [job.result() for job in jobs]
    return merge_evaluation_shards(parts, pairs)


def load_sealed_model(study, name, device='cuda:0'):
    activate()
    from .run import verify, frozen, sources, environment
    from mmengine.config import Config
    from mmseg.utils import register_all_modules
    from mmseg.apis import init_model
    manifest = load(study / 'study.json')
    actual = sources()
    changed = {k for k in set(actual) | set(manifest['sources']) if actual.get(k) != manifest['sources'].get(k)}
    if changed - {'v2/evaluate.py', 'v2/tta_test_entry.py'}:
        raise RuntimeError('Model/training source drift during evaluation: '+str(sorted(changed)))
    if environment() != manifest['environment']:
        raise RuntimeError('Evaluation environment differs from the registered study')
    if not changed:
        manifest = verify(study)
    record = frozen(study, name)
    folder = study / 'frozen' / name
    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(str(folder / 'config.py'))
    # inference_model reads cfg.test_pipeline, not the training data loader.
    cfg.test_pipeline = [p for p in cfg.test_dataloader.dataset.pipeline if p['type'] != 'LoadAnnotations']
    # The immutable seal was checked above; trust only this exact checkpoint.
    import torch
    checkpoint = (folder / 'model.pth').resolve()
    original_load = torch.load
    def load_trusted(file, *positional, **keywords):
        try:
            trusted = Path(file).resolve() == checkpoint
        except (TypeError, ValueError, OSError):
            trusted = False
        if trusted:
            keywords['weights_only'] = False
        return original_load(file, *positional, **keywords)
    torch.load = load_trusted
    try:
        model = init_model(cfg, str(checkpoint), device=device)
    finally:
        torch.load = original_load
    model.eval()
    return model, cfg, manifest, record


def dataset_pairs(spec, root):
    if spec.dataset == 'ade20k':
        ids = sorted(p.stem for p in (root / 'images/validation').glob('*.jpg'))
        if len(ids) != 2000:
            raise ValueError('Expected complete ADE20K validation split')
        return [(i, root / 'images/validation' / (i+'.jpg'), root / 'annotations/validation' / (i+'.png')) for i in ids]
    ids = [s.strip() for s in (root / 'ImageSets/SegmentationContext/val.txt').read_text().splitlines() if s.strip()]
    if len(ids) != 5105 or len(set(ids)) != len(ids):
        raise ValueError('Expected complete official Context59 validation split')
    return [(i, root / 'JPEGImages' / (i+'.jpg'), root / 'SegmentationClassContext' / (i+'.png')) for i in ids]


def load_mapping(path, expected_sha):
    if not expected_sha or sha256(path) != expected_sha:
        raise ValueError('An explicit verified mapping SHA256 is required')
    value = load(path)
    rows = value.get('mapping', value).get('classes')
    if not isinstance(rows, list) or len(rows) != 47:
        raise ValueError('Expected 47 classes in mapping.classes or classes')
    rows = sorted(rows, key=lambda x: x['index'])
    if ([r['index'] for r in rows] != list(range(47))
        or len({r['ade_class_id'] for r in rows}) != 47
        or len({r['coco_pixel_id'] for r in rows}) != 47
        or any(not 0 <= r['ade_class_id'] < 150 or not 0 <= r['coco_pixel_id'] < 255 for r in rows)):
        raise ValueError('Invalid or non-bijective shared-class mapping')
    source = np.full(150, 47, dtype=np.int64)  # Dedicated error column, never renormalize away errors.
    target = np.full(256, 255, dtype=np.int64)
    for row in rows:
        source[row['ade_class_id']] = row['index']
        target[row['coco_pixel_id']] = row['index']
    return source, target, rows


def run_multiscale_iou(study, spec, reference, cfg, output, world_size=None):
    """Run official MMSeg IoUMetric with six scales and horizontal-flip TTA."""
    from mmengine.config import Config

    world_size = evaluation_world_size(world_size)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = (study / 'frozen' / spec.id / 'model.pth').resolve()
    test_cfg = Config.fromfile(str(study / 'frozen' / spec.id / 'config.py'))
    test_cfg.work_dir = str(output / 'runner')
    test_cfg.tta_model = dict(type='SegTTAModel')
    test_cfg.tta_pipeline = [
        dict(type='LoadImageFromFile', backend_args=None),
        dict(type='TestTimeAug', transforms=[
            [dict(type='Resize', scale_factor=ratio, keep_ratio=True) for ratio in MS_RATIOS],
            [dict(type='RandomFlip', prob=0.0, direction='horizontal'),
             dict(type='RandomFlip', prob=1.0, direction='horizontal')],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')],
        ]),
    ]
    config_path = output / 'tta_config.py'
    test_cfg.dump(str(config_path))
    launcher_log = output / 'launcher.log'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc_per_node={world_size}', '--module', 'research.ad_pcpf_v2.tta_test_entry',
        '--trusted-checkpoint', str(checkpoint), str(config_path), str(checkpoint), '--tta',
        '--launcher', 'pytorch', '--work-dir', str(output / 'runner')]
    env = dict(os.environ)
    search_path = [str(ROOT), str(ROOT / 'mmsegmentation')]
    if env.get('PYTHONPATH'):
        search_path.append(env['PYTHONPATH'])
    env['PYTHONPATH'] = os.pathsep.join(search_path)
    with launcher_log.open('w', encoding='utf-8') as handle:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        tail = launcher_log.read_text(encoding='utf-8', errors='replace')[-12000:]
        raise RuntimeError(f'Multiscale MMSeg test failed ({result.returncode}):\n{tail}')

    metric_files = []
    runner_dir = output / 'runner'
    for path in runner_dir.rglob('*.json'):
        try:
            value = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError):
            continue
        if all(key in value for key in ('mIoU', 'mAcc', 'aAcc')):
            metric_files.append((path.stat().st_mtime_ns, path, value))
    if not metric_files:
        raise RuntimeError(f'Multiscale runner exited without standard IoU metrics; see {launcher_log}')
    _, metric_path, metrics = max(metric_files)
    scores = {key: float(metrics[key]) for key in ('mIoU', 'mAcc', 'aAcc')}
    if any(not np.isfinite(value) or not 0 <= value <= 100 for value in scores.values()):
        raise RuntimeError(f'Invalid multiscale IoUMetric values: {scores}')
    per_class_iou = {key[4:]: (float(value) if np.isfinite(float(value)) else None)
                     for key, value in metrics.items() if str(key).startswith('IoU.')}
    logs = sorted(runner_dir.rglob('*.log'), key=lambda p: p.stat().st_mtime_ns)
    return dict(**scores, evaluator='MMSegmentation.IoUMetric', metric_json=str(metric_path),
        per_class_IoU=per_class_iou,
        per_class_log=str(logs[-1]) if logs else str(launcher_log),
        scales=list(MS_RATIOS), flips=['none', 'horizontal'], view_count=len(MS_RATIOS)*2,
        checkpoint_sha256=reference['checkpoint_sha256'], config_sha256=sha256(config_path),
        evaluation_world_size=world_size, training_steps=0, test_time_adaptation=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--scope', choices=('clean','transfer47'), default='clean')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', choices=('ss', 'ms', 'both'), default='both',
                        help='SS is the frozen config inference; MS uses six-scale x flip TTA')
    parser.add_argument('--eval-world-size', '--tta-world-size', dest='tta_world_size', type=int, default=None,
                        help='GPU count for BOTH SS/MS; default all visible GPUs, minimum2')
    parser.add_argument('--images', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--mapping', type=Path)
    parser.add_argument('--mapping-sha256')
    parser.add_argument('--device', default='cuda:0', help='Legacy option; GPU allocation now uses CUDA_VISIBLE_DEVICES and --eval-world-size')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.error('Evaluation consumes device resources; explicitly pass --execute')
    try:
        args.tta_world_size = evaluation_world_size(args.tta_world_size)
    except ValueError as exc:
        parser.error(str(exc))
    if args.scope == 'transfer47' and args.scales != 'ss':
        parser.error('transfer47 currently supports --scales ss only')
    study, output = args.study.resolve(), args.output.resolve()
    spec = get_experiment(args.experiment)
    model, cfg, manifest, reference = load_sealed_model(study, spec.id, 'cpu')
    terminal_eval = load(study / 'frozen' / spec.id / 'terminal_eval.json') if args.scope == 'clean' else None
    import torch
    from mmseg.apis import inference_model
    from .metrics import image_statistics, aggregate
    mapping = None
    if args.scope == 'clean':
        pairs = dataset_pairs(spec, Path(manifest['datasets'][spec.dataset]['root']))
    else:
        if spec.dataset != 'ade20k' or not all((args.images, args.labels, args.mapping, args.mapping_sha256)):
            parser.error('transfer47 needs an ADE model and --images/--labels/--mapping/--mapping-sha256')
        source_lut, target_lut, mapping = load_mapping(args.mapping, args.mapping_sha256)
        images = sorted(args.images.glob('*.jpg'))
        labels = sorted(args.labels.glob('*.png'))
        if len(images) != 5000 or len(labels) != 5000 or {p.stem for p in images} != {p.stem for p in labels}:
            raise ValueError('Expected exactly 5000 COCO-Stuff val image/annotation pairs')
        pairs = [(p.stem, p, args.labels / (p.stem+'.png')) for p in images]
    inventory = [(i, str(im.resolve()), im.stat().st_size, im.stat().st_mtime_ns,
                  str(gt.resolve()), gt.stat().st_size, gt.stat().st_mtime_ns) for i, im, gt in pairs]
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'evaluation_manifest.json', dict(scope=args.scope, experiment=spec.id,
        checkpoint_sha256=reference['checkpoint_sha256'], source_study_sha256=sha256(study / 'study.json'),
        evaluator_sha256=sha256(__file__), mapping_sha256=args.mapping_sha256,
        mapping=mapping, images=len(pairs), inventory_kind='paths_sizes_mtime',
        inventory_sha256=hashlib.sha256(json.dumps(inventory).encode()).hexdigest(),
        scales=args.scales, multi_scale=dict(ratios=list(MS_RATIOS), flips=['none','horizontal'],
        evaluator='MMSegmentation.IoUMetric') if args.scales in ('ms','both') else None,
        inference=dict(cfg.model.test_cfg), evaluation_world_size=args.tta_world_size,
        evaluation_source_sha256=sha256(__file__), training_source_sha256=manifest['sources'].get('v2/evaluate.py'),
        model_training_sources_unchanged=True, training_steps=0, test_time_adaptation=False))
    cfg.dump(str(output / 'inference_config.py'))
    rows, counts, ids = [], [], []
    transfer_cm = np.zeros((47, 48), dtype=np.int64)
    mapped_pixels, valid_target_pixels = 0, 0
    scales = {}
    if args.scales in ('ss', 'both'):
        del model
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        merged = run_parallel_single_scale(study, spec.id, args.scope, pairs,
                                          args.tta_world_size, source_lut if mapping else None,
                                          target_lut if mapping else None)
        rows, counts, ids = merged['rows'], merged['counts'], merged['ids']
        transfer_cm = merged['transfer_cm']
        mapped_pixels, valid_target_pixels = merged['mapped_pixels'], merged['valid_target_pixels']
        if args.scope == 'clean':
            ss_report, matrix, bi, bu = aggregate(rows, spec.classes)
            if abs(float(ss_report['mIoU']) - float(terminal_eval['miou'])) > .02:
                raise RuntimeError('Offline SS confusion matrix differs from fixed-terminal IoUMetric by >0.02 pp')
            scales['SS'] = dict(mIoU=terminal_eval['miou'],
                mAcc=terminal_eval['macc'], aAcc=terminal_eval['aacc'],
                per_class_IoU=ss_report['per_class_IoU'], images=len(pairs),
                evaluator='MMSegmentation.IoUMetric (fixed terminal); offline per-class confusion cross-check')
            np.savez_compressed(output / 'boundary_counts.npz', intersection=bi, union=bu)
        else:
            matrix = transfer_cm
            inter = np.diag(matrix[:,:47])
            union = matrix.sum(1) + matrix[:,:47].sum(0) - inter
            iou = np.divide(inter, union, out=np.full(47, np.nan), where=union > 0)
            ss_report = dict(mIoU=float(np.nanmean(iou)*100),
                per_class_IoU=[None if not np.isfinite(v) else float(v*100) for v in iou],
                evaluable_classes=int((union > 0).sum()), images=len(pairs), mapped_pixels=mapped_pixels,
                target_valid_pixels=valid_target_pixels, coverage=mapped_pixels/max(1,valid_target_pixels),
                protocol='ADE argmax then fixed47 mapping; unmapped predictions remain an error column')
            scales['SS'] = ss_report
        np.save(output / 'confusion.npy', matrix, allow_pickle=False)
        np.savez_compressed(output / 'per_image_counts.npz', ids=np.asarray(ids), counts=np.stack(counts))
    if args.scales in ('ms', 'both'):
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        scales['MS'] = run_multiscale_iou(study, spec, reference, cfg,
            output / 'multi_scale', args.tta_world_size)
    if args.scope == 'transfer47':
        report = scales['SS']
    else:
        report = dict(SS_mIoU=scales.get('SS', {}).get('mIoU'),
            MS_mIoU=scales.get('MS', {}).get('mIoU'), scales=scales,
            checkpoint_sha256=reference['checkpoint_sha256'], protocol=manifest['protocol'],
            experiment=spec.id, images=len(pairs), evaluation_world_size=args.tta_world_size, training_steps=0, test_time_adaptation=False)
        if args.scales in ('ss', 'both'):
            report.update({key: value for key, value in scales['SS'].items()
                           if key not in ('per_class_IoU', 'images', 'evaluator')})
    if args.scope == 'transfer47':
        np.save(output / 'confusion.npy', matrix, allow_pickle=False)
        np.savez_compressed(output / 'per_image_counts.npz', ids=np.asarray(ids), counts=np.stack(counts))
    report['evaluation_world_size'] = args.tta_world_size
    write_once(output / 'results.json', dict(report,
        confusion_sha256=sha256(output / 'confusion.npy') if (output / 'confusion.npy').is_file() else None,
        per_image_sha256=sha256(output / 'per_image_counts.npz')
            if (output / 'per_image_counts.npz').is_file() else None))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
