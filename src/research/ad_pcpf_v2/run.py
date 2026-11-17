"""V2 paper coordinator: prepare -> per-experiment smoke -> terminal -> seal.

No command launches GPU work unless --execute is explicit. No automatic restart,
source-lock bypass, V1 continuation, or substitution of published scores.
"""
from __future__ import annotations
import argparse
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from .bootstrap import ROOT, activate, check_loaded_paths
from .artifacts import sha256, load, write_once, identity
from .experiments import (PROTOCOL, FDAM_BASELINE_PROTOCOL, R9_PROTOCOL, R10_PROTOCOL, RSP_PROTOCOL, CFRT_PROTOCOL, study_protocol, PRETRAIN_SHA256, EXPERIMENTS, HISTORICAL, GROUPS,
                          expand, get_experiment)
from .config import make_config, smoke_config, diagnostic_config, validate_config

from .runtime import resolve_runtime, runtime_from_config

DEFAULT_OUTPUT = ROOT.parent / 'trained_models/ADPCPF-PAPER-V2-R7-RUNTIME'


@contextmanager
def study_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def environment():
    import torch
    import mmengine
    import mmcv
    paths = check_loaded_paths()
    packages = {}
    for name in ('torch', 'timm', 'mmengine', 'mmsegmentation', 'mmcv', 'ftfy',
                 'prettytable', 'cityscapesscripts', 'numpy', 'scipy'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = 'source-tree'
    return dict(python=sys.version, torch_cuda=torch.version.cuda, packages=packages,
                mmcv=mmcv.__version__, paths=paths, mmengine=str(Path(mmengine.__file__).resolve()))


def sources():
    import timm
    import mmseg
    import mmengine
    import mmcv
    groups = {'v2': Path(__file__).parent, 'timm': Path(timm.__file__).resolve().parent,
              'mmseg': Path(mmseg.__file__).resolve().parent,
              'mmengine': Path(mmengine.__file__).resolve().parent,
              'mmcv': Path(mmcv.__file__).resolve().parent,
              'dcpa_shared': ROOT / 'experiments/dcpa', 'pcpf6': ROOT / 'research/pcpf_v6'}
    result = {}
    for group, folder in groups.items():
        for path in sorted(folder.rglob('*.py')):
            result[group + '/' + str(path.relative_to(folder))] = sha256(path)
    for path in ('research/pcpf_v7/core.py', 'FDAM/FDAM_mmseg/mmcv_custom/deit_fdam.py',
                 'FDAM/SOURCE_REVISION.md', 'FDAM/AUTHORIZATION.md'):
        result[path] = sha256(ROOT / path)
    for path in (
            'FDAM/FDAM_mmseg/configs/vit/upernet_deit-s16_512x512_160k_ade20k.py',
            'FDAM/FDAM_mmseg/configs/vit/upernet_vit-b16_mln_512x512_160k_ade20k.py',
            'FDAM/FDAM_mmseg/configs/_base_/models/upernet_vit-b16_ln_mln.py',
            'FDAM/FDAM_mmseg/configs/_base_/datasets/ade20k.py',
            'FDAM/FDAM_mmseg/configs/_base_/schedules/schedule_160k.py',
            'FDAM/FDAM_mmseg/configs/_base_/default_runtime.py',
            'mmsegmentation/configs/vit/vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512.py',
            'mmsegmentation/configs/vit/vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py'):
        result[path] = sha256(ROOT / path)
    return result


def dataset_index(root, dataset):
    root = Path(root)
    rows = []
    if dataset == 'ade20k':
        for split, expected in (('training', 20210), ('validation', 2000)):
            images = sorted((root / 'images' / split).glob('*.jpg'))
            labels = sorted((root / 'annotations' / split).glob('*.png'))
            if len(images) != expected or len(labels) != expected or {p.stem for p in images} != {p.stem for p in labels}:
                raise ValueError(f'Incomplete ADE20K {split} pairs')
            rows.extend(images + labels)
    elif dataset == 'context59':
        split_ids = {}
        for split, expected in (('train', 4998), ('val', 5105)):
            index = root / 'ImageSets/SegmentationContext' / f'{split}.txt'
            ids = index.read_text().splitlines()
            ids = [i.strip() for i in ids if i.strip()]
            if len(ids) != expected or len(set(ids)) != expected:
                raise ValueError(f'Expected official Context59 {split} split ({expected})')
            if any('/' in i or '\\' in i or i in ('.', '..') for i in ids):
                raise ValueError('Invalid split identifier')
            split_ids[split] = set(ids)
            rows.append(index)
            for i in ids:
                rows.extend((root / 'JPEGImages' / (i + '.jpg'), root / 'SegmentationClassContext' / (i + '.png')))
        if split_ids['train'] & split_ids['val']:
            raise ValueError('Context train/val leakage')
    else:
        raise ValueError('Unknown dataset')
    def record_file(path):
        stat = path.stat()
        return (str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns)
    # Remote attribute lookups are latency-bound. map preserves the original
    # inventory order, so the fingerprint is identical to the serial algorithm.
    print(f'Verifying {dataset} inventory: {len(rows)} files', flush=True)
    records = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index, record in enumerate(pool.map(record_file, rows), 1):
            records.append(record)
            if index % 10000 == 0 or index == len(rows):
                print(f'{dataset} inventory {index}/{len(rows)}', flush=True)
    import hashlib
    return dict(kind='names_sizes_mtime_not_content_hash', files=len(rows),
                sha256=hashlib.sha256(json.dumps(records).encode()).hexdigest())


def manifest_runtime(manifest):
    return resolve_runtime(**{key: manifest[key] for key in ('world_size', 'per_gpu', 'global_batch', 'seed')})


def check_runtime_args(args, manifest):
    runtime = manifest_runtime(manifest)
    for key, value in runtime.items():
        requested = getattr(args, key, None)
        if requested is not None and requested != value:
            raise ValueError(f'{key}={requested} differs from registered {value}; use a new output directory')


def prepare(args, output):
    from mmengine.config import Config
    if (output / 'study.json').exists():
        manifest = verify(output)
        check_runtime_args(args, manifest)
        if args.experiments and expand(args.experiments) != [r['id'] for r in manifest['experiments']]:
            raise ValueError('Cannot extend a locked study in place; choose a new output directory')
        return manifest
    if any(p.name != '.lock' for p in output.iterdir()):
        raise FileExistsError('Nonempty unregistered study directory; preserving existing files')
    runtime = resolve_runtime(args.world_size, args.per_gpu, args.global_batch, args.seed)
    names = expand(args.experiments or ['current'])
    protocol = study_protocol(names)
    roots = {'ade20k': args.ade_root.resolve(), 'context59': args.context_root.resolve() if args.context_root else None}
    weights = {'small': args.pretrained_s.resolve() if args.pretrained_s else None,
               'base': args.pretrained_b.resolve() if args.pretrained_b else None}
    expected = {'small': PRETRAIN_SHA256, 'base': args.pretrained_b_sha256}
    datasets, pretrained, configs = {}, {}, []
    # Complete preflight before writing any configs.
    env, code = environment(), sources()
    for name in names:
        e = get_experiment(name)
        if e.dataset not in datasets:
            if roots[e.dataset] is None:
                raise ValueError(f'Explicit --{e.dataset} data root required')
            datasets[e.dataset] = dict(root=str(roots[e.dataset]), index=dataset_index(roots[e.dataset], e.dataset))
        if e.backbone not in pretrained:
            path = weights[e.backbone]
            if path is None or not path.is_file() or not expected[e.backbone]:
                raise ValueError(f'Explicit {e.backbone} checkpoint and trusted SHA256 required')
            if sha256(path) != expected[e.backbone]:
                raise ValueError('Pretrained byte hash mismatch')
            pretrained[e.backbone] = dict(path=str(path), sha256=expected[e.backbone])
        config = output / 'configs' / (name + '.py')
        work = output / 'runs' / name
        value = make_config(name, roots[e.dataset], weights[e.backbone], work, expected[e.backbone], **runtime)
        validate_config(value)
        configs.append((e, config, value))
    rows = []
    for e, path, value in configs:
        path.parent.mkdir(parents=True, exist_ok=True)
        Config(value).dump(str(path))
        rows.append(dict(id=e.id, spec=asdict(e), steps=e.steps, config=str(path), sha256=sha256(path)))
    manifest = dict(protocol=protocol, **runtime,
        small_ce_normalization='global_eligible_images', experiments=rows, datasets=datasets,
        pretrained=pretrained, sources=code, environment=env)
    write_once(output / 'study.json', manifest)
    for path in [output / 'study.json', *(p for _, p, _ in configs)]:
        path.chmod(0o444)
    return manifest


def verify(output):
    from mmengine.config import Config
    manifest = load(output / 'study.json')
    runtime = manifest_runtime(manifest)

    if (manifest['protocol'] not in (PROTOCOL, FDAM_BASELINE_PROTOCOL, R9_PROTOCOL, R10_PROTOCOL, RSP_PROTOCOL, CFRT_PROTOCOL)
            or manifest['small_ce_normalization'] != 'global_eligible_images'):
        raise ValueError('Wrong protocol; old V1/V2 manifests cannot be migrated or edited')
    if sources() != manifest['sources'] or environment() != manifest['environment']:
        raise RuntimeError('Source/environment drift since registration; do not rewrite the manifest')
    for name, row in manifest['datasets'].items():
        if dataset_index(Path(row['root']), name) != row['index']:
            raise RuntimeError('Dataset index changed')
    for row in manifest['pretrained'].values():
        if sha256(row['path']) != row['sha256']:
            raise RuntimeError('Pretrained checkpoint changed')
    for row in manifest['experiments']:
        if row['spec'] != asdict(get_experiment(row['id'])) or sha256(row['config']) != row['sha256']:
            raise RuntimeError('Experiment/config changed')
        cfg = Config.fromfile(row['config'], import_custom_modules=False).to_dict()
        validate_config(cfg)
        if cfg['paper']['protocol'] != manifest['protocol']:
            raise RuntimeError('Config protocol differs from this study')
        if runtime_from_config(cfg) != runtime:
            raise RuntimeError('Config runtime differs from this registered study')
    return manifest




def selected_gpus(world_size):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    available = [dict(index=int(index.strip()), uuid=uuid.strip())
                 for index, uuid in (line.split(',') for line in lines if line.strip())]
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask is not None:
        selected = []
        for token in mask.split(','):
            token = token.strip()
            matches = [g for g in available if token and
                       (str(g['index']) == token or g['uuid'].startswith(token))]
            if len(matches) != 1:
                raise RuntimeError(f'Unknown/ambiguous CUDA_VISIBLE_DEVICES entry: {token!r}')
            selected.append(matches[0])
        if len({g['uuid'] for g in selected}) != len(selected):
            raise RuntimeError('Duplicate GPUs in CUDA_VISIBLE_DEVICES')
    else:
        selected = available
    if len(selected) < world_size:
        raise RuntimeError(f'world={world_size} requires {world_size} visible GPUs; found {len(selected)}')
    return selected[:world_size]


def idle_gpus(world_size, *, allow_sharing=False):
    selected = selected_gpus(world_size)
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                     '--format=csv,noheader,nounits'], text=True).splitlines()
    selected_uuids = {gpu['uuid'] for gpu in selected}
    busy = []
    for line in lines:
        if not line.strip():
            continue
        pid, uuid = [part.strip() for part in line.split(',')]
        if uuid in selected_uuids and pid != str(os.getpid()):
            busy.append((pid,uuid))
    if busy and not allow_sharing:
        pid,uuid = busy[0]
        raise RuntimeError(f'Selected GPU {uuid} is busy (PID {pid}); select idle GPUs or explicitly use --share-gpus')
    if busy:
        print(f'Explicit GPU sharing enabled: preserving {len(busy)} existing GPU processes.',flush=True)
        for gpu in selected:
            gpu['existing_pids'] = [pid for pid,uuid in busy if uuid == gpu['uuid']]
    return selected


def launch(config, work, resume=None, *, allow_sharing=False, topology_proof=None,
           budget_extension=False):
    from mmengine.config import Config
    cfg = Config.fromfile(str(config), import_custom_modules=False).to_dict()
    runtime = runtime_from_config(cfg)
    selected = idle_gpus(runtime['world_size'], allow_sharing=allow_sharing)
    if resume is None:
        work.mkdir(parents=True, exist_ok=False)
    elif not work.is_dir() or not Path(resume).is_file():
        raise FileNotFoundError('Resume requires this experiment directory and an existing checkpoint')
    env = dict(os.environ, PYTHONHASHSEED=str(runtime['seed']), OMP_NUM_THREADS='1')
    env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu['uuid'] for gpu in selected)
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT), str(ROOT / 'mmsegmentation'), env.get('PYTHONPATH', '')))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f"--nproc_per_node={runtime['world_size']}",
               '--module', 'research.ad_pcpf_v2.worker', str(config)]
    if resume:
        command += ['--resume-from', str(Path(resume).resolve())]
        if budget_extension:
            command += ['--extend-20k-checkpoint']
    if topology_proof:
        if not resume:
            raise ValueError('Topology proof requires a checkpoint')
        command += ['--topology-proof', str(Path(topology_proof).resolve())]
    stamp = time.time_ns()
    log = work / f'console_{stamp}.log'
    print(f'Launching {config}; logs: {log}', flush=True)
    with log.open('x') as handle:
        rc = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    receipt = dict(returncode=rc, runtime=runtime, gpus=selected, gpu_sharing_allowed=allow_sharing, config_sha256=sha256(config), log=str(log), resume=str(resume) if resume else None)
    write_once(work / f'process_exit_{stamp}.json', receipt)
    (work / 'last_process_exit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if rc:
        raise RuntimeError(f'Training process exit={rc}; preserved {log}')


def completed(work, config, name, steps, phase):
    import torch
    record = load(work / 'completed.json')
    evaluation = load(work / 'terminal_eval.json')
    exit_record = load(work / 'last_process_exit.json')
    from mmengine.config import Config
    config_dict = Config.fromfile(str(config), import_custom_modules=False).to_dict()
    runtime = runtime_from_config(config_dict)
    expected = dict(protocol=get_experiment(name).protocol, experiment=name, seed=runtime['seed'], runtime=runtime,
                    phase=phase, config_sha256=sha256(config))
    if (any(record.get(k) != v or evaluation.get(k) != v for k, v in expected.items())
        or record['iteration'] != steps or evaluation['iteration'] != steps
        or evaluation['selection'] != 'fixed_terminal' or record['miou'] != evaluation['miou']
        or record.get('macc') != evaluation.get('macc') or record.get('aacc') != evaluation.get('aacc')
        or evaluation.get('evaluator') != 'MMSegmentation.IoUMetric'
        or any(not math.isfinite(float(record.get(k, float('nan')))) or not 0 <= float(record[k]) <= 100
               for k in ('miou', 'macc', 'aacc'))
        or exit_record['returncode'] != 0 or exit_record['config_sha256'] != expected['config_sha256']
        or not Path(exit_record['log']).is_file()
        or sha256(record['checkpoint']) != record['checkpoint_sha256']
        or sha256(work / 'gradient_probe.json') != record['gradient_probe_sha256']
        ):
        raise RuntimeError('Terminal provenance mismatch')
    history_path = work / 'validation_metrics.jsonl'
    try:
        validation_history = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()
                             if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError('Scheduled validation history is missing or malformed') from exc

    val_interval = config_dict['train_cfg']['val_interval']
    expected_validations = ([steps] if phase in ('smoke','engineering_diagnostic')
                            else list(range(val_interval, steps + 1, val_interval)))
    if ([item.get('iteration') for item in validation_history] != expected_validations
        or any(item.get('evaluator') != 'MMSegmentation.IoUMetric'
               or not all(math.isfinite(float(item.get(key, float('nan'))))
                          and 0 <= float(item[key]) <= 100 for key in ('mIoU', 'mAcc', 'aAcc'))
               for item in validation_history)
        or validation_history[-1].get('selection') != 'fixed_terminal'
        or any(validation_history[-1].get(source) != evaluation.get(target)
               for source, target in (('mIoU','miou'), ('mAcc','macc'), ('aAcc','aacc')))):
        raise RuntimeError('Scheduled validation history does not match the protocol/terminal metrics')
    cost = load(work / 'cost_profile.json')
    if (record.get('cost_profile_sha256') != sha256(work / 'cost_profile.json')
        or record.get('cost_profile_status') != cost.get('profiling_status')
                or cost.get('protocol') != expected['protocol'] or cost.get('experiment') != name
        or cost.get('config_sha256') != expected['config_sha256']
        or cost.get('checkpoint_sha256') != record['checkpoint_sha256']
        or cost.get('total_parameters', 0) <= 0
        or cost.get('flops_counted_G') is None
        or not math.isfinite(float(cost['flops_counted_G']))
        or not math.isfinite(float(cost.get('latency_median_ms', float('nan'))))
        or not math.isfinite(float(cost.get('peak_allocated_GB', float('nan'))))):
        raise RuntimeError('Required Params/FLOPs/latency/peak-memory profile is incomplete or mismatched')
    checkpoint = torch.load(record['checkpoint'], map_location='cpu', weights_only=False)
    if checkpoint.get('meta', {}).get('paper') != expected or int(checkpoint.get('meta', {}).get('iter', -1)) != steps:
        raise RuntimeError('Checkpoint metadata mismatch')
    return record


def audit_smoke(output, row):
    name = row['id']
    receipt = load(output / 'smoke_receipts' / (name + '.json'))
    if (receipt.get('status') != 'PASS' or receipt.get('study_sha256') != sha256(output / 'study.json')
        or receipt.get('formal_config_sha256') != row['sha256']
        or receipt.get('runtime') != manifest_runtime(load(output / 'study.json'))
        or receipt.get('experiment') != name or receipt.get('protocol') != get_experiment(name).protocol
        or receipt.get('phase') != 'smoke' or receipt.get('iteration') != 200
        or not receipt.get('checkpoint_sha256') or not receipt.get('smoke_config_sha256')
        or receipt.get('terminal_eval', {}).get('evaluator') != 'MMSegmentation.IoUMetric'
        or receipt.get('terminal_eval', {}).get('selection') != 'fixed_terminal'
        or receipt.get('terminal_eval', {}).get('iteration') != 200
        or len(receipt.get('validation_metrics', [])) != 1
        or receipt.get('validation_metrics', [{}])[0].get('iteration') != 200
        or receipt.get('validation_metrics', [{}])[0].get('evaluator') != 'MMSegmentation.IoUMetric'
        or any(not math.isfinite(float(receipt['terminal_eval'].get(key, float('nan'))))
               or not 0 <= float(receipt['terminal_eval'][key]) <= 100 for key in ('miou','macc','aacc'))
        or receipt.get('cost_profile', {}).get('protocol') != get_experiment(name).protocol
        or receipt.get('cost_profile', {}).get('experiment') != name
        or receipt.get('cost_profile', {}).get('flops_counted_G') is None
        or receipt.get('cost_profile', {}).get('profiling_status') not in ('PASS', 'PARTIAL')
        or receipt.get('gradient_probe', {}).get('finite') is not True
        or receipt.get('gradient_probe', {}).get('connected_trainable_parameters', 0) <= 0
        or receipt.get('initialization', {}).get('phase') != 'smoke'):
        raise RuntimeError('Smoke receipt is not valid for this exact study/config')
    if get_experiment(name).protocol in (RSP_PROTOCOL, CFRT_PROTOCOL):
        diagnostics = receipt.get('region_preservation_diagnostics', [])
        if ([d.get('iteration') for d in diagnostics] != [1,50,100,150,200]
                or not any(d.get('rsp_inner_pixels', 0) > 0 for d in diagnostics)
                or receipt['cost_profile'].get('training_teacher') is not None
                or any(not math.isfinite(float(v)) for d in diagnostics for k,v in d.items()
                       if k.startswith('rsp_') or k.startswith('loss_rsp'))):
            raise RuntimeError('R9-RSP region/gradient smoke evidence is incomplete')
    return receipt['initialization']['shared_state_sha256']


def smoke(output, row, *, allow_sharing=False):
    from mmengine.config import Config
    name = row['id']
    work, path = output / 'smoke' / name, output / 'smoke_configs' / (name + '.py')
    if (output / 'smoke_receipts' / (name + '.json')).exists():
        shared = audit_smoke(output, row)
        shutil.rmtree(work, ignore_errors=True)
        path.unlink(missing_ok=True)
        return shared
    if work.exists() or path.exists():
        raise FileExistsError('Incomplete smoke preserved; register a fresh output, do not fabricate PASS')
    cfg = Config.fromfile(row['config'], import_custom_modules=False).to_dict()
    cfg = smoke_config(cfg, work)
    validate_config(cfg, allow_smoke=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    Config(cfg).dump(str(path))
    launch(path, work, allow_sharing=allow_sharing)
    record = completed(work, path, name, 200, 'smoke')
    receipt = dict(status='PASS', protocol=get_experiment(name).protocol, experiment=name, phase='smoke', iteration=200,
        runtime=manifest_runtime(load(output / 'study.json')),
        study_sha256=sha256(output / 'study.json'), formal_config_sha256=row['sha256'],
        smoke_config_sha256=sha256(path), checkpoint_sha256=record['checkpoint_sha256'],
        terminal_eval=load(work / 'terminal_eval.json'), gradient_probe=load(work / 'gradient_probe.json'),
        validation_metrics=[json.loads(line) for line in (work / 'validation_metrics.jsonl').read_text().splitlines()
                            if line.strip()],
        initialization=load(work / 'initialization.json'),
        cost_profile=load(work / 'cost_profile.json'),
        console_log_sha256=sha256(load(work / 'last_process_exit.json')['log']),
        completed_record_sha256=sha256(work / 'completed.json'))
    if get_experiment(name).protocol in (RSP_PROTOCOL, CFRT_PROTOCOL):
        receipt['region_preservation_diagnostics'] = [json.loads(line) for line in
            (work / 'region_preservation_diagnostics.jsonl').read_text().splitlines() if line.strip()]
    write_once(output / 'smoke_receipts' / (name + '.json'), receipt)
    shutil.rmtree(work)
    path.unlink()
    return audit_smoke(output, row)


def diagnose(output, row, *, allow_sharing=False):
    from mmengine.config import Config
    audit_smoke(output, row)
    work = output / 'engineering' / (row['id'] + '_2000')
    path = output / 'engineering_configs' / (row['id'] + '_2000.py')
    if work.exists() or path.exists():
        raise FileExistsError('Preserve prior diagnostic evidence; use a fresh study for a new attempt')
    cfg = diagnostic_config(Config.fromfile(row['config'], import_custom_modules=False).to_dict(), work)
    validate_config(cfg, allow_diagnostic=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    Config(cfg).dump(str(path))
    launch(path, work, allow_sharing=allow_sharing)
    record = completed(work, path, row['id'], 2000, 'engineering_diagnostic')
    if load(work / 'initialization.json')['shared_state_sha256'] != audit_smoke(output, row):
        raise RuntimeError('Engineering initialization differs from ImageNet smoke initialization')
    write_once(work / 'engineering_receipt.json', dict(record, role='engineering_only_not_paper_result',
        formal_training_steps=row['steps'], initialization='fresh ImageNet; not M16 continuation'))
    return record


SEALED_FILES = ('model.pth', 'config.py', 'reference.json', 'terminal_eval.json', 'validation_metrics.jsonl',
                'training.log', 'initialization.json', 'gradient_probe.json', 'cost_profile.json')


def frozen(output, name):
    folder = output / 'frozen' / name
    seal = load(folder / 'seal.json')
    if set(seal) != set(SEALED_FILES) or any(sha256(folder / p) != digest for p, digest in seal.items()):
        raise RuntimeError('Frozen reference seal mismatch')
    record = load(folder / 'reference.json')
    spec = get_experiment(name)
    runtime = manifest_runtime(load(output / 'study.json'))
    if (record['protocol'] != spec.protocol or record['experiment'] != name or record['seed'] != runtime['seed'] or record.get('runtime') != runtime
        or record['phase'] != 'formal' or record['iteration'] != spec.steps
        or record['study_sha256'] != sha256(output / 'study.json')
        or record['miou'] != load(folder / 'terminal_eval.json')['miou']
        or record['macc'] != load(folder / 'terminal_eval.json')['macc']
        or record['aacc'] != load(folder / 'terminal_eval.json')['aacc']
        or record.get('cost_profile_sha256') != sha256(folder / 'cost_profile.json')
        or load(folder / 'cost_profile.json').get('checkpoint_sha256') != record['checkpoint_sha256']
        or record['checkpoint_sha256'] != sha256(folder / 'model.pth')
        or record['config_sha256'] != sha256(folder / 'config.py')):
        raise RuntimeError('Frozen experiment identity mismatch')
    history = [json.loads(line) for line in (folder / 'validation_metrics.jsonl').read_text().splitlines()
               if line.strip()]
    terminal = load(folder / 'terminal_eval.json')
    if (not history or history[-1].get('iteration') != record['iteration']
        or history[-1].get('evaluator') != 'MMSegmentation.IoUMetric'
        or any(history[-1].get(source) != terminal.get(target)
               for source, target in (('mIoU','miou'), ('mAcc','macc'), ('aAcc','aacc')))):
        raise RuntimeError('Frozen validation history does not match the terminal evaluation')
    return record



def seed_resume_work(output, row, checkpoint, init_digest, *, budget_extension=False):
    """Carry validated provenance through a same-global-batch DDP resize resume."""
    from mmengine.config import Config
    checkpoint = Path(checkpoint).resolve()
    source_work = checkpoint.parent
    source_study = source_work.parent.parent
    source_manifest = load(source_study / 'study.json')
    source_config = source_study / 'configs' / (row['id'] + '.py')
    source_history = source_work / 'validation_metrics.jsonl'
    if not source_config.is_file() or not source_history.is_file():
        raise FileNotFoundError('Resume source config/validation history is missing')
    source_runtime = manifest_runtime(source_manifest)
    target_manifest = load(output / 'study.json')
    target_runtime = manifest_runtime(target_manifest)
    protocols_match = source_manifest['protocol'] == target_manifest['protocol']
    extension_protocols_match = (
        source_manifest['protocol'] == 'DCPA-COMPETITORS-R1-20000'
        and target_manifest['protocol'] == 'DCPA-COMPETITORS-R1-160000')
    if ((not protocols_match and not (budget_extension and extension_protocols_match))
            or source_runtime['seed'] != target_runtime['seed']
            or source_runtime['global_batch'] != target_runtime['global_batch']):
        raise ValueError('Resume provenance must preserve protocol, seed and global batch; '
                         'only the explicit sealed 20K-to-160K extension may change protocol')
    source_row = next(r for r in source_manifest['experiments'] if r['id'] == row['id'])
    source_config_sha = sha256(source_config)
    if source_row['sha256'] != source_config_sha:
        raise RuntimeError('Resume source config no longer matches its study manifest')

    checkpoint_iteration = int(checkpoint.stem.removeprefix('iter_'))
    if budget_extension:
        seal_path = source_study / 'frozen' / row['id'] / 'seal.json'
        frozen_checkpoint = source_study / 'frozen' / row['id'] / 'model.pth'
        seal = load(seal_path)
        if (checkpoint_iteration != 20000 or not frozen_checkpoint.is_file()
                or sha256(checkpoint) != seal.get('model.pth')
                or sha256(checkpoint) != sha256(frozen_checkpoint)):
            raise RuntimeError('Budget extension requires the exact sealed iteration-20000 checkpoint')
    target_config = Config.fromfile(row['config'], import_custom_modules=False).to_dict()
    if budget_extension and (
            target_config['train_cfg']['max_iters'] != 160000
            or target_config['param_scheduler'][1].get('end') != 160000):
        raise ValueError('Budget extension target must use the registered 160K PolyLR config')
    interval = target_config['train_cfg']['val_interval']
    expected_prefix = list(range(interval, checkpoint_iteration + 1, interval))
    history_lines = [line for line in source_history.read_text().splitlines() if line.strip()]
    selected_lines = [line for line in history_lines
                      if int(json.loads(line).get('iteration', -1)) <= checkpoint_iteration]
    selected_iterations = [int(json.loads(line)['iteration']) for line in selected_lines]
    if selected_iterations != expected_prefix:
        raise RuntimeError(f'Resume checkpoint lacks the full validation prefix: {selected_iterations}')

    work = output / 'runs' / row['id']
    if work.exists():
        raise FileExistsError('Resume work directory already exists; preserving it')
    work.mkdir(parents=True)
    with (work / 'validation_metrics.jsonl').open('x', encoding='utf-8') as handle:
        handle.write('\n'.join(selected_lines) + '\n')

    receipt = load(output / 'smoke_receipts' / (row['id'] + '.json'))
    init_record = dict(receipt['initialization'])
    init_record.update(protocol=get_experiment(row['id']).protocol, experiment=row['id'],
        seed=target_runtime['seed'], runtime=target_runtime, phase='formal',
        config_sha256=row['sha256'], shared_state_sha256=init_digest,
        resumed_from_checkpoint_sha256=sha256(checkpoint))
    write_once(work / 'initialization.json', init_record)
    write_once(work / 'gradient_probe.json', receipt['gradient_probe'])
    write_once(work / 'resume.json', dict(
        source_study=str(source_study), source_study_sha256=sha256(source_study / 'study.json'),
        source_config=str(source_config), source_config_sha256=source_config_sha,
        source_runtime=source_runtime, target_runtime=target_runtime,
        source_checkpoint=str(checkpoint), source_checkpoint_sha256=sha256(checkpoint),
        source_iteration=checkpoint_iteration, validation_prefix_iterations=selected_iterations,
        validation_prefix_sha256=sha256(source_history), budget_extension=budget_extension,
        schedule_rebase=('R1 160K PolyLR at global iter 20K; optimizer moments preserved and LR rebased'
                         if budget_extension else None),
        global_batch_preserved=True, model_recipe_unchanged=True))


def run_arm(output, row, resume=None, *, allow_sharing=False, topology_proof=None,
            budget_extension=False):

    name = row['id']
    if (output / 'frozen' / name / 'seal.json').exists():
        if resume:
            raise ValueError('Frozen reference cannot be resumed')
        return frozen(output, name)
    init_digest = audit_smoke(output, row)
    work = output / 'runs' / name

    if topology_proof:
        from .topology import stage_evidence
        stage_evidence(topology_proof, output, row, init_digest)
    elif resume and not (work / 'resume.json').exists():
        seed_resume_work(output, row, resume, init_digest, budget_extension=budget_extension)
    if not (work / 'completed.json').exists():
        launch(Path(row['config']), work, resume, allow_sharing=allow_sharing,
               topology_proof=topology_proof, budget_extension=budget_extension)
    record = completed(work, row['config'], name, row['steps'], 'formal')
    if load(work / 'initialization.json')['shared_state_sha256'] != init_digest:
        raise RuntimeError('Formal initialization differs from the smoke')
    if resume and (work / 'resume.json').is_file():
        record['resume_provenance'] = load(work / 'resume.json')
    target = output / 'frozen' / name
    target.mkdir(parents=True, exist_ok=False)
    shutil.copy2(record['checkpoint'], target / 'model.pth')
    shutil.copy2(row['config'], target / 'config.py')
    for file in ('terminal_eval.json', 'validation_metrics.jsonl', 'initialization.json',
                 'gradient_probe.json', 'cost_profile.json'):
        shutil.copy2(work / file, target / file)
    exit_record = load(work / 'last_process_exit.json')
    shutil.copy2(exit_record['log'], target / 'training.log')
    write_once(target / 'reference.json', dict(record, study_sha256=sha256(output / 'study.json')))
    write_once(target / 'seal.json', {p: sha256(target / p) for p in SEALED_FILES})
    for path in target.iterdir():
        path.chmod(0o444)
    return frozen(output, name)


def check_pair_initializations(output, rows, *, smoke_phase):
    for row in rows:
        reference = get_experiment(row['id']).reference
        if not reference:
            continue
        if smoke_phase:
            left_receipt = output / 'smoke_receipts' / (row['id'] + '.json')
            right_receipt = output / 'smoke_receipts' / (reference + '.json')
            if not left_receipt.is_file() or not right_receipt.is_file():
                continue
            left = audit_smoke(output, row)
            reference_row = next((candidate for candidate in rows if candidate['id'] == reference), None)
            if reference_row is None:
                manifest = load(output / 'study.json')
                reference_row = next(candidate for candidate in manifest['experiments'] if candidate['id'] == reference)
            right = audit_smoke(output, reference_row)
        else:
            left_path = output / 'frozen' / row['id'] / 'initialization.json'
            right_path = output / 'frozen' / reference / 'initialization.json'
            if not left_path.is_file() or not right_path.is_file():
                continue
            left = load(left_path)['shared_state_sha256']
            right = load(right_path)['shared_state_sha256']
        if left != right:
            raise RuntimeError(f'Common initialization mismatch: {row["id"]} vs {reference}')


def _sample_experiment_gpus(output, row, stop, gpu_indices):
    path = output / 'resources' / row['id'] / 'gpu_samples.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('w', newline='') as handle:
            csv.writer(handle).writerow(('time_ns', 'experiment', 'gpu_index', 'memory_used_mib', 'utilization_percent'))
    while not stop.is_set():
        try:
            lines = subprocess.check_output([
                'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).splitlines()
            values = {}
            for line in lines:
                gpu, memory, utilization = [part.strip() for part in line.split(',')]
                if int(gpu) in gpu_indices:
                    values[int(gpu)] = (int(memory), None if utilization in ('[N/A]', 'N/A') else float(utilization))
            now = time.time_ns()
            with path.open('a', newline='') as handle:
                writer = csv.writer(handle)
                for gpu, (memory, utilization) in sorted(values.items()):
                    writer.writerow((now, row['id'], gpu, memory, '' if utilization is None else utilization))
        except Exception as exc:
            with path.open('a') as handle:
                handle.write(f'# sampling_error={type(exc).__name__}: {exc}\n')
        stop.wait(10)


def _summarize_gpu_usage(output, row, *, allow_sharing=False):
    path = output / 'resources' / row['id'] / 'gpu_samples.csv'
    samples = []
    if path.is_file():
        with path.open(newline='') as handle:
            samples = list(csv.DictReader(line for line in handle if not line.startswith('#')))
    by_gpu = {}
    for item in samples:
        by_gpu.setdefault(int(item['gpu_index']), []).append(item)
    per_gpu = []
    for gpu in sorted(by_gpu):
        records = by_gpu.get(gpu, [])
        memories = [int(item['memory_used_mib']) for item in records]
        utilizations = sorted(float(item['utilization_percent']) for item in records
                              if item['utilization_percent'] != '')
        p95 = utilizations[min(len(utilizations)-1, round(.95*(len(utilizations)-1)))] if utilizations else None
        per_gpu.append(dict(gpu_index=gpu, sample_count=len(records),
            training_peak_memory_GiB=max(memories)/1024 if memories else None,
            mean_gpu_utilization_percent=sum(utilizations)/len(utilizations) if utilizations else None,
            p95_gpu_utilization_percent=p95, max_gpu_utilization_percent=max(utilizations) if utilizations else None))
    all_util = [float(item['utilization_percent']) for item in samples if item['utilization_percent'] != '']
    record = dict(protocol=get_experiment(row['id']).protocol, experiment=row['id'], per_gpu=per_gpu,
        gpu_sharing_allowed=allow_sharing, memory_scope='selected_device_total_including_other_processes',
        training_peak_memory_GiB=max((g['training_peak_memory_GiB'] or 0 for g in per_gpu), default=0) or None,
        mean_gpu_utilization_percent=sum(all_util)/len(all_util) if all_util else None,
        p95_gpu_utilization_percent=(sorted(all_util)[min(len(all_util)-1, round(.95*(len(all_util)-1)))]
                                     if all_util else None),
        source='nvidia-smi sampled every 10 seconds on selected GPUs during this experiment')
    (path.parent / 'training_usage.json').write_text(json.dumps(record, indent=2) + '\n')


def sequential_job(output, row, action, *, phase, allow_sharing=False):
    runtime = manifest_runtime(load(output / 'study.json'))
    selected = idle_gpus(runtime['world_size'], allow_sharing=allow_sharing)
    monitor_stop = threading.Event()
    monitor = None
    if phase == 'formal':
        monitor = threading.Thread(target=_sample_experiment_gpus, args=(output, row, monitor_stop, {gpu['index'] for gpu in selected}),
                                   name=f'adpcpf-gpu-monitor-{row["id"]}', daemon=True)
        monitor.start()
    try:
        action(output, row)
        print(f'{row["id"]}: PASS', flush=True)
    except Exception as exc:
        print(f'{row["id"]}: FAIL: {exc!r}', flush=True)
        raise
    finally:
        if monitor is not None:
            monitor_stop.set()
            monitor.join()
            _summarize_gpu_usage(output, row, allow_sharing=allow_sharing)


def summarize(output, manifest):
    check_pair_initializations(output, manifest['experiments'], smoke_phase=False)
    scores, multi_scores, costs, resources, evaluations, continuations = {}, {}, {}, {}, {}, {}
    for row in manifest['experiments']:
        name = row['id']
        folder = output / 'frozen' / name
        if (folder / 'seal.json').exists():
            record = frozen(output, name)
            scores[name] = record['miou']
            costs[name] = load(folder / 'cost_profile.json')
            if record.get('resume_provenance'):
                continuations[name] = record['resume_provenance']
            result_path = output / 'evaluations' / name / 'results.json'
            if result_path.is_file():
                result = load(result_path)
                evaluation_manifest = load(result_path.parent / 'evaluation_manifest.json')
                if (result.get('protocol') != manifest['protocol'] or result.get('experiment') != name
                    or result.get('checkpoint_sha256') != record['checkpoint_sha256']
                    or evaluation_manifest.get('checkpoint_sha256') != record['checkpoint_sha256']
                    or evaluation_manifest.get('source_study_sha256') != sha256(output / 'study.json')):
                    raise RuntimeError(f'{name}: SS/MS evaluation does not match this sealed checkpoint')
                scales = result.get('scales', {})
                ss_score = scales.get('SS', {}).get('mIoU')
                if ss_score is not None and abs(float(ss_score) - float(record['miou'])) > .02:
                    raise RuntimeError(f'{name}: offline SS mIoU differs from fixed terminal IoUMetric')
                multi_scores[name] = scales.get('MS', {}).get('mIoU')
                evaluations[name] = result
            else:
                multi_scores[name] = None
                evaluations[name] = None
        else:
            scores[name], costs[name] = None, None
            multi_scores[name], evaluations[name] = None, None
        usage = output / 'resources' / name / 'training_usage.json'
        resources[name] = load(usage) if usage.is_file() else None
    lines = ['| ID | Method | Dataset | Backbone | Steps | SI | SS mIoU | MS mIoU | mAcc | aAcc | Params (M) | FLOPs (G) | FLOP coverage | Latency P50/P90 (ms) | Infer peak (GB) | Train peak (GiB) | GPU util avg/P95 (%) | Reference | ΔSS (pp) |',
             '|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|']
    for name, score in scores.items():
        e = get_experiment(name)
        peer = scores.get(e.reference)
        delta = '' if score is None or peer is None else f'{score-peer:.6f}'
        cost = costs[name] or {}
        usage = resources[name] or {}
        params = '' if not cost.get('total_parameters') else f'{cost["total_parameters"]/1e6:.2f}'
        flops = '' if cost.get('flops_counted_G') is None else f'{cost["flops_counted_G"]:.2f}'
        flop_coverage = cost.get('flop_status', '')
        latency = '' if not cost.get('latency_median_ms') else f'{cost["latency_median_ms"]:.3f}/{cost["latency_p90_ms"]:.3f}'
        inference_peak = '' if not cost.get('peak_allocated_GB') else f'{cost["peak_allocated_GB"]:.2f}'
        train_peak = '' if usage.get('training_peak_memory_GiB') is None else f'{usage["training_peak_memory_GiB"]:.2f}'
        gpu_util = '' if usage.get('mean_gpu_utilization_percent') is None else f'{usage["mean_gpu_utilization_percent"]:.1f}/{usage["p95_gpu_utilization_percent"]:.1f}'
        terminal_path = output / 'frozen' / name / 'terminal_eval.json'
        terminal = load(terminal_path) if terminal_path.is_file() else {}
        ss_eval = evaluations[name].get('scales', {}).get('SS', {}) if evaluations[name] else {}
        # The fixed terminal value is the official single-scale IoUMetric result.
        # evaluate.py's offline confusion matrix is a per-class cross-check only.
        ss_miou = score
        ms_miou = multi_scores[name]
        lines.append(f'| {name} | {e.method}/{e.intervention} | {e.dataset} | {e.backbone} | {e.steps} | '
                     f'{e.si} | {"" if ss_miou is None else format(float(ss_miou),".2f")} | '
                     f'{"" if ms_miou is None else format(float(ms_miou),".2f")} | '
                     f'{terminal.get("macc", "")} | {terminal.get("aacc", "")} | {params} | {flops} | {flop_coverage} | '
                     f'{latency} | {inference_peak} | {train_peak} | {gpu_util} | {e.reference or ""} | {delta} |')
    if continuations:
        lines.extend(['', 'Checkpoint continuation provenance:'])
        for name, provenance in continuations.items():
            lines.append(f'- {name}: source iteration {provenance["source_iteration"]} from '
                         f'{provenance["source_study"]}; optimizer moments preserved and 160K PolyLR rebased.')
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')
    (output / 'summary.json').write_text(json.dumps(dict(protocol=manifest['protocol'], runtime=manifest_runtime(manifest), seed=manifest['seed'], scores=scores,
        multi_scale_scores=multi_scores, evaluations=evaluations,
        cost_profiles=costs, training_gpu_resources=resources,
        continuation_provenance=continuations), indent=2) + '\n')
    from .paper_matrix import render_tables
    paper_tables = render_tables(manifest, scores, multi_scores, costs, evaluations)
    if continuations:
        note = ('> Includes checkpoint continuation from the 20K screening study; this is not an independent '
                '160K-from-ImageNet run.\n\n')
        paper_tables = note + paper_tables
    (output / 'paper_tables.md').write_text(paper_tables)
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list','check','prepare','verify','smoke','diagnose','run','resume','all','summarize'))
    parser.add_argument('--experiments', nargs='+', help='IDs/groups; omitted prepare= current, omitted run= prepared plan')
    parser.add_argument('--seed', type=int, default=None, help='Random seed; new studies default to 1')
    parser.add_argument('--output', type=Path, help='Locked study directory; default includes runtime parameters')
    parser.add_argument('--ade-root', type=Path, default=ROOT.parent / 'benchmark/ADEChallengeData2016')
    parser.add_argument('--context-root', type=Path)
    parser.add_argument('--pretrained-s', type=Path, default=Path(os.environ['DCPA_CHECKPOINT']) if 'DCPA_CHECKPOINT' in os.environ else None)
    parser.add_argument('--pretrained-b', type=Path)
    parser.add_argument('--pretrained-b-sha256')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--extend-20k-checkpoint', action='store_true',
                        help='Explicitly continue a sealed R1 20K checkpoint to a total of 160K steps')
    parser.add_argument('--topology-parent-study', type=Path,
                        help='Explicit same-global-batch continuation from another DDP layout')
    parser.add_argument('--world', '--world-size', '--gpus-per-experiment', dest='world_size', type=int,
                        help='DDP process/GPU count (default 8)')
    parser.add_argument('--batch', '--per-gpu-batch', '--batch-size', dest='per_gpu', type=int,
                        help='Images per GPU (default 2; inferred if --global-batch is supplied)')
    parser.add_argument('--global-batch', type=int, help='Total images per optimizer step: world × batch')
    parser.add_argument('--share-gpus', action='store_true',
                        help='Explicitly share selected GPUs with existing processes; never stop them')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if (args.command in ('prepare','smoke','diagnose','run','resume','all')
            and args.experiments and 'M23' in expand(args.experiments,with_references=False)):
        parser.error('Historical M23-CFRT training entrypoint is retired; use competitors.run plain_primary/plain_ablations.')
    if args.topology_parent_study is not None and args.command != 'resume':
        parser.error('--topology-parent-study is only valid for resume')
    if args.extend_20k_checkpoint and args.command != 'resume':
        parser.error('--extend-20k-checkpoint is only valid with resume')
    smoke_action = partial(smoke, allow_sharing=args.share_gpus)
    run_action = partial(run_arm, allow_sharing=args.share_gpus,
                         budget_extension=args.extend_20k_checkpoint)
    try:
        runtime = resolve_runtime(args.world_size, args.per_gpu, args.global_batch, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output is None:
        selected_protocol = study_protocol(expand(args.experiments or ['current']))
        output_names = {FDAM_BASELINE_PROTOCOL: 'FDAM-DEIT-S-UPERNET-ADE20K-160K-REPLICA-V1',
                        R9_PROTOCOL: 'ADPCPF-PAPER-V3-R9-PYRAMID',
                        R10_PROTOCOL: 'ADPCPF-PAPER-V4-R10-NBLF',
                        RSP_PROTOCOL: 'ADPCPF-PAPER-V6-R9-RSP'}
        base_output = DEFAULT_OUTPUT.with_name(output_names[selected_protocol]) if selected_protocol in output_names else DEFAULT_OUTPUT
        args.output = base_output.with_name(base_output.name +
            f"-w{runtime['world_size']}-b{runtime['per_gpu']}-g{runtime['global_batch']}-s{runtime['seed']}")
    if args.command == 'list':
        for name, record in HISTORICAL.items():
            print(json.dumps(dict(id=name, **record, active=False,
                output=str(ROOT.parent / 'trained_models/ADPCPF-PAPER-V2-R3-MT8-IOU-COST-CPUFLOPS'),
                note='Do not retrain in R7; checkpoint and evaluation receipts are preserved.'),
                ensure_ascii=False))
        for e in EXPERIMENTS.values():
            print(json.dumps(dict(asdict(e), protocol=e.protocol, steps=e.steps, crop=e.crop,
                                  **runtime, active=True), ensure_ascii=False))
        return
    if args.command in ('smoke','diagnose','run','resume','all') and not args.execute:
        parser.error('GPU commands require explicit --execute; use list/prepare/verify first')
    activate()
    if args.command == 'check':
        subprocess.run([sys.executable,'-m','pytest','-q',str(Path(__file__).parent)], cwd=ROOT, check=True)
        return
    output = args.output.resolve()
    with study_lock(output):
        manifest = prepare(args, output) if args.command in ('prepare','all') else verify(output)
        check_runtime_args(args, manifest)
        names = expand(args.experiments, with_references=False) if args.experiments else [r['id'] for r in manifest['experiments']]
        rows = [r for r in manifest['experiments'] if r['id'] in names]
        if set(names) != {r['id'] for r in rows}:
            raise ValueError('Requested experiment not in this locked study')
        if args.command in ('prepare','smoke','diagnose','run','resume','all') and 'M23' in names:
            raise ValueError('Historical M23-CFRT is read-only. Use competitors.run with plain_primary or plain_ablations.')
        if args.command == 'all':
            rows = manifest['experiments']
        if args.command == 'smoke':
            for row in rows:
                sequential_job(output, row, smoke_action, phase='smoke', allow_sharing=args.share_gpus)
                check_pair_initializations(output, manifest['experiments'], smoke_phase=True)
        if args.command == 'diagnose':
            if len(rows) != 1 or rows[0]['id'] != 'M21':
                raise ValueError('diagnose requires exactly --experiments M21')
            sequential_job(output, rows[0], partial(diagnose, allow_sharing=args.share_gpus),
                           phase='engineering_diagnostic', allow_sharing=args.share_gpus)
        if args.command == 'resume' and (len(rows) != 1 or args.checkpoint is None):
            parser.error('resume requires exactly one --experiments ID and --checkpoint')
        if args.command == 'run':
            for row in rows:
                sequential_job(output, row, run_action, phase='formal', allow_sharing=args.share_gpus)
                check_pair_initializations(output, manifest['experiments'], smoke_phase=False)
        if args.command == 'resume':
            proof = None
            if args.topology_parent_study is not None:
                from .topology import create_proof
                proof = create_proof(args.topology_parent_study, output, rows[0], args.checkpoint)
            sequential_job(output, rows[0], lambda study, row: run_action(study, row, args.checkpoint,
                                                                        topology_proof=proof),
                           phase='formal', allow_sharing=args.share_gpus)
            check_pair_initializations(output, manifest['experiments'], smoke_phase=False)
        if args.command == 'all':
            for row in rows:
                sequential_job(output, row, smoke_action, phase='smoke', allow_sharing=args.share_gpus)
                check_pair_initializations(output, manifest['experiments'], smoke_phase=True)
                sequential_job(output, row, run_action, phase='formal', allow_sharing=args.share_gpus)
                check_pair_initializations(output, manifest['experiments'], smoke_phase=False)
        if args.command in ('summarize','run','resume','all'):
            summarize(output, manifest)
        if args.command == 'verify':
            print('SOURCE_CONFIG_DATA_PRETRAIN_ENVIRONMENT_VERIFY_PASS')


if __name__ == '__main__':
    main()
