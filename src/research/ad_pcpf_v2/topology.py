"""Audited same-budget continuation across DDP layouts; never edits a checkpoint."""
from copy import deepcopy
import json
from pathlib import Path

from .artifacts import identity, load, sha256, write_once


INFRASTRUCTURE = {'v2/run.py', 'v2/engine.py', 'v2/topology.py', 'v2/test_topology.py'}


def scientific_config(cfg):
    value = deepcopy(cfg)
    value.pop('work_dir')
    value['train_dataloader'].pop('batch_size')
    for key in ('world_size', 'per_gpu'):
        value['paper']['runtime'].pop(key)
    return value


def check_layout_change(source, target):
    if scientific_config(source) != scientific_config(target):
        raise ValueError('Topology continuation cannot change the scientific recipe or global batch')
    left, right = source['paper']['runtime'], target['paper']['runtime']
    if (left['global_batch'] != right['global_batch'] or left['seed'] != right['seed']
            or left['world_size'] * left['per_gpu'] != left['global_batch']
            or right['world_size'] * right['per_gpu'] != right['global_batch']):
        raise ValueError('Topology change must preserve global batch and seed')
    if (left['world_size'], left['per_gpu']) == (right['world_size'], right['per_gpu']):
        raise ValueError('Use ordinary resume for an unchanged layout')


def create_proof(parent, target, row, checkpoint):
    import torch
    from mmengine.config import Config
    parent, target, checkpoint = Path(parent).resolve(), Path(target).resolve(), Path(checkpoint).resolve()
    old, new = load(parent / 'study.json'), load(target / 'study.json')
    prior = next(r for r in old['experiments'] if r['id'] == row['id'])
    if old['protocol'] != new['protocol'] or old['datasets'] != new['datasets'] or old['pretrained'] != new['pretrained']:
        raise ValueError('Continuation protocol/data/pretrained identity differs')
    changed = {k for k in old['sources'].keys() | new['sources'].keys()
               if old['sources'].get(k) != new['sources'].get(k)}
    if changed - INFRASTRUCTURE:
        raise ValueError(f'Model/training-math source drift: {sorted(changed - INFRASTRUCTURE)}')
    if sha256(prior['config']) != prior['sha256'] or sha256(row['config']) != row['sha256']:
        raise ValueError('Config hash mismatch')
    source_cfg = Config.fromfile(prior['config'], import_custom_modules=False).to_dict()
    target_cfg = Config.fromfile(row['config'], import_custom_modules=False).to_dict()
    check_layout_change(source_cfg, target_cfg)
    if not checkpoint.is_relative_to(parent / 'runs' / row['id']):
        raise ValueError('Checkpoint must belong to the named parent experiment')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck.get('meta', {}).get('paper') != identity(Config(source_cfg), prior['sha256']):
        raise ValueError('Parent checkpoint identity does not match its original config')
    step = int(ck['meta']['iter'])
    if not 200 < step < target_cfg['train_cfg']['max_iters']:
        raise ValueError('Invalid continuation step')
    if not ck.get('optimizer') or not ck.get('param_schedulers'):
        raise ValueError('Full optimizer and scheduler states are required')
    del ck
    work = parent / 'runs' / row['id']
    files = {n: dict(path=str(work / n), sha256=sha256(work / n)) for n in
             ('initialization.json', 'gradient_probe.json', 'validation_metrics.jsonl')}
    proof = dict(kind='DDP_TOPOLOGY_CONTINUATION', experiment=row['id'], iteration=step,
        parent_study=str(parent), parent_study_sha256=sha256(parent / 'study.json'),
        target_study=str(target), target_study_sha256=sha256(target / 'study.json'),
        parent_config=dict(path=prior['config'], sha256=prior['sha256']),
        target_config=dict(path=row['config'], sha256=row['sha256']),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        source_runtime=source_cfg['paper']['runtime'], target_runtime=target_cfg['paper']['runtime'],
        retained_state=['model', 'optimizer', 'param_schedulers', 'message_hub'],
        source_infrastructure_changes=sorted(changed), ancestral_evidence=files,
        verifier_sha256=sha256(__file__),
        limitation='Rank partition and sample ordering may change; no bitwise trajectory equivalence claim.')
    path = target / 'topology_resumes' / (row['id'] + '.json')
    if path.exists():
        if load(path) != proof:
            raise ValueError('Existing topology proof differs; preserve it')
    else:
        write_once(path, proof)
        path.chmod(0o444)
    return path


def verify_proof(path, config_path, checkpoint, meta):
    from mmengine.config import Config
    proof = load(path)
    if (proof['kind'] != 'DDP_TOPOLOGY_CONTINUATION'
            or proof['verifier_sha256'] != sha256(__file__)
            or proof['target_config']['sha256'] != sha256(config_path)
            or Path(proof['target_config']['path']).resolve() != Path(config_path).resolve()
            or proof['checkpoint_sha256'] != sha256(checkpoint)
            or Path(proof['checkpoint']).resolve() != Path(checkpoint).resolve()
            or proof['iteration'] != int(meta['iter'])):
        raise ValueError('Topology proof/checkpoint/config mismatch')
    for key in ('parent_study', 'target_study'):
        if sha256(Path(proof[key]) / 'study.json') != proof[key + '_sha256']:
            raise ValueError('Study manifest changed')
    src = proof['parent_config']
    if sha256(src['path']) != src['sha256']:
        raise ValueError('Parent config changed')
    source_cfg = Config.fromfile(src['path'], import_custom_modules=False)
    target_cfg = Config.fromfile(config_path, import_custom_modules=False)
    check_layout_change(source_cfg.to_dict(), target_cfg.to_dict())
    if meta.get('paper') != identity(source_cfg, src['sha256']):
        raise ValueError('Original checkpoint metadata was altered')
    return proof


def stage_evidence(proof_path, output, row, smoke_digest):
    """Preserve ancestral history, explicitly marked as such, in the new stage."""
    from mmengine.config import Config
    proof = load(proof_path)
    for item in proof['ancestral_evidence'].values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Ancestral evidence changed')
    original = load(proof['ancestral_evidence']['initialization.json']['path'])
    if original['shared_state_sha256'] != smoke_digest:
        raise ValueError('New-layout smoke did not reproduce the original initialization')
    work = Path(output) / 'runs' / row['id']
    work.mkdir(parents=True, exist_ok=False)
    cfg = Config.fromfile(row['config'], import_custom_modules=False)
    write_once(work / 'initialization.json', dict(**identity(cfg, row['sha256']),
        shared_state_sha256=smoke_digest, initialization_kind='ancestral_fresh_initialization',
        ancestral_record=proof['ancestral_evidence']['initialization.json'],
        ancestral_identity=original, zero_residual_check='NOT_REPEATED_ON_TRAINED_WEIGHTS'))
    gradient = load(proof['ancestral_evidence']['gradient_probe.json']['path'])
    write_once(work / 'gradient_probe.json', dict(gradient,
        evidence_kind='ancestral_first200_gradients', ancestral_runtime=proof['source_runtime'],
        current_layout_validated_by_smoke_receipt=str(Path(output) / 'smoke_receipts' / (row['id']+'.json'))))
    history = [json.loads(line) for line in
               Path(proof['ancestral_evidence']['validation_metrics.jsonl']['path']).read_text().splitlines()
               if line.strip()]
    history = [entry for entry in history if entry['iteration'] <= proof['iteration']]
    if [entry['iteration'] for entry in history] != list(range(20000, proof['iteration']+1, 20000)):
        raise ValueError('Missing ancestral scheduled validation')
    (work / 'validation_metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in history))
    write_once(work / 'topology_continuation.json', proof)


def attach_lineage(runner, proof):
    from mmengine.hooks import Hook

    class TopologyLineageHook(Hook):
        def before_save_checkpoint(self, runner, checkpoint):
            checkpoint.setdefault('meta', {})['topology_continuation'] = deepcopy(proof)

    runner.register_hook(TopologyLineageHook(), priority='LOW')
