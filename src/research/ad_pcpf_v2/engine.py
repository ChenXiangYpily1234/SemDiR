"""Independent MMSeg worker/audit hooks. No V1 imports and no implicit GPU launch."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.dist import get_rank, get_world_size
from .artifacts import sha256, load, write_once, shared_digest, identity
from .config import validate_config
from .experiments import PROTOCOL
from .runtime import runtime_from_config


def unwrap(model):
    return model.module if hasattr(model, 'module') else model


def _normalized_resume_config(cfg):
    value = cfg.to_dict()
    value.pop('paper_config_sha256', None)
    value['work_dir'] = '<work_dir>'
    value['train_dataloader']['batch_size'] = '<per_gpu_batch>'
    runtime = runtime_from_config(cfg)
    value['paper']['runtime'] = dict(seed=runtime['seed'], global_batch=runtime['global_batch'])
    return value


def _resume_config_matches(old_cfg, new_cfg):
    """Allow a checkpoint to resume on a different DDP layout at same global batch."""
    old_runtime = runtime_from_config(old_cfg)
    new_runtime = runtime_from_config(new_cfg)
    if (old_runtime['seed'] != new_runtime['seed']
            or old_runtime['global_batch'] != new_runtime['global_batch']):
        return False
    return _normalized_resume_config(old_cfg) == _normalized_resume_config(new_cfg)


def _screen_extension_config_matches(old_cfg, new_cfg):
    """Only permit the registered 20K-to-160K PolyLR checkpoint extension."""
    old_runtime = runtime_from_config(old_cfg)
    new_runtime = runtime_from_config(new_cfg)
    if (old_runtime['seed'] != new_runtime['seed']
            or old_runtime['global_batch'] != new_runtime['global_batch']):
        return False
    old = _normalized_resume_config(old_cfg)
    new = _normalized_resume_config(new_cfg)
    if (old['paper'].get('budget') != 20000
            or new['paper'].get('budget') != 160000
            or old['paper'].get('protocol') != 'DCPA-COMPETITORS-R1-20000'
            or new['paper'].get('protocol') != 'DCPA-COMPETITORS-R1-160000'
            or old['train_cfg'].get('max_iters') != 20000
            or new['train_cfg'].get('max_iters') != 160000):
        return False
    old_schedulers = old.get('param_scheduler', [])
    new_schedulers = new.get('param_scheduler', [])
    if (len(old_schedulers) != 2 or len(new_schedulers) != 2
            or old_schedulers[1].get('type') != 'PolyLR'
            or new_schedulers[1].get('type') != 'PolyLR'
            or old_schedulers[1].get('end') != 20000
            or new_schedulers[1].get('end') != 160000):
        return False
    for value in (old, new):
        value['paper']['budget'] = '<screen-extension>'
        value['paper']['protocol'] = '<screen-extension>'
        value['train_cfg']['max_iters'] = '<screen-extension>'
        value['param_scheduler'][1]['end'] = '<screen-extension>'
    return old == new


def _validate_screen_extension_source(cfg):
    """Validate a sealed R1-20K config against its own registered budget.

    Continuation is launched from an R1-160K CLI process, so the competitor
    registry is otherwise already bound to 160K when it reconstructs the
    source config. Temporarily restore the source registry budget for this
    validation only; the target config and training process remain 160K.
    """
    if cfg.get('paper', {}).get('budget') != 20000:
        raise ValueError('Expected a registered R1 20K source configuration')
    import competitors.registry as registry
    import competitors.registration as registration
    budget_names = ((registry, 'BUDGET', 20000),
                    (registry, 'PROTOCOL', 'DCPA-COMPETITORS-R1-20000'),
                    (registration, 'BUDGET', 20000),
                    (registration, 'PROTOCOL', 'DCPA-COMPETITORS-R1-20000'))
    previous = [(module, name, getattr(module, name)) for module, name, _ in budget_names]
    try:
        for module, name, value in budget_names:
            setattr(module, name, value)
        validate_config(cfg.to_dict())
    finally:
        for module, name, value in previous:
            setattr(module, name, value)


def _rebase_screening_checkpoint(ck, cfg):
    """Keep model/optimizer moments and continue the 160K PolyLR at global iter 20K."""
    start_iter = int(ck.get('meta', {}).get('iter', 0))
    target_iters = int(cfg.train_cfg.max_iters)
    if start_iter != 20000 or target_iters != 160000:
        raise ValueError('Screening continuation must start at iter 20000 and end at iter 160000')
    schedulers = ck.get('param_schedulers')
    if not isinstance(schedulers, list) or len(schedulers) != 2:
        raise ValueError('Screening continuation expects the locked LinearLR + PolyLR scheduler pair')
    groups = ck.get('optimizer', {}).get('param_groups', [])
    if not groups:
        raise ValueError('Screening checkpoint has no optimizer parameter groups')
    group_base_values = []
    for group in groups:
        initial = group.get('initial_lr', group.get('lr'))
        if not isinstance(initial, (int, float)) or not math.isfinite(float(initial)):
            raise ValueError('Screening checkpoint has an invalid optimizer base learning rate')
        group_base_values.append(float(initial))
    for scheduler in schedulers:
        base_values = scheduler.get('base_values')
        if (scheduler.get('param_name') != 'lr' or not isinstance(base_values, list)
                or len(base_values) not in (len(groups), len(groups) + 1)):
            raise ValueError('Screening checkpoint optimizer/scheduler groups do not match')
        # MMEngine R1 checkpoints retain one extra scheduler base entry, while
        # optimizer.state_dict() contains only the live parameter groups. R1
        # uses a single LR across every group, so verify the extra entry is
        # identical, then normalize scheduler state to the live groups.
        if len(base_values) == len(groups) + 1:
            all_values = [float(value) for value in base_values]
            if (not all(math.isfinite(value) for value in all_values)
                    or any(not math.isclose(value, all_values[0], rel_tol=1e-12, abs_tol=0.)
                           for value in all_values[1:])
                    or any(not math.isclose(value, all_values[0], rel_tol=1e-12, abs_tol=0.)
                           for value in group_base_values)):
                raise ValueError('Cannot map unequal scheduler groups to the sealed optimizer groups')
        elif any(not math.isclose(float(value), group_base_values[index], rel_tol=1e-12, abs_tol=0.)
                 for index, value in enumerate(base_values)):
            raise ValueError('Checkpoint scheduler bases differ from optimizer group learning rates')
        scheduler['base_values'] = group_base_values.copy()
        last_values = scheduler.get('_last_value')
        if isinstance(last_values, list) and len(last_values) in (len(groups), len(groups) + 1):
            scheduler['_last_value'] = last_values[:len(groups)]
    schedule = cfg.param_scheduler[1]
    begin, end = int(schedule['begin']), int(schedule['end'])
    total_iters = end - begin - 1
    last_step = start_iter - begin - 1
    if schedule.get('type') != 'PolyLR' or total_iters <= 0 or not 0 <= last_step < total_iters:
        raise ValueError('Invalid target PolyLR schedule for screening continuation')
    state = schedulers[1]
    base_values = state['base_values']
    power, eta_min = float(schedule.get('power', 1.0)), float(schedule.get('eta_min', 0.0))
    remaining_ratio = (total_iters - last_step) / total_iters
    learning_rates = [eta_min + (float(base) - eta_min) * remaining_ratio ** power
                      for base in base_values]
    state.update(begin=begin, end=end, total_iters=total_iters, last_step=last_step,
                 _global_step=start_iter, _last_value=learning_rates,
                 power=power, eta_min=eta_min)
    for group, learning_rate in zip(groups, learning_rates):
        group['lr'] = learning_rate
    ck.setdefault('meta', {})['screening_budget_extension'] = dict(
        source_iteration=start_iter, target_iterations=target_iters,
        scheduler='PolyLR continuation on the locked 160K curve',
        optimizer_moments_preserved=True, optimizer_lr_rebased=True,
        learning_rates=learning_rates)
    return learning_rates


@HOOKS.register_module()
class PaperAuditHook(Hook):
    def before_train(self, runner):
        world = get_world_size()
        batch = runner.cfg.train_dataloader.batch_size
        runtime = runtime_from_config(runner.cfg)
        if (world != runtime['world_size'] or batch != runtime['per_gpu']
                or batch * world != runtime['global_batch']):
            raise RuntimeError(f'Actual DDP layout disagrees with registered runtime: {runtime}')
        model = unwrap(runner.model)
        self.handles, self.seen = [], set()
        self.finite = torch.ones((), dtype=torch.bool, device=next(model.parameters()).device)
        if not runner.cfg.resume:
            digest = shared_digest(model)
            zero_check = getattr(model.backbone, 'check_zero_initialization', None)
            if callable(zero_check):
                zero_check(torch.zeros(1, 3, 65, 79, device=self.finite.device))
            method = getattr(model.backbone, 'method', 'vanilla')
            if get_rank() == 0:
                write_once(Path(runner.work_dir) / 'initialization.json',
                    dict(**identity(runner.cfg, runner.cfg.paper_config_sha256), shared_state_sha256=digest,
                         zero_residual_check='PASS' if method in ('adpcpf', 'adpcpf_r9') else 'not_applicable'))
        if runner.iter < 200:
            self.expected = {n for n, p in model.named_parameters() if p.requires_grad}
            def record(name):
                def hook(gradient):
                    self.seen.add(name)
                    self.finite.logical_and_(torch.isfinite(gradient).all())
                    return gradient
                return hook
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    self.handles.append(parameter.register_hook(record(name)))

    def before_save_checkpoint(self, runner, checkpoint):
        checkpoint.setdefault('meta', {})['paper'] = identity(runner.cfg, runner.cfg.paper_config_sha256)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        model = unwrap(runner.model)
        if hasattr(model, 'set_region_preservation_step'):
            model.set_region_preservation_step(runner.iter + 1)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        values = [v.detach().reshape(-1) for v in (outputs or {}).values() if isinstance(v,torch.Tensor)]
        if values and not bool(torch.isfinite(torch.cat(values)).all()):
            raise FloatingPointError('Non-finite train output')
        iteration = runner.iter + 1
        if torch.cuda.is_available() and iteration % 50 == 0:
            torch.cuda.synchronize()
            runner.message_hub.update_scalar('memory', torch.cuda.max_memory_allocated() / (1024 ** 2))
            torch.cuda.reset_peak_memory_stats()
        model = unwrap(runner.model)
        if iteration == 200:
            missing = sorted(self.expected - self.seen)
            if missing or not bool(self.finite):
                raise RuntimeError(f'Invalid gradient graph/values; missing={missing}')
            relay = getattr(model.backbone, 'relay', None)
            updated = {}
            if relay is not None:
                for name in ('coarse_out', 'detail_out'):
                    module = getattr(relay, name, None)
                    if module is not None:
                        branches = list(module) if isinstance(module, torch.nn.ModuleList) else [module]
                        for index, branch in enumerate(branches):
                            value = float(branch.weight.detach().float().abs().sum())
                            key = f'{name}.{index}' if isinstance(module, torch.nn.ModuleList) else name
                            if not math.isfinite(value) or value <= 0:
                                raise RuntimeError(f'{key} failed to learn in first 200 steps')
                            updated[key] = value
            if get_rank() == 0:
                write_once(Path(runner.work_dir) / 'gradient_probe.json',
                           dict(iteration=iteration, connected_trainable_parameters=len(self.seen),
                                finite=True, projection_update_l1=updated))
            for handle in self.handles:
                handle.remove()
            self.handles = []
        if hasattr(model, 'region_preservation') and (iteration == 1 or iteration % 50 == 0) and get_rank() == 0:
            diagnostics = {key: float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)
                for key, value in (outputs or {}).items() if key.startswith('rsp_') or key == 'loss_rsp'}
            if 'rsp_margin_raw' in diagnostics and 'rsp_weight' in diagnostics:
                diagnostics['loss_rsp_global'] = diagnostics['rsp_margin_raw'] * diagnostics['rsp_weight']
            with (Path(runner.work_dir) / 'region_preservation_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(dict(iteration=iteration, **diagnostics), allow_nan=False)+'\n')
        if iteration == 200 or iteration % 20000 == 0:
            relay = getattr(model.backbone, 'relay', None)
            if relay is not None and get_rank() == 0:
                entry = {k: v.detach().cpu().tolist() for k, v in relay.last_diagnostics.items()}
                with (Path(runner.work_dir) / 'depth_diagnostics.jsonl').open('a') as f:
                    f.write(json.dumps(dict(iteration=iteration, **entry), allow_nan=False) + '\n')

    def after_val_epoch(self, runner, metrics=None):
        values = {key: float((metrics or {})[key]) for key in ('mIoU', 'mAcc', 'aAcc')}
        if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values.values()):
            raise RuntimeError('Invalid scheduled IoUMetric values')
        per_class_iou = {}
        for key, value in (metrics or {}).items():
            if str(key).startswith('IoU.'):
                numeric = float(value)
                per_class_iou[str(key)[4:]] = numeric if math.isfinite(numeric) else None
        record = dict(**identity(runner.cfg, runner.cfg.paper_config_sha256),
            iteration=runner.iter, **{key: values[key] for key in ('mIoU', 'mAcc', 'aAcc')},
            per_class_IoU=per_class_iou, units='percent',
            selection='fixed_terminal' if runner.iter == runner.max_iters else 'scheduled_validation',
            evaluator='MMSegmentation.IoUMetric', primary_metric='mIoU',
            auxiliary_metrics=['mAcc', 'aAcc'])
        if get_rank() == 0:
            work = Path(runner.work_dir)
            history = work / 'validation_metrics.jsonl'
            with history.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, allow_nan=False) + '\n')
            if runner.iter == runner.max_iters:
                write_once(work / 'terminal_eval.json', dict(**identity(runner.cfg, runner.cfg.paper_config_sha256),
                    iteration=runner.iter, miou=values['mIoU'], macc=values['mAcc'], aacc=values['aAcc'],
                    per_class_IoU=per_class_iou, units='percent', selection='fixed_terminal',
                    evaluator='MMSegmentation.IoUMetric', primary_metric='mIoU',
                    auxiliary_metrics=['mAcc', 'aAcc']))


def main():
    import sys
    from mmengine.config import Config
    from .bootstrap import check_loaded_paths
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--resume-from')
    parser.add_argument('--topology-proof')
    parser.add_argument('--extend-20k-checkpoint', action='store_true',
                        help='Continue a sealed R1 20K checkpoint to R1 160K with the 160K PolyLR schedule')
    args = parser.parse_args()
    check_loaded_paths()
    cfg = Config.fromfile(args.config)
    validate_config(cfg.to_dict(), allow_smoke=True, allow_diagnostic=True)
    # MMEngine imports every strategy when importing Runner, including optional
    # DeepSpeed. This paper protocol uses torchrun/DDP; a broken optional DeepSpeed
    # install must not prevent the supported runner from starting.
    strategy = cfg.get('strategy')
    if isinstance(strategy, dict) and strategy.get('type') == 'DeepSpeedStrategy':
        raise ValueError('AD-PCPF PAPER V2 R7 does not support DeepSpeedStrategy')
    try:
        import deepspeed  # noqa: F401
    except Exception as exc:
        print(f'Optional DeepSpeed is unavailable; continuing with MMEngine DDP runner: {type(exc).__name__}: {exc}',
              file=sys.stderr, flush=True)
        sys.modules['deepspeed'] = None
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=False)
    from mmengine.runner import Runner
    cfg.paper_config_sha256 = sha256(args.config)
    expected = identity(cfg, cfg.paper_config_sha256)
    continuation = None
    if args.extend_20k_checkpoint and (not args.resume_from or args.topology_proof):
        raise ValueError('--extend-20k-checkpoint requires --resume-from and cannot be combined with topology proof')
    if args.topology_proof and not args.resume_from:
        raise ValueError('Topology continuation requires --resume-from')
    if args.resume_from:
        if cfg.paper.smoke:
            raise ValueError('Incomplete smoke must be preserved and rerun in a new study, not resumed')
        ck = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        meta = ck.get('meta', {})

        checkpoint_path = Path(args.resume_from).resolve()
        load_checkpoint_path = checkpoint_path
        if args.extend_20k_checkpoint:
            source_config = checkpoint_path.parent.parent.parent / 'configs' / (cfg.paper.experiment + '.py')
            if not source_config.is_file():
                raise ValueError('Screening checkpoint has no registered source config')
            source_cfg = Config.fromfile(str(source_config), import_custom_modules=False)
            _validate_screen_extension_source(source_cfg)
            source_identity = identity(source_cfg, sha256(source_config))
            if (meta.get('paper') != source_identity
                    or not _screen_extension_config_matches(source_cfg, cfg)):
                raise ValueError('20K extension requires the same experiment, model/data recipe, seed and global batch; '
                                 'only the registered 20K-to-160K PolyLR budget extension is allowed')
            learning_rates = _rebase_screening_checkpoint(ck, cfg)
            load_checkpoint_path = Path(cfg.work_dir) / 'continuation_input_iter_20000.pth'
            rank = int(os.environ.get('RANK', '0'))
            if rank == 0:
                if load_checkpoint_path.exists():
                    raise FileExistsError('Continuation input checkpoint already exists; preserving it')
                load_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_checkpoint = load_checkpoint_path.with_name(
                    f'.{load_checkpoint_path.name}.{os.getpid()}.tmp')
                torch.save(ck, temporary_checkpoint)
                os.replace(temporary_checkpoint, load_checkpoint_path)
                print(f'Rebased 20K checkpoint LR for 160K continuation: {learning_rates}', flush=True)
            else:
                deadline = time.monotonic() + 300
                while not load_checkpoint_path.is_file():
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Rank 0 did not publish the rebased continuation checkpoint')
                    time.sleep(.1)
        elif args.topology_proof:
            from .topology import verify_proof
            continuation = verify_proof(args.topology_proof, args.config, args.resume_from, meta)
            if not 0 < int(meta.get('iter', 0)) < cfg.train_cfg.max_iters:
                raise ValueError('Invalid resume iteration')
            if continuation is None:
                continuation = meta.get('topology_continuation')
        else:
            source_config = checkpoint_path.parent.parent.parent / 'configs' / (cfg.paper.experiment + '.py')
            if not source_config.is_file():
                raise ValueError('Resume checkpoint has no registered source config')
            source_cfg = Config.fromfile(str(source_config), import_custom_modules=False)
            validate_config(source_cfg.to_dict())
            source_config_sha = sha256(source_config)
            source_identity = identity(source_cfg, source_config_sha)
            if (meta.get('paper') != source_identity
                    or not _resume_config_matches(source_cfg, cfg)
                    or not 0 < int(meta.get('iter', 0)) < cfg.train_cfg.max_iters):
                raise ValueError('Resume requires the same experiment/protocol, seed, model recipe, budget and global batch; '
                                 'only DDP world size and per-GPU batch may change')
        del ck
        cfg.resume, cfg.load_from = True, str(load_checkpoint_path)
    runner = Runner.from_cfg(cfg)
    if continuation is not None:
        from .topology import attach_lineage
        attach_lineage(runner, continuation)
    # PyTorch 2.6 defaults torch.load(weights_only=True), which cannot restore
    # MMEngine MessageHub and NumPy RNG state from a full training checkpoint.
    # The resume checkpoint was identity-checked above and is either the
    # sealed source file or a transformed copy written from that file. Permit
    # full pickle loading only for this exact path; keep the default elsewhere.
    original_torch_load = torch.load
    trusted_resume_path = Path(cfg.load_from).resolve() if cfg.resume else None
    if trusted_resume_path is not None:
        def load_trusted_resume(file, *args, **kwargs):
            try:
                candidate = Path(file).resolve()
            except (TypeError, OSError):
                candidate = None
            if candidate == trusted_resume_path:
                kwargs['weights_only'] = False
            return original_torch_load(file, *args, **kwargs)
        torch.load = load_trusted_resume
    try:
        runner.train()
    finally:
        torch.load = original_torch_load
    if get_rank() == 0:
        work = Path(cfg.work_dir)
        record = load(work / 'terminal_eval.json')
        checkpoint = work / f'iter_{cfg.train_cfg.max_iters}.pth'
        if runner.iter != cfg.train_cfg.max_iters or not checkpoint.is_file():
            raise RuntimeError('Missing terminal checkpoint')
        completed_record = dict(record, protocol=cfg.paper.protocol, checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint), gradient_probe_sha256=sha256(work / 'gradient_probe.json'))
        if continuation is not None:
            completed_record['topology_continuation'] = continuation
        from .profile import measure_model_cost
        model = unwrap(runner.model)
        engineering = cfg.paper.get('engineering_diagnostic', False)
        warmup, repeats = (2, 5) if cfg.paper.smoke or engineering else (100, 1000)
        try:
            profile, _ = measure_model_cost(model, cfg, device=next(model.parameters()).device,
                checkpoint_sha256=completed_record['checkpoint_sha256'], warmup=warmup, repeats=repeats)
            ready = (profile.get('flops_counted_G') is not None
                     and profile.get('latency_median_ms') is not None
                     and profile.get('total_parameters', 0) > 0)
            profile_status = 'PASS' if ready and profile.get('flop_status') == 'analyzer_covered' else 'PARTIAL'
            profile_error = None
        except Exception as exc:
            profile = dict(experiment=cfg.paper.experiment, protocol=cfg.paper.protocol,
                config_sha256=cfg.paper_config_sha256, checkpoint_sha256=completed_record['checkpoint_sha256'],
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                flops_counted_G=None, flop_status='failed',
                profiling_error=f'{type(exc).__name__}: {exc}')
            profile_status, profile_error = 'FAILED', profile['profiling_error']
        profile.update(protocol=cfg.paper.protocol, config_sha256=cfg.paper_config_sha256,
                       iteration=runner.iter, profiling_status=profile_status,
                       measurement_kind='engineering_cost_check' if engineering else ('smoke_cost_check' if cfg.paper.smoke else 'publication_measurement'))
        if profile_error:
            profile['profiling_error'] = profile_error
        write_once(work / 'cost_profile.json', profile)
        completed_record['cost_profile_sha256'] = sha256(work / 'cost_profile.json')
        completed_record['cost_profile_status'] = profile_status
        write_once(work / 'completed.json', completed_record)
