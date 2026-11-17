"""Run an exported recipe without needing historical studies or result files."""
from pathlib import Path
import argparse
import importlib
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'src/mmsegmentation')]
from experiment_index import EXPERIMENTS


def resolve(value, data_root, pretrain_root):
    if isinstance(value, str):
        return value.replace('${DATA_ROOT}', str(data_root)).replace('${PRETRAIN_ROOT}', str(pretrain_root))
    if isinstance(value, dict):
        return {key: resolve(item, data_root, pretrain_root) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(resolve(item, data_root, pretrain_root) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['list', 'inspect', 'check', 'train', 'test'])
    parser.add_argument('experiment', nargs='?')
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get('SEMDIR_DATA_ROOT', 'data')))
    parser.add_argument('--pretrain-root', type=Path, default=Path(os.environ.get('SEMDIR_PRETRAIN_ROOT', 'pretrained')))
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Required for training or evaluation.')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    if args.action == 'list':
        for spec in EXPERIMENTS:
            print(f"{spec['alias']:30} {spec['dataset']:15} {spec['backbone']:8} "
                  f"stop={spec['iterations']} schedule={spec['schedule_end']} {spec['config']}")
        return
    spec = next((item for item in EXPERIMENTS if item['alias'] == args.experiment), None)
    if spec is None:
        parser.error('Select an experiment from the list action.')
    from mmengine.config import Config
    cfg = Config(resolve(Config.fromfile(str(ROOT / spec['config']), import_custom_modules=False).to_dict(),
                         args.data_root.resolve(), args.pretrain_root.resolve()))
    if args.action == 'inspect':
        print(cfg.pretty_text)
        return
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    for module in cfg.custom_imports.imports:
        importlib.import_module(module)
    from mmseg.registry import MODELS
    from mmengine.registry import HOOKS
    from mmseg.registry import DATASETS
    assert MODELS.get(cfg.model.type) is not None
    assert MODELS.get(cfg.model.backbone.type) is not None
    assert HOOKS.get('SemDiRIterationHook') is not None
    for key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
        assert DATASETS.get(cfg[key].dataset.type) is not None
    assert cfg.train_cfg.max_iters == spec['iterations']
    assert cfg.param_scheduler[-1].end == spec['schedule_end']
    if args.action == 'check':
        print(f"PASS: {spec['alias']}: config, model/dataset registration and RSP iteration hook")
        return
    if not args.execute:
        parser.error('train/test require --execute; check and inspect never start training or evaluation.')
    if args.work_dir is None:
        parser.error('Use --work-dir to select a new output directory outside the code package.')
    work_dir = args.work_dir.resolve()
    if work_dir == ROOT or ROOT in work_dir.parents:
        parser.error('Keep run outputs outside this source-only package.')
    if args.action == 'train' and work_dir.exists() and any(work_dir.iterdir()) and not args.resume:
        parser.error('Training output directory is nonempty; select a new directory or explicitly --resume.')
    if args.action == 'test' and args.checkpoint is None:
        parser.error('test requires --checkpoint.')
    cfg.work_dir = str(work_dir)
    cfg.launcher = args.launcher
    cfg.resume = args.resume
    if args.checkpoint is not None:
        cfg.load_from = str(args.checkpoint.resolve())
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('--batch-size must be positive.')
        cfg.train_dataloader.batch_size = args.batch_size
    if args.workers is not None:
        if args.workers < 0:
            parser.error('--workers must be nonnegative.')
        for key in ['train_dataloader', 'val_dataloader', 'test_dataloader']:
            cfg[key].num_workers = args.workers
            cfg[key].persistent_workers = args.workers > 0
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))
    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    if args.action == 'train':
        runner.train()
    else:
        runner.test()


if __name__ == '__main__':
    main()
