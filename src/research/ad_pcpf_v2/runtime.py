"""Resolve execution parameters; no gradient accumulation is used."""
from .experiments import WORLD, PER_GPU, SEED


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value


def resolve_runtime(world_size=None, per_gpu=None, global_batch=None, seed=None):
    world_size = WORLD if world_size is None else positive_integer(world_size, 'world')
    if per_gpu is None:
        if global_batch is None:
            per_gpu = PER_GPU
        else:
            positive_integer(global_batch, 'global batch')
            if global_batch % world_size:
                raise ValueError('Without gradient accumulation, global batch must be divisible by world')
            per_gpu = global_batch // world_size
    positive_integer(per_gpu, 'per-GPU batch')
    expected = world_size * per_gpu
    if global_batch is not None and positive_integer(global_batch, 'global batch') != expected:
        raise ValueError(f'Inconsistent parameters: global batch must equal world × batch = {expected}; '
                         'gradient accumulation is disabled')
    seed = SEED if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return dict(world_size=world_size, per_gpu=per_gpu, global_batch=expected, seed=seed)


def runtime_from_config(cfg):
    runtime = dict(cfg['paper']['runtime'])
    if runtime != resolve_runtime(**runtime):
        raise ValueError('Invalid runtime metadata')
    positive_integer(cfg['train_dataloader']['batch_size'], 'dataloader batch')
    if type(cfg['randomness']['seed']) is not int:
        raise ValueError('Config seed must be an integer')
    if cfg['train_dataloader']['batch_size'] != runtime['per_gpu'] or cfg['randomness']['seed'] != runtime['seed']:
        raise ValueError('Runtime metadata disagrees with the dataloader or seed')
    return runtime
