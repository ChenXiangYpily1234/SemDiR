"""Small standard-library provenance helpers shared by coordinator and workers."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def shared_digest(model):
    from .fdam import is_fdam_extra
    import torch
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith('backbone.relay.') or is_fdam_extra(key):
            continue
        h.update(key.encode())
        h.update(str((tuple(value.shape), value.dtype)).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def identity(cfg, config_sha):
    phase = 'engineering_diagnostic' if cfg.paper.get('engineering_diagnostic', False) else ('smoke' if cfg.paper.smoke else 'formal')
    return dict(protocol=cfg.paper.protocol, experiment=cfg.paper.experiment,
                seed=cfg.randomness.seed, runtime=dict(cfg.paper.runtime), phase=phase,
                config_sha256=config_sha)
