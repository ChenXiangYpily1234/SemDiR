"""Prefer the user's vendored MMSeg; never import FDAM's legacy environment."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def activate():
    for path in (ROOT, ROOT / 'mmsegmentation'):
        if not path.is_dir():
            raise FileNotFoundError(f'Required source directory: {path}')
    vendor = str(ROOT / 'mmsegmentation')
    root = str(ROOT)
    for value in (vendor, root):
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
    if any('FDAM_mmseg' in p for p in sys.path):
        raise RuntimeError('Remove FDAM_mmseg from PYTHONPATH; it is legacy source, not the active MMSeg')


def check_loaded_paths():
    import timm
    import mmseg
    expected = ((timm, ROOT / 'timm'), (mmseg, ROOT / 'mmsegmentation' / 'mmseg'))
    for module, directory in expected:
        if not Path(module.__file__).resolve().is_relative_to(directory.resolve()):
            raise RuntimeError(f'Wrong imported {module.__name__}: {module.__file__}; expected {directory}')
    return {m.__name__: str(Path(m.__file__).resolve()) for m, _ in expected}
