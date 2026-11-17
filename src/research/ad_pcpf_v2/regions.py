"""Class-count-parametric SI mask, preserving the legacy divisible-grid definition.

Only the small majority grid goes to CPU for scipy connectivity. No GT enters inference.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.ndimage import label, maximum_filter
from torch.nn import functional as F
try:
    from numba import njit
except ImportError:
    njit = None


def _component_small_numpy(images, limit, ignore_index):
    """One union-find pass across all semantic classes; exact 8-connectivity."""
    batch,height,width = images.shape
    result = np.zeros(images.shape,dtype=np.bool_)
    for bi in range(batch):
        parents = np.arange(height*width,dtype=np.int32)
        sizes = np.zeros(height*width,dtype=np.int32)
        for y in range(height):
            for x in range(width):
                cls = images[bi,y,x]
                if cls == ignore_index:
                    continue
                current = y*width+x
                # Only already visited neighbors of the eight-connected grid.
                for direction in range(4):
                    py = y if direction == 0 else y-1
                    px = x-1 if direction <= 1 else (x if direction == 2 else x+1)
                    if py < 0 or px < 0 or px >= width or images[bi,py,px] != cls:
                        continue
                    a,b = current,py*width+px
                    while parents[a] != a:
                        parents[a] = parents[parents[a]]
                        a = parents[a]
                    while parents[b] != b:
                        parents[b] = parents[parents[b]]
                        b = parents[b]
                    if a < b:
                        parents[b] = a
                    else:
                        parents[a] = b
        for i in range(height*width):
            root = i
            while parents[root] != root:
                parents[root] = parents[parents[root]]
                root = parents[root]
            parents[i] = root
            sizes[root] += 1
        for y in range(height):
            for x in range(width):
                if images[bi,y,x] != ignore_index:
                    result[bi,y,x] = sizes[parents[y*width+x]] <= limit
    return result


_component_small = (njit(nogil=True,cache=False)(_component_small_numpy)
                    if njit is not None else None)


@torch.no_grad()
def small_interior_mask(labels, patch_size, num_classes, ignore_index=255):
    if labels.ndim != 3 or labels.dtype != torch.long or labels.shape[0] == 0:
        raise ValueError('Expected nonempty BHW int64 labels')
    b, h, w = labels.shape
    ph, pw = map(int, patch_size)
    if ph < 1 or pw < 1 or h % ph or w % pw or not 1 < num_classes < ignore_index:
        raise ValueError('SI requires integral patch tiles and a valid explicit label space')
    valid = labels != ignore_index
    if bool((((labels < 0) | (labels >= num_classes)) & valid).any()):
        raise ValueError('GT outside configured label space')
    gh, gw = h // ph, w // pw
    gy = torch.arange(h, device=labels.device) // ph
    gx = torch.arange(w, device=labels.device) // pw
    pid = gy[:, None] * gw + gx[None, :]
    global_pid = pid[None] + torch.arange(b, device=labels.device)[:, None, None] * gh * gw
    encoded = (global_pid * num_classes + labels)[valid]
    hist = torch.bincount(encoded, minlength=b * gh * gw * num_classes).reshape(b, gh * gw, num_classes)
    counts = hist.sum(-1)
    peak, majority = hist.max(-1)
    keep = (counts.double() / (ph * pw) >= .70) & (peak.double() / counts.clamp_min(1) >= .70)
    majority = torch.where(keep, majority, ignore_index).reshape(b, gh, gw)
    cpu = majority.cpu().numpy()
    selected = np.zeros(cpu.shape, dtype=bool)
    for bi in range(b):
        for cls in np.unique(cpu[bi]):
            if cls == ignore_index:
                continue
            components, _ = label(cpu[bi] == cls, structure=np.ones((3, 3), dtype=np.uint8))
            sizes = np.bincount(components.ravel())
            good = sizes <= 4
            good[0] = False
            selected[bi] |= good[components]
    chosen = torch.where(torch.as_tensor(selected, device=labels.device), majority, ignore_index)
    small = valid & (labels == chosen[:, gy[:, None], gx[None, :]])
    edge = torch.zeros_like(valid)
    diff = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[:, 1:] |= diff
    edge[:, :-1] |= diff
    diff = (labels[:, :, 1:] != labels[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, 1:] |= diff
    edge[:, :, :-1] |= diff
    return small & ~edge


def local_small_ce(logits, labels, patch_size, num_classes, ignore_index=255):
    mask = small_interior_mask(labels, patch_size, num_classes, ignore_index)
    aligned = F.interpolate(logits.float(), labels.shape[-2:], mode='bilinear', align_corners=False)
    pixel_loss = F.cross_entropy(aligned, labels, ignore_index=ignore_index, reduction='none')
    counts = mask.sum((1, 2))
    eligible = counts > 0
    per_image = (pixel_loss * mask).sum((1, 2)) / counts.clamp_min(1)
    n = eligible.sum().to(dtype=torch.float32)
    mean = (per_image * eligible).sum() / n.clamp_min(1)
    return mean, n


@torch.no_grad()
def ordinary_interior_mask(labels, num_classes=150, boundary_radius=3,
                           small_area_fraction=.005, ignore_index=255):
    """GT-only training mask on the actual augmented/padded label canvas.

    Small means an eight-connected semantic component, not an instance. The
    boundary band dilates valid four-neighbor class changes by radius pixels
    with a square kernel, matching the earlier regional diagnostics.
    """
    if labels.ndim != 3 or labels.dtype != torch.long or labels.shape[0] == 0:
        raise ValueError('Expected nonempty BHW int64 training labels')
    if type(boundary_radius) is not int or boundary_radius < 0:
        raise ValueError('Boundary radius must be a nonnegative integer')
    if not 0 < small_area_fraction < 1 or not 1 < num_classes < ignore_index:
        raise ValueError('Invalid semantic-region settings')
    cpu = labels.detach().cpu().numpy()
    valid = cpu != ignore_index
    if np.any(((cpu < 0) | (cpu >= num_classes)) & valid):
        raise ValueError('GT outside configured label space')
    edge = np.zeros_like(valid)
    different = (cpu[:, 1:] != cpu[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    edge[:, 1:] |= different
    edge[:, :-1] |= different
    different = (cpu[:, :, 1:] != cpu[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    edge[:, :, 1:] |= different
    edge[:, :, :-1] |= different
    if boundary_radius:
        edge = maximum_filter(edge,size=(1,2*boundary_radius+1,2*boundary_radius+1),mode='constant',cval=False)
    small = np.zeros(cpu.shape, dtype=bool)
    limit = small_area_fraction * labels.shape[-2] * labels.shape[-1]
    if _component_small is not None:
        small = _component_small(cpu,limit,ignore_index)
        return torch.as_tensor(valid & ~edge & ~small,device=labels.device)
    for bi, image in enumerate(cpu):
        for cls in np.unique(image):
            if cls == ignore_index:
                continue
            components, _ = label(image == cls, structure=np.ones((3, 3), dtype=np.uint8))
            sizes = np.bincount(components.ravel())
            selected = (sizes > 0) & (sizes <= limit)
            selected[0] = False
            small[bi] |= selected[components]
    return torch.as_tensor(valid & ~edge & ~small,device=labels.device)
