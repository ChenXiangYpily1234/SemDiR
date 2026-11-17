"""Dataset-level segmentation statistics and explicitly defined local diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, label


def image_statistics(pred, gt, classes, diagnostics=False):
    pred, gt = np.asarray(pred, dtype=np.int64), np.asarray(gt, dtype=np.int64)
    if pred.shape != gt.shape or gt.ndim != 2:
        raise ValueError('Prediction and original-resolution GT must align')
    valid = (gt >= 0) & (gt < classes)
    if np.any((pred[valid] < 0) | (pred[valid] >= classes)):
        raise ValueError('Prediction outside native label space')
    matrix = np.bincount(classes * gt[valid] + pred[valid], minlength=classes**2).reshape(classes, classes)
    row = dict(confusion=matrix, boundary_intersection=np.zeros(classes, dtype=np.int64),
               boundary_union=np.zeros(classes, dtype=np.int64), small_sum=0., small_count=0)
    if not diagnostics:
        return row
    h, w = gt.shape
    radius = max(1, int(round(.02 * np.hypot(h, w))))
    kernel = np.ones((3, 3), dtype=bool)
    safe = ~binary_dilation(~valid, structure=kernel, iterations=radius, border_value=0)
    for cls in np.union1d(np.unique(gt[valid]), np.unique(pred[valid])):
        target, prediction = (gt == cls) & valid, (pred == cls) & valid
        tb = target & ~binary_erosion(target, structure=kernel, iterations=radius, border_value=0)
        pb = prediction & ~binary_erosion(prediction, structure=kernel, iterations=radius, border_value=0)
        row['boundary_intersection'][cls] = np.count_nonzero(tb & pb & safe)
        row['boundary_union'][cls] = np.count_nonzero((tb | pb) & safe)
        components, n = label(target, structure=np.ones((3, 3), dtype=np.uint8))
        if n:
            sizes = np.bincount(components.ravel())
            correct = np.bincount(components[pred == cls].ravel(), minlength=len(sizes))
            selected = np.flatnonzero((sizes > 0) & (sizes <= .005 * h * w))
            selected = selected[selected != 0]
            row['small_sum'] += float((correct[selected] / sizes[selected]).sum())
            row['small_count'] += int(len(selected))
    return row


def aggregate(rows, classes):
    if not rows:
        raise ValueError('No evaluation samples')
    cm = sum((r['confusion'] for r in rows), np.zeros((classes, classes), dtype=np.int64))
    inter, gt_count, pred_count = np.diag(cm), cm.sum(1), cm.sum(0)
    union = gt_count + pred_count - inter
    present = union > 0
    if not present.any():
        raise ValueError('No evaluable semantic classes')
    iou = np.divide(inter, union, out=np.full(classes, np.nan), where=present)
    bi = sum((r['boundary_intersection'] for r in rows), np.zeros(classes, dtype=np.int64))
    bu = sum((r['boundary_union'] for r in rows), np.zeros(classes, dtype=np.int64))
    bn = int((bu > 0).sum())
    sn = sum(r['small_count'] for r in rows)
    return dict(mIoU=float(np.nanmean(iou) * 100),
                per_class_IoU=[None if not np.isfinite(x) else float(x * 100) for x in iou],
                evaluable_classes=int(present.sum()),
                mBIoU=float(np.mean(bi[bu > 0] / bu[bu > 0]) * 100) if bn else None,
                boundary_evaluable_classes=bn,
                small_region_recall=float(sum(r['small_sum'] for r in rows) / sn * 100) if sn else None,
                small_regions=sn, images=len(rows)), cm, bi, bu
