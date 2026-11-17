"""Global eligible-image normalization under standard DDP gradient averaging."""
import torch
import math
from torch import Tensor
import torch.distributed as dist
from torch.nn import functional as F


def scale_local_mean(local_mean: Tensor, local_count: Tensor,
                     global_count: Tensor, world_size: int) -> Tensor:
    """Use R*S_local/N_global, so DDP's mean gives the global objective.

    Counts are detached. Zero-count ranks retain the local graph with a zero
    multiplier. This assumes default DDP averaging, not a custom comm hook.
    """
    return (local_mean * local_count.detach().to(local_mean)
            * world_size / global_count.detach().to(local_mean).clamp_min(1))


def global_eligible_mean(local_mean: Tensor, local_count: Tensor):
    if local_mean.ndim != 0 or local_count.numel() != 1:
        raise ValueError('Scalar mean and eligible image count required')
    count = local_count.detach().to(device=local_mean.device, dtype=torch.float32).reshape(())
    if not bool(torch.isfinite(count)) or bool(count < 0) or bool(count != count.round()):
        raise ValueError('Eligible count must be a finite nonnegative integer')
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    totals = torch.stack((local_mean.detach().float()*count, count))
    if world > 1:
        # Every rank calls, including ranks with no eligible image.
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    loss = scale_local_mean(local_mean, count, totals[1], world)
    return loss, totals[0]/totals[1].clamp_min(1), totals[1]


def region_semantic_margin(logits, labels, ordinary_mask, margin=.5, ignore_index=255):
    """Global class-balanced interior hinge margin, with DDP-mean gradients.

    Counts and class presence refer to the entire distributed batch. Empty
    ranks still participate in every collective and retain a zero local graph.
    No teacher, second forward, GT-dependent inference or learned parameters.
    """
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('RSP margin must be finite and nonnegative')
    if (logits.ndim != 4 or logits.shape[1] < 2 or labels.ndim != 3
            or logits.shape[0] != labels.shape[0] or labels.dtype != torch.long
            or ordinary_mask.shape != labels.shape or ordinary_mask.dtype != torch.bool):
        raise ValueError('Expected BCHW logits, BHW int64 labels and boolean interior mask')
    classes = logits.shape[1]
    with torch.autocast(device_type=logits.device.type, enabled=False):
        aligned = F.interpolate(logits.float(), labels.shape[-2:], mode='bilinear', align_corners=False)
        valid = (labels >= 0) & (labels < classes) & (labels != ignore_index)
        selected = ordinary_mask & valid
        safe_labels = labels.clamp(0, classes-1)
        true = aligned.gather(1, safe_labels[:, None]).squeeze(1)
        values, indices = aligned.topk(2, dim=1)
        wrong = torch.where(indices[:, 0] == safe_labels, values[:, 1], values[:, 0])
        difference = true - wrong
        pixel_loss = F.relu(margin-difference)
        # Fixed-size reductions avoid nonzero()/bincount() output-size CPU
        # synchronization on every rank. Masked pixels contribute exact zero.
        ids = safe_labels.reshape(-1)
        local_sums = aligned.new_zeros(classes).scatter_add(
            0,ids,pixel_loss.masked_fill(~selected,0).reshape(-1))
        counts = torch.zeros(classes,device=labels.device,dtype=torch.int64).scatter_add(
            0,ids,selected.reshape(-1).to(torch.int64))
        summary = torch.stack((difference.detach().masked_fill(~selected, 0).sum().double(),
                               ((difference.detach() < margin) & selected).sum().double(),
                               valid.sum().double(),
                               ((aligned.detach().argmax(1) != safe_labels) & selected).sum().double()))
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        report = torch.cat((local_sums.detach().double(),summary))
        if world > 1:
            dist.all_reduce(counts)
            dist.all_reduce(report)
        summary = report[classes:]
        report = report[:classes]
        present = counts > 0
        number = present.sum().clamp_min(1)
        denominators = counts.clamp_min(1).to(local_sums)
        loss = world * (local_sums / denominators).sum() / number
        total = counts.sum()
        logged = (report/counts.clamp_min(1)).sum()/number
    return loss, dict(rsp_margin_raw=logged.float(), rsp_inner_pixels=total.float(),
        rsp_present_classes=present.sum().float(), rsp_valid_pixels=summary[2].float(),
        rsp_inner_fraction=total.float()/summary[2].clamp_min(1).float(),
        rsp_mean_margin=(summary[0]/total.clamp_min(1)).float(),
        rsp_violation_fraction=(summary[1]/total.clamp_min(1)).float(),
        rsp_error_fraction=(summary[3]/total.clamp_min(1)).float())
