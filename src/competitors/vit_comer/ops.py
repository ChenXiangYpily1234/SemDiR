"""Same MSDeformAttn operator, using installed MMCV CUDA / its PyTorch reference.

FP32 is required by the installed extension; casts remain differentiable.
No legacy standalone extension or old mmcv.runner package is installed.
"""
import torch
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)


class MSDeformAttnFunction:
    @staticmethod
    def apply(value, shapes, starts, locations, weights, step):
        dtype = value.dtype
        with torch.autocast(device_type=value.device.type, enabled=False):
            if value.is_cuda:
                out = MultiScaleDeformableAttnFunction.apply(
                    value.float(), shapes, starts, locations.float(), weights.float(), step)
            else:
                out = multi_scale_deformable_attn_pytorch(
                    value.float(), shapes, locations.float(), weights.float())
        return out.to(dtype)
