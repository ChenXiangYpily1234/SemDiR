"""ViTController paper-mechanism adapter for matched-budget comparison.

This is an independent implementation of the published layer-selection rule,
not the authors' official code. It uses all DeiT block outputs, obtains
channel-wise layer weights from the corresponding CLS tokens, fuses patch
tokens into one feature map, then delegates the four-scale conversion to the
caller-provided MMSeg Feature2Pyramid module.

Reference: https://arxiv.org/abs/2305.01279, Eqs. (1)-(6) and Fig. 2.
This independent reimplementation is NOT the authors' original package.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def vitcontroller_features(encoder: nn.Module, images: Tensor,
                           feature2pyramid: nn.Module) -> tuple[Tensor, ...]:
    """Fuse every ViT layer using CLS-conditioned per-channel softmax weights.

    The encoder must expose timm's ``forward_intermediates`` API and a
    ``patch_embed.dynamic_feat_size`` method. The pyramid module must implement
    MMSeg Feature2Pyramid with rescales ``(4, 2, 1, 0.5)``.

    Args:
        encoder: Pretrained DeiT/ViT encoder.
        images: Input batch in NCHW layout.
        feature2pyramid: Four-scale feature conversion module.

    Returns:
        Four NCHW feature maps for a UPerNet-style decoder.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW images, got shape {tuple(images.shape)}")
    if not hasattr(encoder, "blocks") or not hasattr(encoder, "patch_embed"):
        raise TypeError("ViTController requires a timm VisionTransformer encoder")
    if not callable(getattr(encoder, "forward_intermediates", None)):
        raise TypeError("Encoder must expose forward_intermediates")

    outputs = encoder.forward_intermediates(
        images,
        indices=None,
        return_prefix_tokens=True,
        norm=False,
        output_fmt="NLC",
        intermediates_only=True,
    )
    if len(outputs) != len(encoder.blocks):
        raise RuntimeError(
            f"Expected one intermediate per block ({len(encoder.blocks)}), got {len(outputs)}"
        )

    patch_layers: list[Tensor] = []
    cls_layers: list[Tensor] = []
    for index, item in enumerate(outputs):
        if not isinstance(item, Sequence) or len(item) != 2:
            raise TypeError(f"Layer {index} did not return (patch_tokens, prefix_tokens)")
        patches, prefixes = item
        if (not isinstance(patches, Tensor) or not isinstance(prefixes, Tensor) or
                patches.ndim != 3 or prefixes.ndim != 3 or prefixes.shape[1] < 1):
            raise ValueError(f"Invalid patch/prefix tensors at layer {index}")
        if patches.shape[0] != prefixes.shape[0] or patches.shape[2] != prefixes.shape[2]:
            raise ValueError(f"Patch/prefix batch or channel mismatch at layer {index}")
        patch_layers.append(patches)
        cls_layers.append(prefixes[:, 0])

    reference_shape = patch_layers[0].shape
    if any(layer.shape != reference_shape for layer in patch_layers):
        raise ValueError("All ViT layers must have the same patch-token shape")
    if any(layer.shape != cls_layers[0].shape for layer in cls_layers):
        raise ValueError("All ViT layers must have the same CLS-token shape")

    # M in the paper is [depth, channel]; softmax is over depth separately
    # for every channel, then each layer's vector scales its patch map.
    cls_matrix = torch.stack(cls_layers, dim=0)
    layer_weights = torch.softmax(cls_matrix, dim=0)
    fused = sum(
        patches * layer_weights[index].unsqueeze(1)
        for index, patches in enumerate(patch_layers)
    )

    grid = encoder.patch_embed.dynamic_feat_size(images.shape[-2:])
    grid = (int(grid[0]), int(grid[1]))
    batch, tokens, channels = fused.shape
    if tokens != grid[0] * grid[1]:
        raise ValueError(
            f"Patch grid {grid} expects {grid[0] * grid[1]} tokens, got {tokens}"
        )
    fused = fused.reshape(batch, grid[0], grid[1], channels).permute(0, 3, 1, 2).contiguous()
    pyramid_inputs = (fused, fused, fused, fused)
    pyramid = feature2pyramid(pyramid_inputs)
    if not isinstance(pyramid, (tuple, list)) or len(pyramid) != 4:
        raise TypeError("Feature2Pyramid must return exactly four feature maps")
    if any(not isinstance(feature, Tensor) or feature.ndim != 4 for feature in pyramid):
        raise ValueError("Every ViTController pyramid output must be an NCHW tensor")
    if any(feature.shape[0] != batch or feature.shape[1] != channels for feature in pyramid):
        raise ValueError("Pyramid outputs must preserve batch and embedding dimensions")
    return tuple(pyramid)
