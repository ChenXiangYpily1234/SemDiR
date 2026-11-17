"""MMSegmentation backbone adapter for the isolated ViTController baseline.

This reproduces the published CLS-conditioned layer fusion rule and uses
MMSegmentation's Feature2Pyramid, but is not the authors' official code. Runs only in a new source-signed study; do not edit or attach active studies.
"""
from __future__ import annotations

import importlib

import torch
from mmseg.models.necks import Feature2Pyramid
from mmseg.registry import MODELS

from .mmseg_adapter import TimmDeiTBackbone


_FUSION = importlib.import_module(
    "experiments.dcpa.vitcontroller_matched"
)


@MODELS.register_module()
class TimmDeiTViTControllerBackbone(TimmDeiTBackbone):
    """All-layer CLS-conditioned fusion with the paper's four-scale pyramid."""

    def __init__(self, pyramid_norm_cfg=None, **kwargs):
        kwargs["dcpa"] = False
        super().__init__(**kwargs)
        if pyramid_norm_cfg is None:
            pyramid_norm_cfg = dict(type="SyncBN", requires_grad=True)
        self.feature2pyramid = Feature2Pyramid(
            embed_dim=self.encoder.embed_dim,
            rescales=[4, 2, 1, 0.5],
            norm_cfg=pyramid_norm_cfg,
        )

    def forward(self, images: torch.Tensor):
        return _FUSION.vitcontroller_features(
            self.encoder, images, self.feature2pyramid)
