"""MMSegmentation-compatible timm DeiT-S feature backbone.

Used only by experiments/dcpa/run.py; not registered unless imported.
"""
from __future__ import annotations

import torch
from mmengine.model import BaseModule
from mmseg.registry import MODELS

import timm




@MODELS.register_module()
class TimmDeiTBackbone(BaseModule):
    def __init__(
            self,
            model_name="deit_small_patch16_224.fb_in1k",
            image_size=512,
            pretrained=True,
            pretrained_checkpoint=None,
            dcpa=False,
            layers=(9, 10, 11),
            selector="topk",
            tau=0.1,
            strength=0.25,
            steps=1,
            projection="halfspace",
            keep_ratio=0.25,
            out_indices=(2, 5, 8, 11),
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.encoder = timm.create_model(
            model_name,
            pretrained=pretrained and pretrained_checkpoint is None,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        self.pretrained_checkpoint = pretrained_checkpoint
        # MMSeg calls init_weights() after constructing the full model. The timm
        # VisionTransformer child is reinitialized there, so restore pretrained
        # values only after BaseModule has completed its recursive initialization.
        self._initial_pretrained_state = None
        if pretrained and pretrained_checkpoint is None:
            self._initial_pretrained_state = {
                key: value.detach().cpu().clone()
                for key, value in self.encoder.state_dict().items()
            }
        # Freeze before DDP wraps the model; this norm is excluded from the
        # intermediate feature path and therefore receives no loss gradient.
        for parameter in self.encoder.norm.parameters():
            parameter.requires_grad_(False)
        self.out_indices = tuple(out_indices)
        if dcpa:
            raise ValueError(
                "SRUP has been archived to old/methods/srup; use active VCRD/ALMR modes."
            )

    def init_weights(self):
        if self._is_init:
            return
        super().init_weights()

        if self.pretrained_checkpoint is not None:
            checkpoint = torch.load(
                self.pretrained_checkpoint, map_location="cpu", weights_only=True
            )
            state_dict = checkpoint.get("model", checkpoint)
            state_dict = self._adapt_pretrained_state_dict(state_dict)
            incompatible = self.encoder.load_state_dict(state_dict, strict=False)
            allowed_missing = {
                key for key in incompatible.missing_keys if key.startswith("head.")
            }
            allowed_unexpected = {
                key for key in incompatible.unexpected_keys if key.startswith("head.")
            }
            if (set(incompatible.missing_keys) != allowed_missing or
                    set(incompatible.unexpected_keys) != allowed_unexpected):
                raise RuntimeError(
                    "Unexpected DeiT checkpoint mismatch: "
                    f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
                )
        elif self._initial_pretrained_state is not None:
            state_dict = self._adapt_pretrained_state_dict(
                self._initial_pretrained_state)
            self.encoder.load_state_dict(state_dict, strict=True)
            self._initial_pretrained_state = None

        # dynamic_img_size resamples absolute position embeddings from the actual
        # patch grid at each forward. Keep timm's Parameter objects intact here:
        # MMEngine tracks them during init_weights and rejects parameter replacement.

    def _adapt_pretrained_state_dict(self, state_dict):
        """Map a standard DeiT checkpoint into this encoder's module layout."""
        return state_dict

    def forward(self, x: torch.Tensor):
        return tuple(self.encoder.forward_intermediates(
            x, indices=list(self.out_indices), output_fmt="NCHW",
            intermediates_only=True, norm=False,
        ))
