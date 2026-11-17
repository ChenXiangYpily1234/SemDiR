"""Standalone V2 paper models; no imports from the removed ad_pcpf_v1 package."""
from __future__ import annotations
import hashlib
import math
import re
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import torch
from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.backbones import VisionTransformer
from mmseg.utils import add_prefix
from timm.layers import DropPath
from experiments.dcpa.vitcontroller_mmseg_adapter import TimmDeiTViTControllerBackbone
from .relay import TaskGatedAllDepthRelay, SemanticPyramidRelay
from .regions import local_small_ce, ordinary_interior_mask
from .loss import global_eligible_mean, region_semantic_margin
from .fdam import FDAMBlock, is_fdam_extra


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _deit_checkpoint_to_mmseg(state):
    """Map the released timm DeiT-S state dict to MMSeg's ViT naming."""
    converted = {}
    for key, value in state.items():
        # The segmentation backbone has no classifier or final ImageNet norm.
        if key.startswith(('head.', 'head_dist.', 'norm.')):
            continue
        if key.startswith('patch_embed.proj.'):
            key = key.replace('patch_embed.proj.', 'patch_embed.projection.', 1)
        elif key.startswith('blocks.'):
            key = key.replace('blocks.', 'layers.', 1)
            key = key.replace('.norm1.', '.ln1.', 1)
            key = key.replace('.norm2.', '.ln2.', 1)
            key = key.replace('.attn.qkv.', '.attn.attn.in_proj_', 1)
            key = key.replace('.attn.proj.', '.attn.attn.out_proj.', 1)
            key = key.replace('.mlp.fc1.', '.ffn.layers.0.0.', 1)
            key = key.replace('.mlp.fc2.', '.ffn.layers.1.', 1)
        if key in converted:
            raise RuntimeError(f'Duplicate converted DeiT parameter: {key}')
        converted[key] = value
    return converted


@MODELS.register_module()
class FDAMBaselineDeiTBackbone(VisionTransformer):
    """MMSeg VisionTransformer used by FDAM's published DeiT-S baseline.

    The official config uses this backbone and MultiLevelNeck. The released
    checkpoint is kept in its original form; conversion and hash verification
    happen here so the run has no implicit download or unverified weights.
    """

    def __init__(self, pretrained_checkpoint, pretrained_sha256, **kwargs):
        if not pretrained_checkpoint or not re.fullmatch(r'[0-9a-f]{64}', str(pretrained_sha256)):
            raise ValueError('An explicit DeiT checkpoint and SHA256 are required')
        depth = kwargs.get('num_layers', 12)
        kwargs.pop('pretrained', None)
        kwargs.pop('init_cfg', None)
        super().__init__(pretrained=None, init_cfg=None, **kwargs)
        self.pretrained_checkpoint = str(pretrained_checkpoint)
        self.expected_pretrained_sha256 = str(pretrained_sha256)
        self.method = 'vanilla'
        self.depth = depth

    def init_weights(self):
        if self._is_init:
            return
        # Initialize non-pretrained tensors first, then strictly load every
        # backbone parameter from the verified released checkpoint.
        super().init_weights()
        path = Path(self.pretrained_checkpoint)
        if file_sha256(path) != self.expected_pretrained_sha256:
            raise RuntimeError('Pretrained checkpoint hash mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        state = _deit_checkpoint_to_mmseg(state)
        if 'pos_embed' in state and state['pos_embed'].shape != self.pos_embed.shape:
            pos_grid = math.isqrt(state['pos_embed'].shape[1] - 1)
            if pos_grid * pos_grid != state['pos_embed'].shape[1] - 1:
                raise RuntimeError('Expected a square pretrained DeiT positional grid')
            state['pos_embed'] = self.resize_pos_embed(
                state['pos_embed'],
                (self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size),
                (pos_grid, pos_grid), self.interpolate_mode)
        incompatible = self.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                'DeiT checkpoint mapping mismatch: '
                f'missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}')
        self._is_init = True


@MODELS.register_module()
class ADPCPFPaperBackbone(TimmDeiTViTControllerBackbone):
    def __init__(self, method='adpcpf_r9', pretrained_checkpoint=None, pretrained_sha256=None,
                 drop_path_rate=.1, relay_options=None, **kwargs):
        if method not in ('vanilla', 'readout_only', 'adpcpf', 'adpcpf_r9', 'fdam'):
            raise ValueError('Unregistered paper method')
        if pretrained_checkpoint is None or pretrained_sha256 is None:
            raise ValueError('Explicit checkpoint and SHA256 required; no silent downloads')
        kwargs.pop('pretrained', None)
        super().__init__(pretrained=False, pretrained_checkpoint=None, **kwargs)
        self.pretrained_checkpoint = str(pretrained_checkpoint)
        self.expected_pretrained_sha256 = pretrained_sha256
        self.method, self.depth = method, len(self.encoder.blocks)
        self.indices = tuple(math.ceil(self.depth * q / 4) - 1 for q in (1, 2, 3, 4))
        for block, rate in zip(self.encoder.blocks, torch.linspace(0, drop_path_rate, self.depth).tolist()):
            block.drop_path1, block.drop_path2 = DropPath(rate), DropPath(rate)
        with torch.random.fork_rng(devices=[]):
            if method == 'adpcpf':
                self.relay = TaskGatedAllDepthRelay(self.encoder.embed_dim, **(relay_options or {}))
            elif method == 'adpcpf_r9':
                self.relay = SemanticPyramidRelay(self.encoder.embed_dim, **(relay_options or {}))
            elif method == 'fdam':
                self.encoder.blocks = torch.nn.Sequential(*(FDAMBlock(block) for block in self.encoder.blocks))
        self._paper_initialized = False

    def init_weights(self):
        if self._paper_initialized:
            return
        # timm initializes the encoder at construction. Restore the entire shared
        # pretrained encoder without reinitializing method-specific submodules.
        path = Path(self.pretrained_checkpoint)
        if file_sha256(path) != self.expected_pretrained_sha256:
            raise RuntimeError('Pretrained checkpoint hash mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        state = {k: v for k, v in state.items() if not k.startswith(('head.', 'head_dist.'))}
        missing, extra = self.encoder.load_state_dict(state, strict=False)
        expected_missing = {k for k in self.encoder.state_dict() if is_fdam_extra(k)} if self.method == 'fdam' else set()
        if set(missing) != expected_missing or extra:
            raise RuntimeError(f'Pretrained mapping mismatch: missing={missing}, unexpected={extra}')
        # Feature2Pyramid is a plain torch.nn.Module, not an MMEngine
        # BaseModule. Its ConvTranspose2d layers receive PyTorch's standard
        # constructor initialization, so it has no init_weights() hook.
        if hasattr(self, 'relay'):
            self.relay.reset_relay()
        self._paper_initialized = True
        self._is_init = True

    def _features(self, image):
        h, w = map(int, self.encoder.patch_embed.dynamic_feat_size(image.shape[-2:]))
        if self.method == 'fdam':
            for block in self.encoder.blocks:
                block.grid, block.prefix_tokens = (h, w), self.encoder.num_prefix_tokens
        outputs = self.encoder.forward_intermediates(image, indices=None, return_prefix_tokens=True,
                    norm=False, output_fmt='NLC', intermediates_only=True)
        if len(outputs) != self.depth:
            raise RuntimeError('All ViT intermediate blocks required')
        patches, cls = [], []
        for patch, prefix in outputs:
            if patch.ndim != 3 or prefix.ndim != 3 or prefix.shape[1] < 1 or patch.shape[1] != h * w:
                raise RuntimeError('Expected plain-ViT spatial tokens and CLS token')
            patches.append(patch)
            cls.append(prefix[:, 0])
        b, _, c = patches[0].shape
        maps = [x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous() for x in patches]
        if self.method not in ('adpcpf', 'readout_only'):
            return maps, None
        weights = torch.stack(cls).softmax(0)
        fused = sum(x * weights[i, :, None, :] for i, x in enumerate(patches))
        return maps, fused.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    def forward(self, image):
        maps, fused = self._features(image)
        if self.method == 'readout_only':
            return self.feature2pyramid((fused,) * 4)
        if self.method == 'adpcpf_r9':
            return self.relay(image, maps, self.indices, self.feature2pyramid)
        if self.method == 'adpcpf':
            return self.relay(image, maps, fused, self.feature2pyramid)
        return self.feature2pyramid(tuple(maps[i] for i in self.indices))

    @torch.no_grad()
    def check_zero_initialization(self, images):
        if self.method not in ('adpcpf', 'adpcpf_r9'):
            return
        training = self.training
        self.eval()
        try:
            maps, fused = self._features(images)
            if self.method == 'adpcpf_r9':
                reference = self.feature2pyramid(tuple(maps[i] for i in self.indices))
                actual = self.relay(images, maps, self.indices, self.feature2pyramid)
            else:
                reference = self.feature2pyramid((fused,) * 4)
                actual = self.relay(images, maps, fused, self.feature2pyramid)
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=0., atol=1e-6)
        finally:
            self.train(training)


@MODELS.register_module()
class ADPCPFPaperSegmentor(EncoderDecoder):
    def __init__(self, small_ce_weight=0., small_ce_normalization='global_eligible_images', **kwargs):
        super().__init__(**kwargs)
        if small_ce_normalization != 'global_eligible_images' or small_ce_weight not in (0., .05):
            raise ValueError('Only registered global-SI protocols are supported')
        if self.backbone.method == 'adpcpf_r9' and small_ce_weight != 0.:
            raise ValueError('R9 requires no SI (small_ce_weight=0)')
        self.small_ce_weight = float(small_ce_weight)
        self.small_ce_normalization = small_ce_normalization

    def _loss_and_logits(self, inputs, data_samples):
        features = self.extract_feat(inputs)
        logits = self.decode_head(features)
        losses = add_prefix(self.decode_head.loss_by_feat(logits, data_samples), 'decode')
        if self.with_auxiliary_head:
            losses.update(self._auxiliary_head_forward_train(features, data_samples))
        zero = logits.float().sum() * 0.
        raw, count = zero, zero.detach()
        if self.small_ce_weight:
            labels = torch.stack([s.gt_sem_seg.data[0] for s in data_samples]).to(inputs.device, torch.long)
            if labels.shape[-2:] != inputs.shape[-2:]:
                raise ValueError('SI GT and padded input must align')
            raw, count = local_small_ce(logits, labels, self.backbone.encoder.patch_embed.patch_size,
                                         self.decode_head.num_classes, self.decode_head.ignore_index)
            scaled, report, global_count = global_eligible_mean(raw, count)
            losses['loss_small_ce'] = self.small_ce_weight * scaled
            losses['small_ce_raw'], losses['small_ce_image_count'] = report.detach(), global_count.detach()
        else:
            losses.update(loss_small_ce=zero, small_ce_raw=zero.detach(), small_ce_image_count=count)
        return losses, logits

    def loss(self, inputs, data_samples):
        return self._loss_and_logits(inputs, data_samples)[0]


@MODELS.register_module()
class ADPCPFRegionPreservingSegmentor(ADPCPFPaperSegmentor):
    """Original R9 inference with a single-forward, train-only RSP objective."""
    def __init__(self, region_preservation, **kwargs):
        super().__init__(**kwargs)
        relay = getattr(self.backbone, 'relay', None)
        if (self.backbone.method != 'adpcpf_r9' or relay is None or relay.novelty
                or relay.branch_budget is not None or relay.rgb_budget is not None
                or relay.residual_scale != 1.):
            raise ValueError('R9-RSP requires the unchanged R9 inference structure')
        self.region_preservation = deepcopy(region_preservation)
        options = self.region_preservation
        if (not math.isfinite(options['margin']) or options['margin'] < 0
                or not math.isfinite(options['max_weight']) or options['max_weight'] < 0
                or type(options['warmup_iters']) is not int or options['warmup_iters'] <= 0
                or type(options['boundary_radius']) is not int or options['boundary_radius'] < 0
                or not 0 < options['small_area_fraction'] < 1):
            raise ValueError('Invalid RSP margin, schedule or region definition')
        self._rsp_iteration = 0
        object.__setattr__(self, '_rsp_mask_pool', None)

    def set_region_preservation_step(self, iteration):
        if type(iteration) is not int or iteration < 0:
            raise ValueError('RSP iteration must be a nonnegative integer')
        self._rsp_iteration = iteration

    def loss(self, inputs, data_samples):
        options = self.region_preservation
        if not options.get('enabled', True) or options['max_weight'] == 0:
            return self._loss_and_logits(inputs,data_samples)[0]
        labels = torch.stack([s.gt_sem_seg.data[0] for s in data_samples]).to(inputs.device, torch.long)
        if labels.shape[-2:] != inputs.shape[-2:]:
            raise ValueError('RSP labels must match the actual augmented/padded input')
        args = (self.decode_head.num_classes,options['boundary_radius'],
                options['small_area_fraction'],self.decode_head.ignore_index)
        if inputs.is_cuda:
            # Copy labels before the backbone is queued, then overlap the
            # GT-only CPU work with the student's single GPU forward.
            if self._rsp_mask_pool is None:
                object.__setattr__(self,'_rsp_mask_pool',ThreadPoolExecutor(max_workers=1))
            cpu_labels = labels.detach().cpu()
            future = self._rsp_mask_pool.submit(ordinary_interior_mask,cpu_labels,*args)
            losses,logits = self._loss_and_logits(inputs,data_samples)
            ordinary = future.result().to(inputs.device)
        else:
            ordinary = ordinary_interior_mask(labels,*args)
            losses,logits = self._loss_and_logits(inputs,data_samples)
        loss, diagnostics = region_semantic_margin(logits, labels, ordinary,
            options['margin'], self.decode_head.ignore_index)
        weight = options['max_weight'] * min(1., self._rsp_iteration/options['warmup_iters'])
        losses.update(loss_rsp=weight*loss, rsp_weight=loss.new_tensor(weight), **diagnostics)
        return losses
