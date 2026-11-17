"""Self-contained registered paper configurations; no dependency on V1.

R7 defaults can be overridden through registered runtime parameters. M01-M02 are
preserved R3 historical runs; never prepare them in the R7 output directory.
"""
from __future__ import annotations
from copy import deepcopy
import re
from .experiments import PROTOCOL, SEED, WORLD, PER_GPU, PRETRAIN_SHA256, get_experiment, relay_options

from .runtime import resolve_runtime, runtime_from_config

ARMS = ('M19',)  # R10 Full; compare against preserved M03, without retraining it.
STEPS = 160000


def _build(name, data_root, pretrained, work_dir, pretrained_sha256, *,
           world_size=WORLD, per_gpu=PER_GPU, global_batch=None, seed=SEED):
    runtime = resolve_runtime(world_size, per_gpu, global_batch, seed)
    e = get_experiment(name)
    if not re.fullmatch('[0-9a-f]{64}', str(pretrained_sha256)):
        raise ValueError('A verified 64-character pretrained SHA256 is required')
    if e.backbone == 'small' and pretrained_sha256 != PRETRAIN_SHA256:
        raise ValueError('DeiT-S initialization is fixed')
    crop, classes = e.crop, e.classes
    channels = 384 if e.backbone == 'small' else 768
    resize = (2048, 512) if e.dataset == 'ade20k' else (520, 520)
    stride = 341 if crop == 512 else 320
    norm = dict(type='SyncBN', requires_grad=True)
    train_pipe = [dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', reduce_zero_label=True),
        dict(type='RandomResize', scale=resize, ratio_range=(.5, 2.), keep_ratio=True),
        dict(type='RandomCrop', crop_size=(crop, crop), cat_max_ratio=.75),
        dict(type='RandomFlip', prob=.5), dict(type='PhotoMetricDistortion'), dict(type='PackSegInputs')]
    val_pipe = [dict(type='LoadImageFromFile'), dict(type='Resize', scale=resize, keep_ratio=True),
        dict(type='LoadAnnotations', reduce_zero_label=True), dict(type='PackSegInputs')]
    def dataset(train):
        if e.dataset == 'ade20k':
            split = 'training' if train else 'validation'
            return dict(type='ADE20KDataset', data_root=str(data_root), reduce_zero_label=True,
                        data_prefix=dict(img_path='images/' + split, seg_map_path='annotations/' + split),
                        pipeline=train_pipe if train else val_pipe)
        return dict(type='PascalContextDataset59', data_root=str(data_root), reduce_zero_label=True,
                    data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
                    ann_file='ImageSets/SegmentationContext/' + ('train.txt' if train else 'val.txt'),
                    pipeline=train_pipe if train else val_pipe)
    train_loader = dict(batch_size=runtime['per_gpu'], num_workers=4, persistent_workers=True,
                        sampler=dict(type='InfiniteSampler', shuffle=True), dataset=dataset(True))
    val_loader = dict(batch_size=1, num_workers=4, persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False), dataset=dataset(False))
    cfg = dict(default_scope='mmseg',
        custom_imports=dict(imports=['research.ad_pcpf_v2.adapter', 'research.ad_pcpf_v2.engine'],
                            allow_failed_imports=False),
        paper=dict(protocol=e.protocol, experiment=e.id, dataset=e.dataset, backbone=e.backbone,
                   reference=e.reference, pretrained_sha256=pretrained_sha256, smoke=False, runtime=runtime),
        model=dict(type='ADPCPFPaperSegmentor', small_ce_weight=e.si,
            small_ce_normalization='global_eligible_images',
            data_preprocessor=dict(type='SegDataPreProcessor', mean=[123.675,116.28,103.53],
                std=[58.395,57.12,57.375], bgr_to_rgb=True, pad_val=0, seg_pad_val=255,
                size=(crop,crop), test_cfg=dict(size_divisor=16)),
            backbone=dict(type='ADPCPFPaperBackbone', method=e.method,
                model_name='deit_small_patch16_224.fb_in1k' if e.backbone == 'small' else 'deit_base_patch16_224.fb_in1k',
                image_size=crop, pretrained_checkpoint=str(pretrained), pretrained_sha256=pretrained_sha256,
                drop_path_rate=.1, relay_options=relay_options(e)),
            decode_head=dict(type='UPerHead', in_channels=[channels]*4, in_index=[0,1,2,3],
                pool_scales=(1,2,3,6), channels=512, dropout_ratio=.1, num_classes=classes,
                norm_cfg=deepcopy(norm), align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1., avg_non_ignore=True)),
            auxiliary_head=dict(type='FCNHead', in_channels=channels, in_index=2, channels=256,
                num_convs=1, concat_input=False, dropout_ratio=.1, num_classes=classes,
                norm_cfg=deepcopy(norm), align_corners=False,
                loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=.4, avg_non_ignore=True)),
            train_cfg=dict(), test_cfg=dict(mode='slide', crop_size=(crop,crop), stride=(stride,stride))),
        train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=deepcopy(val_loader),
        val_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU'], collect_device='cpu'),
        test_evaluator=dict(type='IoUMetric', iou_metrics=['mIoU'], collect_device='cpu'),
        train_cfg=dict(type='IterBasedTrainLoop', max_iters=e.steps, val_interval=20000),
        val_cfg=dict(type='ValLoop'), test_cfg=dict(type='TestLoop'),
        optim_wrapper=dict(type='AmpOptimWrapper', dtype='bfloat16', loss_scale=1., accumulative_counts=1,
            clip_grad=dict(max_norm=1., norm_type=2),
            optimizer=dict(type='AdamW', lr=6e-5, betas=(.9,.999), eps=1e-8, weight_decay=.01),
            paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0., custom_keys={
                'pos_embed':dict(decay_mult=0.), 'cls_token':dict(decay_mult=0.),
                'context.raw_beta':dict(decay_mult=0.)})),
        param_scheduler=[dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
                         dict(type='PolyLR', eta_min=0., power=1., by_epoch=False, begin=1500, end=e.steps)],
        default_hooks=dict(timer=dict(type='IterTimerHook'), logger=dict(type='LoggerHook', interval=50,
            log_metric_by_epoch=False), param_scheduler=dict(type='ParamSchedulerHook'),
            checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000,
                            save_last=True, max_keep_ckpts=-1, save_best=None),
            sampler_seed=dict(type='DistSamplerSeedHook')),
        custom_hooks=[dict(type='PaperAuditHook')],
        env_cfg=dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
                     dist_cfg=dict(backend='nccl')),
        randomness=dict(seed=runtime['seed'], diff_rank_seed=False, deterministic=False),
        log_processor=dict(by_epoch=False), log_level='INFO',
        visualizer=dict(type='SegLocalVisualizer', vis_backends=[dict(type='LocalVisBackend')]),
        auto_scale_lr=dict(enable=False, base_batch_size=runtime['global_batch']), find_unused_parameters=False,
        resume=False, load_from=None, work_dir=str(work_dir), launcher='pytorch')
    if e.id == 'M19':
        cfg['paper']['comparison_reference'] = 'M03'
    if e.id in ('M21', 'M23'):
        cfg['paper']['comparison_reference'] = 'M03'
        cfg['paper']['architecture_reference'] = 'M16'
        cfg['model']['type'] = 'ADPCPFRegionPreservingSegmentor'
        cfg['model']['region_preservation'] = dict(enabled=True, margin=.5,
            max_weight=.05, warmup_iters=10000, boundary_radius=3, small_area_fraction=.005)
    if e.id == "M23":
        cfg["paper"]["architecture_reference"] = "M21"
    return cfg


def make_config(name, data_root, pretrained, work_dir, pretrained_sha256=None, *,
                world_size=WORLD, per_gpu=PER_GPU, global_batch=None, seed=SEED):
    digest = pretrained_sha256 or (PRETRAIN_SHA256 if get_experiment(name).backbone == 'small' else None)
    return _build(name, data_root, pretrained, work_dir, digest, world_size=world_size,
                  per_gpu=per_gpu, global_batch=global_batch, seed=seed)


def _as_smoke(cfg, work_dir):
    out = deepcopy(cfg)
    out['paper']['smoke'] = True
    out['work_dir'] = str(work_dir)
    out['train_cfg'].update(max_iters=200, val_interval=200)
    out['default_hooks']['checkpoint']['interval'] = 200
    for key in ('val_dataloader', 'test_dataloader'):
        out[key]['dataset']['indices'] = list(range(16))
    return out


def smoke_config(cfg, work_dir):
    validate_config(cfg)
    return _as_smoke(cfg, work_dir)


def _as_diagnostic(cfg, work_dir):
    if cfg['paper']['experiment'] != 'M21':
        raise ValueError('This engineering diagnostic is registered only for M21')
    out = deepcopy(cfg)
    out['paper']['engineering_diagnostic'] = True
    out['work_dir'] = str(work_dir)
    out['train_cfg'].update(max_iters=2000, val_interval=2000)
    out['default_hooks']['checkpoint']['interval'] = 2000
    for key in ('val_dataloader', 'test_dataloader'):
        out[key]['dataset']['indices'] = list(range(16))
    return out


def diagnostic_config(cfg, work_dir):
    validate_config(cfg)
    return _as_diagnostic(cfg, work_dir)


def validate_config(cfg, *, allow_smoke=False, allow_diagnostic=False):
    try:
        runtime = runtime_from_config(cfg)
        meta = cfg['paper']
        expected = _build(meta['experiment'], cfg['train_dataloader']['dataset']['data_root'],
            cfg['model']['backbone']['pretrained_checkpoint'], cfg['work_dir'], meta['pretrained_sha256'], **runtime)
        # Archived hierarchical FDAM configs remain exactly reproducible; new
        # registrations explicitly identify the corrected pre-DropPath order.
        if (meta.get('competition') and meta.get('experiment') in ('M43', 'M47')
                and meta.get('method') == 'adapted_fdam'
                and 'implementation_revision' not in meta
                and 'fdam_frequency_order' not in meta
                and 'fdam_order' not in cfg['model']['backbone']):
            expected['model']['backbone'].pop('fdam_order', None)
            expected['paper'].pop('fdam_frequency_order', None)
            expected['paper'].pop('implementation_revision', None)
        if meta['smoke'] is True and allow_smoke:
            expected = _as_smoke(expected, cfg['work_dir'])
        elif meta.get('engineering_diagnostic', False) and allow_diagnostic:
            expected = _as_diagnostic(expected, cfg['work_dir'])
        if cfg != expected:
            raise ValueError('Registered configuration changed; create a new protocol, not an override')
    except (KeyError, TypeError) as exc:
        raise ValueError('Incomplete paper configuration') from exc
