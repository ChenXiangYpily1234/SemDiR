# SemDiR source-only recipe. Resolve asset paths with tools/run.py.
auto_scale_lr = {'base_batch_size': 16, 'enable': False}

custom_imports = {'allow_failed_imports': False,
 'imports': ['research.ad_pcpf_v2.adapter',
             'research.ad_pcpf_v2.engine',
             'competitors.registration',
             'competitors.semdir_a2_main',
             'semdir_release.hooks']}

default_hooks = {'checkpoint': {'by_epoch': False,
                'interval': 20000,
                'max_keep_ckpts': -1,
                'save_best': None,
                'save_last': True,
                'type': 'CheckpointHook'},
 'logger': {'interval': 50, 'log_metric_by_epoch': False, 'type': 'LoggerHook'},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'timer': {'type': 'IterTimerHook'}}

default_scope = 'mmseg'

env_cfg = {'cudnn_benchmark': False,
 'dist_cfg': {'backend': 'nccl'},
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0}}

find_unused_parameters = False

launcher = 'none'

load_from = None

log_level = 'INFO'

log_processor = {'by_epoch': False}

model = {'auxiliary_head': {'align_corners': False,
                    'channels': 256,
                    'concat_input': False,
                    'dropout_ratio': 0.1,
                    'in_channels': 384,
                    'in_index': 2,
                    'loss_decode': {'avg_non_ignore': True,
                                    'loss_weight': 0.4,
                                    'type': 'CrossEntropyLoss',
                                    'use_sigmoid': False},
                    'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                    'num_classes': 150,
                    'num_convs': 1,
                    'type': 'FCNHead'},
 'backbone': {'direct_hidden': 32,
              'pretrained_checkpoint': '${PRETRAIN_ROOT}/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth',
              'pretrained_sha256': '3f89aee6e380026163f9f709ed14fc1726e0f8e2578d1f2d12bb465bbc8ae237',
              'type': 'A2SwinBackbone'},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_val': 0,
                       'seg_pad_val': 255,
                       'size': (512, 512),
                       'std': [58.395, 57.12, 57.375],
                       'test_cfg': {'size_divisor': 16},
                       'type': 'SegDataPreProcessor'},
 'decode_head': {'align_corners': False,
                 'channels': 512,
                 'dropout_ratio': 0.1,
                 'in_channels': [96, 192, 384, 768],
                 'in_index': [0, 1, 2, 3],
                 'loss_decode': {'avg_non_ignore': True,
                                 'loss_weight': 1.0,
                                 'type': 'CrossEntropyLoss',
                                 'use_sigmoid': False},
                 'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                 'num_classes': 150,
                 'pool_scales': (1, 2, 3, 6),
                 'type': 'UPerHead'},
 'region_preservation': {'boundary_radius': 3,
                         'enabled': True,
                         'margin': 0.5,
                         'max_weight': 0.05,
                         'small_area_fraction': 0.005,
                         'warmup_iters': 10000},
 'small_ce_normalization': 'global_eligible_images',
 'small_ce_weight': 0.0,
 'test_cfg': {'crop_size': (512, 512), 'mode': 'slide', 'stride': (341, 341)},
 'train_cfg': {},
 'type': 'A2SwinSegmentor'}

optim_wrapper = {'accumulative_counts': 1,
 'clip_grad': {'max_norm': 1.0, 'norm_type': 2},
 'dtype': 'bfloat16',
 'loss_scale': 1.0,
 'optimizer': {'betas': (0.9, 0.999),
               'eps': 1e-08,
               'lr': 6e-05,
               'type': 'AdamW',
               'weight_decay': 0.01},
 'paramwise_cfg': {'bias_decay_mult': 0.0,
                   'custom_keys': {'cls_token': {'decay_mult': 0.0},
                                   'context.raw_beta': {'decay_mult': 0.0},
                                   'pos_embed': {'decay_mult': 0.0}},
                   'norm_decay_mult': 0.0},
 'type': 'AmpOptimWrapper'}

param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 1500, 'start_factor': 1e-06, 'type': 'LinearLR'},
 {'begin': 1500, 'by_epoch': False, 'end': 160000, 'eta_min': 0.0, 'power': 1.0, 'type': 'PolyLR'}]

randomness = {'deterministic': False, 'diff_rank_seed': False, 'seed': 1}

resume = False

test_cfg = {'type': 'TestLoop'}

test_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': 'images/validation',
                             'seg_map_path': 'annotations/validation'},
             'data_root': '${DATA_ROOT}',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (2048, 512), 'type': 'Resize'},
                          {'reduce_zero_label': True, 'type': 'LoadAnnotations'},
                          {'type': 'PackSegInputs'}],
             'reduce_zero_label': True,
             'type': 'ADE20KDataset'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

test_evaluator = {'collect_device': 'cpu', 'iou_metrics': ['mIoU'], 'type': 'IoUMetric'}

train_cfg = {'max_iters': 160000, 'type': 'IterBasedTrainLoop', 'val_interval': 20000}

train_dataloader = {'batch_size': 4,
 'dataset': {'data_prefix': {'img_path': 'images/training', 'seg_map_path': 'annotations/training'},
             'data_root': '${DATA_ROOT}',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'reduce_zero_label': True, 'type': 'LoadAnnotations'},
                          {'keep_ratio': True,
                           'ratio_range': (0.5, 2.0),
                           'scale': (2048, 512),
                           'type': 'RandomResize'},
                          {'cat_max_ratio': 0.75, 'crop_size': (512, 512), 'type': 'RandomCrop'},
                          {'prob': 0.5, 'type': 'RandomFlip'},
                          {'type': 'PhotoMetricDistortion'},
                          {'type': 'PackSegInputs'}],
             'reduce_zero_label': True,
             'type': 'ADE20KDataset'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': True, 'type': 'InfiniteSampler'}}

val_cfg = {'type': 'ValLoop'}

val_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': 'images/validation',
                             'seg_map_path': 'annotations/validation'},
             'data_root': '${DATA_ROOT}',
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (2048, 512), 'type': 'Resize'},
                          {'reduce_zero_label': True, 'type': 'LoadAnnotations'},
                          {'type': 'PackSegInputs'}],
             'reduce_zero_label': True,
             'type': 'ADE20KDataset'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

val_evaluator = {'collect_device': 'cpu', 'iou_metrics': ['mIoU'], 'type': 'IoUMetric'}

visualizer = {'type': 'SegLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}

custom_hooks = [{'type': 'SemDiRIterationHook'}]

