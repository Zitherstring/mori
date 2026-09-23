# MORI-seg: Object-Aware RTMDet-Ins-L, 100 epochs from the COCO pretrained checkpoint.
# Set `data_root` below to your COCO-format dataset (6 classes: cap, dt, pt, ptc, tuft, ves).
custom_imports = dict(imports=['mori_seg'], allow_failed_imports=False)
auto_scale_lr = dict(base_batch_size=256, enable=True)
backend_args = None
base_lr = 0.004
custom_hooks = [
    dict(
        ema_type='MORIExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=80,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    640,
                    640,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    640,
                    640,
                ),
                recompute_bbox=True,
                type='RandomCrop'),
            dict(min_gt_bbox_wh=(
                1,
                1,
            ), type='FilterAnnotations'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
    dict(
        filename_tmpl='epoch_{}_after80.pth',
        interval=1,
        max_keep_ckpts=-1,
        save_begin=80,
        save_best=None,
        save_last=False,
        type='CheckpointHook'),
]
data_root = 'data/KI_82/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=5,
        save_best='coco/segm_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_l_8xb32-300e_coco/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 100
metainfo = dict(
    classes=(
        'cap',
        'dt',
        'pt',
        'ptc',
        'tuft',
        'ves',
    ),
    palette=[
        (
            220,
            20,
            60,
        ),
        (
            0,
            255,
            0,
        ),
        (
            0,
            0,
            255,
        ),
        (
            255,
            255,
            0,
        ),
        (
            255,
            0,
            255,
        ),
        (
            0,
            255,
            255,
        ),
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=1.0,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        type='CSPNeXt',
        widen_factor=1.0),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        lambda_objaware_dis=1.0,
        lambda_objaware_emb=5.0,
        lambda_objaware_reg=1.0,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_mask=dict(
            eps=5e-06, loss_weight=2.0, reduction='mean', type='DiceLoss'),
        neighbor_distance=10.0,
        neighbor_distance_is_input=True,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=6,
        objaware_balance_reg_fg_bg=True,
        objaware_bg_reg_weight=1.0,
        objaware_boundary_focal_alpha=0.25,
        objaware_boundary_focal_gamma=2.0,
        objaware_boundary_hidden_channels=32,
        objaware_boundary_loss_type='bce',
        objaware_boundary_loss_weight=0.2,
        objaware_boundary_pos_weight=2.0,
        objaware_boundary_suppress_gamma=0.5,
        objaware_boundary_suppress_infer=False,
        objaware_boundary_warmup_iters=2000,
        objaware_boundary_width=2,
        objaware_boundary_width_is_input=False,
        objaware_dist_target_exp_alpha=3.0,
        objaware_dist_target_transform='exp',
        objaware_embedding_dim=16,
        objaware_export_map=True,
        objaware_hidden_channels=32,
        objaware_include_background=False,
        objaware_local_constraint=True,
        objaware_min_instance_pixels=4,
        objaware_same_class_only=False,
        objaware_warmup_iters=5000,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='MORIObjectAwareRTMDetInsSepBNHead'),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            256,
            512,
            1024,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=3,
        out_channels=256,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        mask_thr_binary=0.5,
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.6, type='nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
num_classes = 6
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=50,
        begin=50,
        by_epoch=True,
        convert_to_iter_based=True,
        end=100,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
randomness = dict(deterministic=False, seed=42)
resume = False
stage2_num_epochs = 20
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/val.json',
        backend_args=None,
        data_prefix=dict(img='val/images/'),
        data_root='data/KI_82/',
        metainfo=dict(
            classes=(
                'cap',
                'dt',
                'pt',
                'ptc',
                'tuft',
                'ves',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    255,
                    255,
                    0,
                ),
                (
                    255,
                    0,
                    255,
                ),
                (
                    0,
                    255,
                    255,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/KI_82/annotations/val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=100, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=8,
    dataset=dict(
        ann_file='annotations/train.json',
        backend_args=None,
        data_prefix=dict(img='train/images/'),
        data_root='data/KI_82/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'cap',
                'dt',
                'pt',
                'ptc',
                'tuft',
                'ves',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    255,
                    255,
                    0,
                ),
                (
                    255,
                    0,
                    255,
                ),
                (
                    0,
                    255,
                    255,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                poly2mask=False,
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True),
            dict(img_scale=(
                640,
                640,
            ), pad_val=114.0, type='CachedMosaic'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    1280,
                    1280,
                ),
                type='RandomResize'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    640,
                    640,
                ),
                recompute_bbox=True,
                type='RandomCrop'),
            dict(type='YOLOXHSVRandomAug'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(
                img_scale=(
                    640,
                    640,
                ),
                max_cached_images=20,
                pad_val=(
                    114,
                    114,
                    114,
                ),
                ratio_range=(
                    1.0,
                    1.0,
                ),
                type='CachedMixUp'),
            dict(min_gt_bbox_wh=(
                1,
                1,
            ), type='FilterAnnotations'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(img_scale=(
        640,
        640,
    ), pad_val=114.0, type='CachedMosaic'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            1280,
            1280,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            640,
            640,
        ),
        recompute_bbox=True,
        type='RandomCrop'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(
        img_scale=(
            640,
            640,
        ),
        max_cached_images=20,
        pad_val=(
            114,
            114,
            114,
        ),
        ratio_range=(
            1.0,
            1.0,
        ),
        type='CachedMixUp'),
    dict(min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        poly2mask=False,
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            640,
            640,
        ),
        recompute_bbox=True,
        type='RandomCrop'),
    dict(min_gt_bbox_wh=(
        1,
        1,
    ), type='FilterAnnotations'),
    dict(type='YOLOXHSVRandomAug'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/val.json',
        backend_args=None,
        data_prefix=dict(img='val/images/'),
        data_root='data/KI_82/',
        metainfo=dict(
            classes=(
                'cap',
                'dt',
                'pt',
                'ptc',
                'tuft',
                'ves',
            ),
            palette=[
                (
                    220,
                    20,
                    60,
                ),
                (
                    0,
                    255,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
                (
                    255,
                    255,
                    0,
                ),
                (
                    255,
                    0,
                    255,
                ),
                (
                    0,
                    255,
                    255,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/KI_82/annotations/val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/stage1_100e'

