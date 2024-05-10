_base_ = ['../../../_base_/default_runtime.py']

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=10,
    dynamic_intervals=[(195, 1)])

#学习率调整
param_scheduler = [
    dict(
        type='QuadraticWarmupLR',
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=5,
        T_max=280,
        end=280,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=180, end=200),
]

# model
widen_factor = 0.375
deepen_factor = 0.33


optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({'neck.encoder': dict(lr_mult=0.05)})),
    clip_grad=dict(max_norm=0.1, norm_type=2))

input_size = (640, 640)
codec = dict(type='YOLOXPoseAnnotationProcessor', input_size=input_size)



## 数据处理
train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),
    dict(
        type='Mosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution='uniform',
        transform_mode='perspective',
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(
        type='YOLOXMixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        pad_val=114.0,
        pre_transform=[dict(type='LoadImage', backend_args=None)]),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
train_pipeline_stage2 = [
    dict(type='LoadImage'),
    dict(
        type='BottomupRandomAffine',
        input_size=(640, 640),
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type='long',
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip'),
    dict(type='FilterAnnotations', by_kpt=True, by_box=True, keep_empty=False),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs')
]

# dataset设置
data_root = '/kaggle/input/cow-pose-coco/Cow/'
data_mode = 'bottomup'
dataset_type = 'CowposeDataset'

train_dataloader = dict(
# batch_size
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset = dict(
        type=dataset_type,
        data_root=data_root,
        # 标注文件路径为 {data_root}/{ann_file}
        # 例如： aaa/annotations/xxx.json
        ann_file='train/train.json',
        data_mode=data_mode,
        data_prefix=dict(img='train/img'),
        # 指定元信息配置文件
        metainfo=dict(from_file='configs/_base_/datasets/cowpose.py'),
        pipeline=train_pipeline_stage1)
)
val_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='val/val.json',
        data_prefix=dict(img='val/img'),
        test_mode=True,
        metainfo=dict(from_file='configs/_base_/datasets/cowpose.py'),
        pipeline=val_pipeline,
    ))

test_dataloader = val_dataloader


val_evaluator = dict(
    type='CocoMetric',
    ann_file= data_root +'/val/val.json'
)
test_evaluator = val_evaluator


# hooks
custom_hooks = [
    dict(
        type='YOLOXPoseModeSwitchHook',
        num_last_epochs=20,
        new_train_pipeline=train_pipeline_stage2,
        priority=48),
    dict(
        type='RTMOModeSwitchHook',
        epoch_attributes={
            280: {
                'proxy_target_cc': True,
                'loss_mle.loss_weight': 5.0,
                'loss_oks.loss_weight': 10.0
            },
        },
        priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49),
]

model = dict(    
    type='BottomupPoseEstimator',
    init_cfg=dict(
        type='Kaiming',
        layer='Conv2d',
        a=2.23606797749979,
        distribution='uniform',
        mode='fan_in',
        nonlinearity='leaky_relu'),
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type='BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1),
        ]),
   backbone=dict(
        type='CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='https://download.openmmlab.com/mmdetection/v2.0/'
        #     'yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_'
        #     '20211121_095711-4592a793.pth',
        #     prefix='backbone.',
        # )),
    neck=dict(
        type='HybridEncoder',
        in_channels=[128, 256, 512],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[1, 2],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU'))),
        projector=dict(
            type='ChannelMapper',
            in_channels=[256, 256],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=2)),
 head=dict(
        type='YOLOXPoseHead',
        num_keypoints=17,
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=256,
            feat_channels=256,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='Swish')),
        prior_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[16, 32]),
        assigner=dict(
            type='SimOTAAssigner',
            dynamic_k_indicator='oks',
            oks_calculator=dict(type='PoseOKS', metainfo='configs/_base_/datasets/cowpose.py'),
            use_keypoints_for_center=True),
        overlaps_power=0.5,

        loss_cls=dict(
            type='BCELoss',
            reduction='sum',
            loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss',
            mode='square',
            eps=1e-16,
            reduction='sum',
            loss_weight=5.0),
        loss_obj=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='sum',
            loss_weight=1.0),
        loss_oks=dict(
            type='OKSLoss',
            reduction='none',
            metainfo='configs/_base_/datasets/cowpose.py',
            norm_target_weight=True,
            loss_weight=30.0),
        loss_vis=dict(
            type='BCELoss',
            use_target_weight=True,
            reduction='mean',
            loss_weight=1.0),
        loss_mle=dict(
            type='MLECCLoss',
            use_target_weight=True,
            loss_weight=1.0,
        ),
        loss_bbox_aux=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    test_cfg=dict(
        input_size=input_size,
        score_thr=0.1,
        nms_thr=0.65,
    ))
