from mmengine.config import read_base

with read_base():
    from mmpose.configs._base_.default_runtime import *  # noqa

from mmcv.transforms import RandomChoice, RandomChoiceResize
from mmengine.dataset import DefaultSampler
from mmengine.optim import LinearLR, MultiStepLR
from torch.nn import GroupNorm
from torch.optim import Adam

from mmpose.codecs import EDPoseLabel
from mmpose.datasets import (BottomupRandomChoiceResize, BottomupRandomCrop,
                             CocoDataset, LoadImage, PackPoseInputs,
                             RandomFlip)
from mmpose.evaluation import CocoMetric
from mmpose.models import (BottomupPoseEstimator, ChannelMapper,
                           PoseDataPreprocessor, ResNet, GroupHead, RTDETR, RTPoseHead)
from mmpose.models.utils import FrozenBatchNorm2d

# runtime
train_cfg.update(max_epochs=50, val_interval=10)  # noqa
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2)
)
# learning policy
param_scheduler = [
    dict(type=LinearLR, begin=0, end=500, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(
        type=MultiStepLR,
        begin=0,
        end=140,
        milestones=[33, 45],
        gamma=0.1,
        by_epoch=True)
]
# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=80)

widen_factor = 0.5
deepen_factor = 0.33
# hooks
default_hooks.update(  # noqa
    checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(type=EDPoseLabel, num_select=50, num_keypoints=17)

# model settings
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
        type='CSPDarknetV8',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        # init_cfg=dict(
        #     type='Pretrained',
        #     checkpoint='https://download.openmmlab.com/mmdetection/v2.0/'
        #     'yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_'
        #     '20211121_095711-4592a793.pth',
        #     prefix='backbone.',
        # )
    ),
    neck=dict(
        type='HybridEncoder',
        in_channels=[128, 256, 256],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[0,1, 2],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=1024,
                ffn_drop=0.0,
                act_cfg=dict(type='GELU'))),
        projector=dict(
            type='ChannelMapper',
            in_channels=[256, 256, 256],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='BN'),
            num_outs=3)
    ),
    head=dict(
        type='RTPoseHead',
        num_classes=80,
        sync_cls_avg_factor=True,
        encoder=None,
        decoder=dict(
            num_layers=3,
            eval_idx=-1,
            layer_cfg=dict(
                self_attn_cfg=dict(embed_dims=256, num_heads=8,
                                   dropout=0.0),  # 0.1 for DeformDETR
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_levels=3,  # 4 for DeformDETR
                    dropout=0.0),  # 0.1 for DeformDETR
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=1024,  # 2048 for DINO
                    ffn_drop=0.0)),  # 0.1 for DeformDETR
            post_norm_cfg=None),
        out_head=dict(num_classes=2),
        positional_encoding=dict(
            num_pos_feats=128,
            temperatureH=20,
            temperatureW=20,
            normalize=True),
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=1.0,  # 0.4 for DN-DETR
            group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        loss_cls=dict(
            type='RTVarifocalLoss',
            use_sigmoid=True,
            use_rtdetr=True,
            gamma=2.0,
            alpha=0.75,  # 0.25 in DINO
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        data_decoder=codec),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# enable DDP training when rescore net is used
find_unused_parameters = True

# base dataset settings
# data_root = '/kaggle/input/cow-pose-coco/Cow/'
data_root = 'dataset/Cow/'
data_mode = 'bottomup'
dataset_type = 'CowposeDataset'

# pipelines
train_pipeline = [
    dict(type=LoadImage),
    dict(type=RandomFlip, direction='horizontal'),
    dict(
        type=RandomChoice,
        transforms=[
            [
                dict(
                    type=RandomChoiceResize,
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type=BottomupRandomChoiceResize,
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type=BottomupRandomCrop,
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type=BottomupRandomChoiceResize,
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type=PackPoseInputs),
]

val_pipeline = [
    dict(type=LoadImage),
    dict(
        type=BottomupRandomChoiceResize,
        scales=[(800, 1333)],
        keep_ratio=True,
        backend='pillow'),
    dict(
        type=PackPoseInputs,
        meta_keys=('id', 'img_id', 'img_path', 'crowd_index', 'ori_shape',
                   'img_shape', 'input_size', 'input_center', 'input_scale',
                   'flip', 'flip_direction', 'flip_indices', 'raw_ann_info',
                   'skeleton_links'))
]

# data loaders
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 标注文件路径为 {data_root}/{ann_file}
        # 例如： aaa/annotations/xxx.json
        ann_file='train/train.json',
        data_mode=data_mode,
        data_prefix=dict(img='train/img'),
        # 指定元信息配置文件
        metainfo=dict(from_file='configs/_base_/datasets/cowpose.py'),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='val/val.json',
        data_prefix=dict(img='val/img'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type=CocoMetric,
    nms_mode='none',
    score_mode='keypoint',
    ann_file=data_root + '/val/val.json'
)
test_evaluator = val_evaluator

