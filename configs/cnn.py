_base_ = ['_base_/default_runtime.py']

#data_root = '/kaggle/input/cow-pose-coco/Cow/'
data_root = '/Users/apple/Desktop/mmpose/dataset/Cow/'
data_mode = 'bottomup'
dataset_type = 'CowposeDataset'

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

optim_wrapper = dict(
    type='OptimWrapper',
    constructor='ForceDefaultOptimWrapperConstructor',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
    clip_grad=dict(max_norm=0.1, norm_type=2))

input_size = (640, 640)

train_pipeline_stage1 = [
    dict(type='LoadImage', backend_args=None),]

val_pipeline = [
    dict(type='LoadImage'),
    dict(
        type='BottomupResize', input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type='PackPoseInputs')
]


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