# =========================
# capsule baseline config
# =========================

# 优先用这个；如果你本地没有这个文件，再看下面“备用方案”
_base_ = 'yolo_world_l_dual_vlpan_2e-4_80e_8gpus_finetune_coco.py'

custom_imports = dict(
    imports=['yolo_world'],
    allow_failed_imports=False
)

class_name = (
    'color repetition',
    'crack',
    'depression',
    'empty capsules',
    'black spot'
)

num_classes = len(class_name)
num_training_classes = num_classes

data_root = 'dataset/coco_capsule-5label-300/'
class_text_path = data_root + 'class_texts.json'

fusion_type = 'add'

max_epochs = 80
close_mosaic_epochs = 10
base_lr = 2e-4
train_batch_size_per_gpu = 4   # 先降到 4，方便排查数值稳定性

load_from = 'weights/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth'

# 关键：类别相关字段要一起改
model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(
        _delete_=True,
        type='YOLOWDetDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False),
    backbone=dict(
        depth_fusion=dict(
            type='DepthFeatureFusion',
            fusion_type=fusion_type,
            fusion_indices=(1, ),
            depth_in_channels=1)),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_training_classes
        )
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_training_classes)
    )
)

# 关键：text transform 也要和 5 类对齐
text_transform = [
    dict(
        type='RandomLoadText',
        num_neg_samples=(num_classes, num_classes),
        max_num_samples=num_training_classes,
        padding_to_max=True,
        padding_value=''
    ),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'flip', 'flip_direction', 'texts')
    )
]

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadDepthAndFuse',
         img_dirname='images',
         depth_dirname='depth',
         depth_ext='.png',
         mode='raw_concat'),
    dict(type='LoadAnnotations', with_bbox=True)
]

mosaic_affine_transform = [
    dict(
        type='MultiModalMosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100.,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114))
]

train_pipeline = [
    *pre_transform,
    *mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=_base_.mixup_prob,
        pre_transform=[*pre_transform, *mosaic_affine_transform]),
    dict(type='mmdet.RandomFlip', prob=0.5),
    *text_transform,
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=_base_.img_scale),
    dict(
        type='LetterResize',
        scale=_base_.img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale,
                             1 + _base_.affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)),
    dict(type='mmdet.RandomFlip', prob=0.5),
    *text_transform,
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolow_collate'),
    dataset=dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root,
            ann_file='annotations/instances_train2017.json',
            data_prefix=dict(img='train2017/images/'),
            metainfo=dict(classes=class_name),
            filter_cfg=dict(filter_empty_gt=False, min_size=32)
        ),
        class_text_path=class_text_path,
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type='MultiModalDataset',
        dataset=dict(
            type='YOLOv5CocoDataset',
            data_root=data_root,
            ann_file='annotations/instances_val2017.json',
            data_prefix=dict(img='val2017/images/'),
            metainfo=dict(classes=class_name),
            test_mode=True
        ),
        class_text_path=class_text_path,
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
            dict(type='LoadDepthAndFuse',
                 img_dirname='images',
                 depth_dirname='depth',
                 depth_ext='.png',
                 mode='raw_concat'),
            dict(scale=_base_.img_scale, type='YOLOv5KeepRatioResize'),
            dict(
                allow_scale_up=False,
                pad_val=dict(img=114),
                scale=_base_.img_scale,
                type='LetterResize'),
            dict(_scope_='mmdet', type='LoadAnnotations', with_bbox=True),
            dict(type='LoadText'),
            dict(
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param', 'texts'),
                type='mmdet.PackDetInputs'),
        ]
    )
)

# 你当前没有 test annotation，先不要配 test_dataloader / test_evaluator

val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val2017.json'
)

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=5,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs),
                        _base_.val_interval_stage2)]
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2)
]

default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3, save_best='auto')
)

optim_wrapper = dict(
    optimizer=dict(lr=base_lr)
)
