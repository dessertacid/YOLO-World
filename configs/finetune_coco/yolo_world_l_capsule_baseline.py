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

max_epochs = 80
close_mosaic_epochs = 10
base_lr = 2e-4
train_batch_size_per_gpu = 4   # 先降到 4，方便排查数值稳定性

load_from = 'weights/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth'

# 关键：类别相关字段要一起改
model = dict(
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
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

train_pipeline = [
    *_base_.pre_transform,
    *_base_.mosaic_affine_transform,
    dict(
        type='YOLOv5MultiModalMixUp',
        prob=_base_.mixup_prob,
        pre_transform=[*_base_.pre_transform, *_base_.mosaic_affine_transform]),
    *_base_.last_transform[:-1],
    *text_transform,
]

train_pipeline_stage2 = [
    *_base_.train_pipeline_stage2[:-2],
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
        pipeline=_base_.test_pipeline
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
