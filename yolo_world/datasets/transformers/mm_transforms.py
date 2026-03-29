# Copyright (c) Tencent Inc. All rights reserved.
import json
import random
from typing import Tuple

import numpy as np
import cv2
import os
from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomLoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 num_neg_samples: Tuple[int, int] = (80, 80),
                 max_num_samples: int = 80,
                 padding_to_max: bool = False,
                 padding_value: str = '') -> None:
        self.prompt_format = prompt_format
        self.num_neg_samples = num_neg_samples
        self.max_num_samples = max_num_samples
        self.padding_to_max = padding_to_max
        self.padding_value = padding_value
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        num_classes = len(class_texts)
        if 'gt_labels' in results:
            gt_label_tag = 'gt_labels'
        elif 'gt_bboxes_labels' in results:
            gt_label_tag = 'gt_bboxes_labels'
        else:
            raise ValueError('No valid labels found in results.')
        positive_labels = set(results[gt_label_tag])

        if len(positive_labels) > self.max_num_samples:
            positive_labels = set(random.sample(list(positive_labels),
                                  k=self.max_num_samples))

        num_neg_samples = min(
            min(num_classes, self.max_num_samples) - len(positive_labels),
            random.randint(*self.num_neg_samples))
        candidate_neg_labels = []
        for idx in range(num_classes):
            if idx not in positive_labels:
                candidate_neg_labels.append(idx)
        negative_labels = random.sample(
            candidate_neg_labels, k=num_neg_samples)

        sampled_labels = list(positive_labels) + list(negative_labels)
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}

        gt_valid_mask = np.zeros(len(results['gt_bboxes']), dtype=bool)
        for idx, label in enumerate(results[gt_label_tag]):
            if label in label2ids:
                gt_valid_mask[idx] = True
                results[gt_label_tag][idx] = label2ids[label]
        results['gt_bboxes'] = results['gt_bboxes'][gt_valid_mask]
        results[gt_label_tag] = results[gt_label_tag][gt_valid_mask]

        if 'instances' in results:
            retaged_instances = []
            for idx, inst in enumerate(results['instances']):
                label = inst['bbox_label']
                if label in label2ids:
                    inst['bbox_label'] = label2ids[label]
                    retaged_instances.append(inst)
            results['instances'] = retaged_instances

        texts = []
        for label in sampled_labels:
            cls_caps = class_texts[label]
            assert len(cls_caps) > 0
            cap_id = random.randrange(len(cls_caps))
            sel_cls_cap = self.prompt_format.format(cls_caps[cap_id])
            texts.append(sel_cls_cap)

        if self.padding_to_max:
            num_valid_labels = len(positive_labels) + len(negative_labels)
            num_padding = self.max_num_samples - num_valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        results['texts'] = texts

        return results


@TRANSFORMS.register_module()
class LoadText:

    def __init__(self,
                 text_path: str = None,
                 prompt_format: str = '{}',
                 multi_prompt_flag: str = '/') -> None:
        self.prompt_format = prompt_format
        self.multi_prompt_flag = multi_prompt_flag
        if text_path is not None:
            with open(text_path, 'r') as f:
                self.class_texts = json.load(f)

    def __call__(self, results: dict) -> dict:
        assert 'texts' in results or hasattr(self, 'class_texts'), (
            'No texts found in results.')
        class_texts = results.get(
            'texts',
            getattr(self, 'class_texts', None))

        texts = []
        for idx, cls_caps in enumerate(class_texts):
            assert len(cls_caps) > 0
            sel_cls_cap = cls_caps[0]
            sel_cls_cap = self.prompt_format.format(sel_cls_cap)
            texts.append(sel_cls_cap)

        results['texts'] = texts

        return results


@TRANSFORMS.register_module()
class LoadDepthAndFuse:

    def __init__(self,
                 img_dirname: str = 'images',
                 depth_dirname: str = 'depth',
                 depth_ext: str = None,
                 mode: str = 'alpha_blend',
                 alpha: float = 0.5,
                 depth_scale: float = 1.0,
                 imread_flag: int = cv2.IMREAD_UNCHANGED,
                 ignore_missing: bool = False) -> None:
        self.img_dirname = img_dirname
        self.depth_dirname = depth_dirname
        self.depth_ext = depth_ext
        self.mode = mode
        self.alpha = float(alpha)
        self.depth_scale = float(depth_scale)
        self.imread_flag = imread_flag
        self.ignore_missing = ignore_missing

    def _infer_depth_path(self, img_path: str) -> str:
        img_dir, img_name = os.path.split(img_path)
        stem, ext = os.path.splitext(img_name)
        depth_ext = self.depth_ext if self.depth_ext is not None else ext
        depth_dir = img_dir.replace(f'{os.sep}{self.img_dirname}{os.sep}',
                                    f'{os.sep}{self.depth_dirname}{os.sep}')
        return os.path.join(depth_dir, stem + depth_ext)

    def __call__(self, results: dict) -> dict:
        if 'img' not in results:
            raise KeyError('LoadDepthAndFuse requires results["img"] to exist. '
                           'Please place it after LoadImageFromFile.')
        img = results['img']
        if img is None:
            raise ValueError('results["img"] is None.')

        img_path = results.get('img_path', None)
        if img_path is None:
            raise KeyError('LoadDepthAndFuse requires results["img_path"] to '
                           'exist to locate depth image.')

        depth_path = results.get('depth_path', None)
        if depth_path is None:
            depth_path = self._infer_depth_path(img_path)

        depth = cv2.imread(depth_path, self.imread_flag)
        if depth is None:
            if self.ignore_missing:
                return results
            raise FileNotFoundError(
                f'Failed to read depth image: {depth_path} (from img_path={img_path})'
            )

        if depth.ndim == 3:
            depth = depth[:, :, 0]

        if depth.shape[:2] != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]),
                               interpolation=cv2.INTER_NEAREST)

        if self.mode in {'raw_add', 'raw_concat'}:
            if depth.dtype == np.uint16:
                depth_u8 = (depth.astype(np.float32) / 256.0).astype(np.uint8)
            elif depth.dtype == np.uint8:
                depth_u8 = depth
            else:
                depth_f = depth.astype(np.float32)
                if depth_f.max() <= 1.0:
                    depth_f = depth_f * 255.0
                depth_u8 = np.clip(depth_f, 0.0, 255.0).astype(np.uint8)

            if self.mode == 'raw_concat':
                depth_1c = depth_u8[:, :, None]
                results['img'] = np.concatenate([img, depth_1c], axis=2)
            else:
                depth_rgb = depth_u8[:, :, None].repeat(3, axis=2).astype(np.float32)
                fused = img.astype(np.float32) + depth_rgb * self.depth_scale
                results['img'] = np.clip(fused, 0.0, 255.0).astype(np.uint8)
        else:
            depth_f = depth.astype(np.float32)
            if depth_f.max() > 255:
                depth_f = depth_f / 65535.0
            else:
                depth_f = depth_f / 255.0
            depth_f = np.clip(depth_f, 0.0, 1.0)

            depth_rgb = (depth_f[:, :, None] * 255.0).repeat(3, axis=2)
            fused = img.astype(np.float32) * (1.0 - self.alpha) + depth_rgb * self.alpha
            results['img'] = np.clip(fused, 0.0, 255.0).astype(np.uint8)

        return results
