# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import os
import pathlib
import pickle
from uuid import uuid4

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.comm import is_main_process, all_gather
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
import numpy as np
import pandas as pd
from maskrcnn_benchmark.modeling.poolers import Pooler
from torch.nn import functional as F
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.pooler_box = self.roi_heads.box.feature_extractor.pooler
        if cfg.MODEL.MASK_ON:
            self.pooler_mask = self.roi_heads.mask.feature_extractor.pooler
        else:
            self.pooler_mask = self.pooler_box

        self.is_extract_feature = cfg.TEST.EXTRACT_FEATURE
        self.is_use_feature = cfg.TEST.USE_FEATURE
        self.output_folder = cfg.OUTPUT_DIR + "/features"
        self.df_file = self.output_folder + "/features_df.pickle"
        self.features_df = None
        self.pair_df = None

        if self.is_extract_feature:
            self.init_extract_feature()
        elif self.is_use_feature:
            self.init_use_feature()
            self.shot = cfg.TEST.SHOT
        # if self.training:
        self.support_pooler = Pooler(
            output_size=(192, 192),
            scales=(1.0, ),
            sampling_ratio=2,
        )
        self.match = torch.nn.Conv2d(512, 256, 1)
        self.detections_per_img = cfg.TEST.DETECTIONS_PER_IMG

    def init_extract_feature(self):
        pathlib.Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.output_folder +
                     "/box").mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.output_folder +
                     "/mask").mkdir(parents=True, exist_ok=True)
        if os.path.isfile(self.df_file):
            with open(self.df_file, 'rb') as handle:
                self.features_df = pickle.load(handle)
        else:
            self.features_df = pd.DataFrame(
                columns=["label", "img_id", "ann_id", "box_path", "mask_path"])

    def finalize_extract_feature(self):
        features_df_per_gpu = self.features_df
        all_features_df = all_gather(features_df_per_gpu)
        if not is_main_process():
            return
        pd.concat(all_features_df).to_pickle(self.df_file)

    def init_use_feature(self):
        with open(self.df_file, 'rb') as handle:
            self.features_df = pickle.load(handle)
        self.pair_df = pd.DataFrame(columns=["img_id", "support_ann_ids"])

    def finalize_use_feature(self):
        pair_df_per_gpu = self.pair_df
        all_pair_df = all_gather(pair_df_per_gpu)
        if not is_main_process():
            return
        pd.concat(all_pair_df).to_pickle(
            self.output_folder + "/all_pair_df.pickle")

    def __del__(self):
        if self.is_extract_feature:
            self.finalize_extract_feature()
        if self.is_use_feature:
            self.finalize_use_feature()

    def save_features(self, rois_box, target):
        labels = target.get_field("labels")
        assert len(labels) == len(rois_box)
        img_ids = target.get_field("img_ids")
        ann_ids = target.get_field("ann_ids")
        for label, roi_box, img_id, ann_id in zip(labels, rois_box, img_ids, ann_ids):
            box_path = "{0}/box/{1}".format(self.output_folder, str(uuid4()))
            self.features_df = self.features_df.append(
                {"label": int(label.cpu()), "img_id": int(img_id.cpu()),
                 "ann_id": int(ann_id.cpu()), "box_path": box_path}, ignore_index=True)

            with open(box_path, "wb") as handle:
                torch.save(roi_box.cpu(), handle)

    def load_from_df(self, target, labels):
        boxes = []
        masks = []
        img_id = target.get_field("img_ids")[0]
        support_ann_ids = []
        for i in torch.unique(labels):
            item = self.features_df[self.features_df.label == int(i.cpu())]
            item = item[item.img_id != img_id]
            num_of_sample = min(self.shot, len(item))
            sample_items = item.sample(num_of_sample)
            support_ann_ids += sample_items.ann_id.to_list()
            box_paths = sample_items.box_path.to_list()
            roi_box_s = [torch.load(f).cuda() for f in box_paths]
            if len(roi_box_s) > 0:
                roi_box_s = torch.stack(roi_box_s).mean(0)
            else:
                raise RuntimeError("Error: label {0} doesn't have support".format(i))

            boxes.append(roi_box_s)

        self.pair_df = self.pair_df.append(
            {"img_id": int(img_id.cpu()), "support_ann_ids": support_ann_ids},
            ignore_index=True)

        boxes = torch.stack(boxes)
        return boxes

    def intersect1d(self, a, b):
        label = np.intersect1d(a.cpu(), b.cpu())
        if label.size > 0:
            label = np.random.choice(label)
            return a == label, b == label
        else:
            return a == a[0], b == b[0]
        
    # def extract_support_feature(idx_s)
    #     idx_s = (idx_s > 0).nonzero().flatten()
    #     idx_s = [torch.randperm(idx_s.size(0))[:1]]

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        labels = [p.get_field("labels").long() for p in targets]
        if self.training:
            sampled_targets = []
            for i in range(0, 4, 2):
                idx_q, idx_s = self.intersect1d(labels[i], labels[i + 1])
                targets[i] = targets[i][idx_q]
                targets[i + 1] = targets[i + 1][idx_s]
                sampled_id = np.random.choice(len(targets[i + 1]))
                sampled_targets.append(targets[i + 1][[sampled_id]])
            support_images = self.support_pooler(
                [images.tensors[[1, 3]]], sampled_targets)
            support_features = self.backbone(support_images)
            query_features = self.backbone(images.tensors[[0, 2]])
            support_features = support_features[0]
            support_features = F.adaptive_avg_pool2d(support_features, 1)
            targets = [targets[0], targets[2]]
            images.image_sizes = [images.image_sizes[0], images.image_sizes[2]]
        else:
            sampled_targets = targets

            support_images = self.support_pooler([images.tensors], sampled_targets)
            support_features = self.backbone(support_images)
            query_features = self.backbone(images.tensors)
            support_features = support_features[0]
            support_features = F.adaptive_avg_pool2d(support_features, 1)
        if self.training:
            features = [torch.cat([q, q - support_features], 1) for q in query_features]
            features = [self.match(f) for f in features]
        else:
            if self.is_extract_feature:
                self.save_features(support_features, targets[0])
                return

            if self.is_use_feature:
                support_features = self.load_from_df(targets[0], labels[0])
                all_results = []
                for support_features_per_class in support_features:
                    support_features_per_class = support_features_per_class.unsqueeze(0)
                    features = [torch.cat([q, q - support_features_per_class], 1) for q in query_features]
                    features = [self.match(f) for f in features]
                    # all_features.append(temp)

                # for features in all_features:
                    proposals, proposal_losses = self.rpn(images, features, targets)
                    # all_proposals.append(proposals)
                    x, result, detector_losses = self.roi_heads(features, proposals, targets)
                    all_results.append(result[0])
                unique_labels = torch.unique(labels[0])
                unique_labels = torch.cat([l.repeat(len(r)) for l, r in zip(unique_labels, all_results)])
                all_results = cat_boxlist(all_results)
                all_results.add_field("labels", unique_labels)
                cls_scores = all_results.get_field("scores")
                number_of_detections = len(all_results)
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                all_results = all_results[keep]
                return [all_results]

        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result

    def prepare_roi_list(self, roi, targets_per_img, labels):
        if self.training:
            roi_q, roi_s = roi.split(targets_per_img)
            roi_q = self.calculate_prototype(roi_q, labels[0])
            roi_s = self.calculate_prototype(roi_s, labels[1])
            return roi_q, roi_s

    def calculate_prototype(self, features, labels):
        if not features.numel():
            return features

        unique_labels = torch.unique(labels)
        if unique_labels.numel() == 1:
            return features.mean(0, keepdim=True)

        avg_features = [features[labels == l].mean(0) for l in unique_labels]
        avg_features = torch.stack(avg_features)
        return avg_features
