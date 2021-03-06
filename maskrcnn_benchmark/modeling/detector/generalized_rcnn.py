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

import pandas as pd

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

    def save_features(self, rois_box, rois_mask, target):
        labels = target.get_field("labels")
        assert len(labels) == len(rois_box)
        assert len(rois_box) == len(rois_mask)
        img_ids = target.get_field("img_ids")
        ann_ids = target.get_field("ann_ids")
        for label, roi_box, roi_mask, img_id, ann_id in zip(labels, rois_box, rois_mask, img_ids, ann_ids):
            box_path = "{0}/box/{1}".format(self.output_folder, str(uuid4()))
            mask_path = "{0}/mask/{1}".format(self.output_folder, str(uuid4()))
            self.features_df = self.features_df.append(
                {"label": int(label.cpu()), "img_id": int(img_id.cpu()),
                 "ann_id": int(ann_id.cpu()), "box_path": box_path,
                 "mask_path": mask_path}, ignore_index=True)

            with open(box_path, "wb") as handle:
                torch.save(roi_box.cpu(), handle)

            with open(mask_path, "wb") as handle:
                torch.save(roi_mask.cpu(), handle)

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
            mask_paths = sample_items.mask_path.to_list()
            roi_box_s = [torch.load(f).cuda() for f in box_paths]
            roi_mask_s = [torch.load(f).cuda() for f in mask_paths]
            if len(roi_box_s) > 0 and len(roi_mask_s):
                roi_box_s = torch.stack(roi_box_s).mean(0)
                roi_mask_s = torch.stack(roi_mask_s).mean(0)
            else:
                raise RuntimeError("Error: label {0} doesn't have support".format(i))

            boxes.append(roi_box_s)
            masks.append(roi_mask_s)

        self.pair_df = self.pair_df.append(
            {"img_id": int(img_id.cpu()), "support_ann_ids": support_ann_ids},
            ignore_index=True)

        boxes = torch.stack(boxes)
        masks = torch.stack(masks)
        return boxes, masks

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
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        rois_box = self.pooler_box(features, targets)
        rois_mask = self.pooler_mask(features, targets)
        target_per_img = [len(p) for p in targets]
        labels = [p.get_field("labels").long() for p in targets]
        if self.training:
            rois_box_q, rois_box_s = self.prepare_roi_list(
                rois_box, target_per_img, labels)
            rois_mask_q, rois_mask_s = self.prepare_roi_list(
                rois_mask, target_per_img, labels)
        else:
            if self.is_extract_feature:
                self.save_features(rois_box, rois_mask, targets[0])
                return

            if self.is_use_feature:
                rois_box_s, rois_mask_s = self.load_from_df(targets[0], labels[0])
                rois_box_q = None
                rois_mask_q = None

        unique_labels = [torch.unique(l) for l in labels]
        meta_data = {"roi_box": (rois_box_q, rois_box_s),
                     "roi_mask": (rois_mask_q, rois_mask_s),
                     "unique_labels": unique_labels
                     }
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(
                features, proposals, targets, meta_data)
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
