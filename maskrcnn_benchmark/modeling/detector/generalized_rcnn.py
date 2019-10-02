# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


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
        self.pooler_mask = self.roi_heads.mask.feature_extractor.pooler

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
            rois_box_q, rois_box_s = self.prepare_roi_list(rois_box, target_per_img, labels)
            rois_mask_q, rois_mask_s = self.prepare_roi_list(rois_mask, target_per_img, labels)
        else:
            pass
        unique_labels = [torch.unique(l, sorted=True) for l in labels]
        meta_data = {"roi_box": (rois_box_q, rois_box_s),
                     "roi_mask": (rois_mask_q, rois_mask_s),
                     "unique_labels": unique_labels
                     }
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets, meta_data)
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
