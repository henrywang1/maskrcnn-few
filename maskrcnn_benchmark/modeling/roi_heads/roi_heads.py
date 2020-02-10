# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head


def mask_avg_pool(fts, mask):
    """
    Extract foreground and background features via masked average pooling

    Args:
        fts: input features, expect shape: B x C x H' x W'
        mask: binary mask, expect shape: B x H x W
    """
    fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
    # import pdb; pdb.set_trace()
    fts = fts * mask.unsqueeze(1).expand(mask.shape[0], 256, 28, 28)
    masked_fts = torch.sum(fts, dim=(
        2, 3)) / (mask.sum(dim=(1, 2)) + 1e-5).unsqueeze(1).expand(mask.shape[0], 256)
    return masked_fts


def calculate_prototype(features, labels):
    if not features.numel():
        return features

    unique_labels = torch.unique(labels)
    if unique_labels.numel() == 1:
        return features.mean(0, keepdim=True)

    avg_features = [features[labels == l].mean(0) for l in unique_labels]
    avg_features = torch.stack(avg_features)
    return avg_features


def prepare_roi_list(roi, targets_per_img, labels):
    roi_q, roi_s = roi.split(targets_per_img)
    roi_q = calculate_prototype(roi_q, labels[0])
    roi_s = calculate_prototype(roi_s, labels[1])
    return roi_q, roi_s


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, meta_data=None):
        losses = {}
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets, meta_data)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_mask = self.mask(mask_features, detections, targets, meta_data)
            losses.update(loss_mask)

            pos_proposals = meta_data["pos_proposals"]

            pred_box_feature = self.box.feature_extractor.pooler(
                features, pos_proposals)
            pred_labels = [p.get_field("labels") for p in pos_proposals]
            pred_masks = meta_data["pred_mask"]
            pred_masks = (pred_masks == 0).long()
            pos_proposals_length = [len(p) for p in pos_proposals]

            pred_box_feature = mask_avg_pool(pred_box_feature, pred_masks)
            pred_roi_q, pred_roi_s = prepare_roi_list(
                pred_box_feature, pos_proposals_length, pred_labels)
            pred_roi_q = pred_roi_q.unsqueeze(-1).unsqueeze(-1)
            pred_roi_s = pred_roi_s.unsqueeze(-1).unsqueeze(-1)
            meta_data["roi_box"] = (pred_roi_q, pred_roi_s)
            meta_data["unique_labels"] = (torch.unique(
                pred_labels[0]), torch.unique(pred_labels[1]))
            _, _, loss_pred_box = self.box(
                features, pos_proposals, targets, meta_data)

            loss_pred_box["loss_classifier_pred"] = loss_pred_box.pop(
                "loss_classifier") * 0.01
            loss_pred_box["loss_box_reg_pred"] = loss_pred_box.pop(
                "loss_box_reg") * 0.01
            losses.update(loss_pred_box)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
            losses.update(loss_keypoint)
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
