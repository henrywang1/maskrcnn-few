# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.modeling.box_coder import BoxCoder


def project_masks_on_boxes(segmentation_masks, proposals, discretization_size):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
    """
    masks = []
    M = discretization_size
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(
        segmentation_masks, proposals
    )

    # FIXME: CPU computation bottleneck, this should be parallelized
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation.
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.get_mask_tensor()
        masks.append(mask)
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)

def project_boxes_on_boxes(matched_bboxes, proposals, discretization_size):
    masks = []
    M = discretization_size
    original_size = proposals.size

    proposals = proposals.bbox.to(torch.device("cpu"))
    matched_bboxes = matched_bboxes.to(torch.device("cpu"))
    # Generate segmentation masks based on matched_bboxes
    polygons = []
    for matched_bbox in matched_bboxes:
        x1, y1, x2, y2 = matched_bbox[0], matched_bbox[1], matched_bbox[2], matched_bbox[3]
        p = [[x1, y1, x1, y2, x2, y2, x2, y1]]
        polygons.append(p)
    segmentation_masks = SegmentationMask(polygons, original_size)

    for segmentation_mask, proposal in zip(segmentation_masks, proposals):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        size = cropped_mask.size
        # LVIS dataset could have tiny boxes
        if size[0] > 1e-2 and size[1] > 1e-2:
            scaled_mask = cropped_mask.resize((M, M))
            mask = scaled_mask.convert(mode="mask")
            mask_tensor = mask.get_mask_tensor()
        else:
            logger = logging.getLogger("maskrcnn_benchmark.trainer")
            logger.warning("%s is too small", cropped_mask)
            mask_tensor = torch.zeros((28, 28), dtype=torch.uint8)
        masks.append(mask_tensor)

    return torch.stack(masks, dim=0).to(dtype=torch.float32)

class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size, use_mil_loss):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        center_weight = torch.zeros((3, 3))
        center_weight[1][1] = 1.
        aff_weights = []
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue
                weight = torch.zeros((3, 3))
                weight[i][j] = 1.
                aff_weights.append(center_weight - weight)
        aff_weights = [w.view(1, 1, 3, 3).to("cuda") for w in aff_weights]
        self.aff_weights = torch.cat(aff_weights, 0)
        self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.use_mil_loss = use_mil_loss

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        masks = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            segmentation_masks = matched_targets.get_field("masks")
            segmentation_masks = segmentation_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            masks_per_image = project_masks_on_boxes(
                segmentation_masks, positive_proposals, self.discretization_size
            )

            labels.append(labels_per_image)
            masks.append(masks_per_image)

        return labels, masks

    def prepare_targets_cr(self, proposals):
        # Sample both negative and positive proposals
        # Only with per col/row labels (without mask)
        labels = []
        for proposals_per_image in proposals:

            matched_bbox = self.box_coder.decode(proposals_per_image.get_field(
                "regression_targets"), proposals_per_image.bbox)

            M = self.discretization_size
            pos_masks_per_image = torch.ones(len(proposals_per_image), M, M, dtype=torch.float32)
            not_matched_idx = (proposals_per_image.get_field(
                "regression_targets") != 0).any(1)
            # if a box fully matched the proposal, we set all values to 1
            # otherwise, we project the box on proposal
            if not_matched_idx.any():
                pos_masks_per_image[not_matched_idx] = project_boxes_on_boxes(
                    matched_bbox[not_matched_idx], proposals_per_image[not_matched_idx], M)

            pos_masks_per_image = pos_masks_per_image.cuda()
            pos_labels = torch.cat(
                [pos_masks_per_image.sum(2), pos_masks_per_image.sum(1)], 1)
            pos_labels = (pos_labels > 0).float()

            labels.append(pos_labels)

        return labels

    def __call__(self, proposals, mask_logits, targets, mask_logits_corr=None):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        labels = cat([p.get_field("proto_labels") for p in proposals])
        labels = (labels > 0).long()
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        if not self.use_mil_loss:
            _, mask_targets = self.prepare_targets(proposals, targets)
            mask_targets = cat(mask_targets, dim=0)
            labels_pos = labels[pos_inds]
            if mask_targets.numel() == 0:
                return mask_logits.sum() * 0
            mask_loss = F.binary_cross_entropy_with_logits(
                mask_logits[pos_inds, labels_pos], mask_targets[pos_inds]
            )
            return mask_loss

        mil_score = mask_logits[:, 1]
        mil_score = torch.cat([mil_score.max(2)[0], mil_score.max(1)[0]], 1)
        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mil_score.numel() == 0:
            return mask_logits.sum() * 0

        labels_cr = self.prepare_targets_cr(proposals)
        labels_cr = cat(labels_cr, dim=0)
        mil_loss = F.binary_cross_entropy_with_logits(
            mil_score[pos_inds], labels_cr[pos_inds])

        mask_logits_n = mask_logits[:, 1:].sigmoid()
        aff_maps = F.conv2d(mask_logits_n, self.aff_weights, padding=(1, 1))
        affinity_loss = mask_logits_n * (aff_maps**2)
        affinity_loss = torch.mean(affinity_loss)

        if mask_logits_corr is not None:
            mil_score_corr = mask_logits_corr[:, 1]
            mil_score_corr = torch.cat(
                [mil_score_corr.max(2)[0], mil_score_corr.max(1)[0]], 1)
            mil_loss_corr = F.binary_cross_entropy_with_logits(
                mil_score_corr[pos_inds], labels_cr[pos_inds])
            mil_loss = mil_loss + mil_loss_corr

            mask_logits_n_corr = mask_logits_corr[:, 1:].sigmoid()
            aff_maps = F.conv2d(mask_logits_n_corr, self.aff_weights, padding=(1, 1))
            affinity_loss_corr = mask_logits_n_corr * (aff_maps**2)
            affinity_loss_corr = torch.mean(affinity_loss_corr)
            affinity_loss = affinity_loss + affinity_loss_corr

        return 1.2 * mil_loss + 0.05* affinity_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher,
        cfg.MODEL.ROI_MASK_HEAD.RESOLUTION,
        cfg.MODEL.ROI_MASK_HEAD.USE_MIL_LOSS
    )

    return loss_evaluator
