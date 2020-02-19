# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
import math

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
    def __init__(self, proposal_matcher, discretization_size, use_mil_loss, use_aff, use_box_mask):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size

        # self.box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        # self.use_mil_loss = use_mil_loss
        # self.use_aff = use_aff
        # if use_box_mask:
        #     assert not use_mil_loss
        # self.use_box_mask = use_box_mask

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

    def __call__(self, proposals, mask_logits, targets):
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
        _, mask_targets = self.prepare_targets(proposals, targets)
        mask_targets = cat(mask_targets, dim=0)
        labels_pos = labels[pos_inds]
        if mask_targets.numel() == 0:
            return mask_logits.sum() * 0

        mask_loss = F.binary_cross_entropy_with_logits(
            mask_logits[pos_inds, labels_pos], mask_targets[pos_inds]
        )

        return mask_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher,
        cfg.MODEL.ROI_MASK_HEAD.RESOLUTION,
        cfg.MODEL.ROI_MASK_HEAD.USE_MIL_LOSS,
        cfg.MODEL.ROI_MASK_HEAD.USE_MIL_USE_AFF,
        cfg.MODEL.ROI_MASK_HEAD.USE_BOX_MASK
    )

    return loss_evaluator
