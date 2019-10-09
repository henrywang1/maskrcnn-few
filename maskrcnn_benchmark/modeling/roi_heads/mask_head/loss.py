# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask


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
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    original_size = proposals.size
    assert matched_bboxes.size == proposals.size, "{}, {}".format(
        matched_bboxes, proposals
    )

    proposals = proposals.bbox.to(torch.device("cpu"))
    matched_bboxes = matched_bboxes.bbox.to(torch.device("cpu"))

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
        scaled_mask = cropped_mask.resize((M, M))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask.get_mask_tensor())
    if len(masks) == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.stack(masks, dim=0).to(device, dtype=torch.float32)

class MaskRCNNLossComputation(object):
    def __init__(self, proposal_matcher, discretization_size):
        """
        Arguments:
            proposal_matcher (Matcher)
            discretization_size (int)
        """
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        mask_h = mask_w = discretization_size
        self.aff_matrixes = [[-1, 0], [1, 0], [0, -1], [0, 1]]
        self.aff_maps = []
        self.ctr_maps = []
        for col in range(mask_h):
            for row in range(mask_w):
                for aff_matrix in self.aff_matrixes:
                    ctr_map = torch.zeros((mask_h, mask_w), device='cuda')
                    ctr_map[col][row] = 1
                    self.ctr_maps.append(ctr_map)

                    aff_map = torch.zeros((mask_h, mask_w), device='cuda')
                    x_offset, y_offset = aff_matrix[0], aff_matrix[1]
                    pix_x = row + x_offset
                    pix_y = col + y_offset
                    if pix_x < 0 or pix_y < 0 or pix_x >= mask_w or pix_y >= mask_h:
                        continue
                    aff_map[pix_y][pix_x] = 1
                    self.aff_maps.append(aff_map)

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
        self.aff_weights = [w.view(1, 1, 3, 3).to("cuda") for w in aff_weights]

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels"]) #, "masks"
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

    def prepare_targets_cr(self, proposals, targets):
        # Sample both negative and positive proposals
        # Only with per col/row labels (without mask)
        labels = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            device = proposals_per_image.bbox.device

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
            pos_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            # delete field "mask"
            new_matched_targets = matched_targets.copy_with_fields(
                ["matched_idxs", "labels"])
            # generate bbox corresponding proposals
            # TODO for now, whole mask (and thus all col/row label) of positive sample is 1, and of negative sample is 0
            #       because fg iou threshold is too high, when
            pos_masks_per_image = project_boxes_on_boxes(
                new_matched_targets[pos_inds], proposals_per_image[pos_inds], self.discretization_size
            )

            # generate label per image
            # initialize as zeros, and thus all labels of negative sample is zeros
            M = self.discretization_size
            labels_per_image = torch.zeros(
                (len(proposals_per_image.bbox), M + M), device=device)  # (n_proposal, 56)

            # generate label of positive sample
            pos_labels = []
            for mask in pos_masks_per_image:
                label_col = [torch.any(mask[col, :] > 0)
                             for col in range(mask.size(0))]
                label_row = [torch.any(mask[:, row] > 0)
                             for row in range(mask.size(1))]
                label = torch.stack(label_col + label_row)
                pos_labels.append(label)
            pos_labels = torch.stack(pos_labels).float()
            labels_per_image[pos_inds] = pos_labels

            # save
            labels.append(labels_per_image)
        return labels
      
    def __call__(self, proposals, mask_logits, targets):
        """
        Arguments:
            proposals (list[BoxList])
            mask_logits (Tensor)
            targets (list[BoxList])

        Return:
            mask_loss (Tensor): scalar tensor containing the loss
        """
        # labels, mask_targets = self.prepare_targets(proposals, targets)
        labels = cat([p.get_field("proto_labels") for p in proposals])
        labels = (labels > 0).long()
        pos_inds = torch.nonzero(labels > 0).squeeze(1)

        mil_score = mask_logits[:, 1]
        mil_score = torch.cat([mil_score.max(1)[0], mil_score.max(2)[0]], 1)

        # torch.mean (in binary_cross_entropy_with_logits) doesn't
        # accept empty tensors, so handle it separately
        if mil_score.numel() == 0:
            return mask_logits.sum() * 0

        labels_cr = self.prepare_targets_cr(proposals, targets)
        labels_cr = cat(labels_cr, dim=0)

        # if an RoI feature is adapted by different classes, the output mask should be zero
        # labels_cr[negtive_inds] = 0.

        mil_loss = F.binary_cross_entropy_with_logits(mil_score[pos_inds], labels_cr[pos_inds])
        mask_logits_n = mask_logits[:, 1:].sigmoid()
        aff_conv = lambda x, w: torch.nn.functional.conv2d(x, weight=w, padding=(1, 1))
        aff_maps = [aff_conv(mask_logits_n, w) for w in self.aff_weights]
        affinity_loss = [mask_logits_n*(aff_map**2) for aff_map in aff_maps]
        affinity_loss = torch.mean(torch.stack(affinity_loss))
        return 1.2 * mil_loss + 0.05* affinity_loss


def make_roi_mask_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = MaskRCNNLossComputation(
        matcher, cfg.MODEL.ROI_MASK_HEAD.RESOLUTION
    )

    return loss_evaluator
