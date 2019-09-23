# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        match_between_matrix = self.match_between_targets(target)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        match_between_idx = (match_between_matrix > 0.9).nonzero()
        high_threshold = self.proposal_matcher.high_threshold
        other_matched_labels = []
        other_matched_indx = []
        another_matched_idxs = matched_idxs.clone().fill_(-1)
        target_labels = target.get_field("labels")
        if match_between_idx.numel():
            for k, v in match_between_idx:
                # IoU with another class is above threshold
                # and matched with ground truth
                idx_k = ((match_quality_matrix[v] >= high_threshold) & (matched_idxs == k)).int()
                idx_v = ((match_quality_matrix[k] >= high_threshold) & (matched_idxs == v)).int()

                another_matched_idxs[idx_k.nonzero()] = v
                another_matched_idxs[idx_v.nonzero()] = k
  
                other_matched_indx.append(idx_k * v + idx_v * k)
                labels = idx_k * target_labels[v] + idx_v * target_labels[k]
                other_matched_labels.append(labels.to(dtype=torch.int64))
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]

        another_target = target.copy_with_fields(["labels"])
        matched_targets_2 = another_target[another_matched_idxs.clamp(min=0)]

        matched_targets.add_field("matched_idxs", matched_idxs)
        matched_targets_2.add_field("matched_idxs", another_matched_idxs)
        
        return matched_targets, other_matched_labels, matched_targets_2

    def match_between_targets(self, target):
        match_quality_matrix = boxlist_iou(target, target)
        num = len(target)
        mask = torch.ones(num, num).masked_fill_(torch.eye(num).bool(), 0)
        match_quality_matrix = match_quality_matrix * mask.cuda()
        match_quality_matrix = match_quality_matrix.triu()
        return match_quality_matrix

    def get_one_hot(self, labels):
        one_hot = torch.zeros(labels.size(0), 1231).to(labels.device)
        return one_hot.scatter_(dim=1, index=labels.unsqueeze(1), value=1.)

    def prepare_targets(self, proposals, targets):
        labels = []
        one_hot_labels = []
        regression_targets = []
        regression_targets_2 = []
        labels_2 = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets, othere_matched_labels, matched_targets_2 = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            one_hot_per_image = self.get_one_hot(labels_per_image)

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler
            matched_idxs_2 = matched_targets_2.get_field("matched_idxs")
            labels_per_image_2 = matched_targets_2.get_field("labels")
            labels_per_image_2 = labels_per_image_2.to(dtype=torch.int64)
            bg_inds = matched_idxs_2 == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image_2[bg_inds] = 0
            # code for debug and print labels
            # if len(othere_matched_labels):
            #     idx = othere_matched_labels[0] > 0
            #     print([all_cat_name[int(i)] for i in labels_per_image[idx]])
            #     print([all_cat_name[int(i)] for i in othere_matched_labels[0][idx]])

            #     if len(othere_matched_labels) > 1:
            #         idx2 = othere_matched_labels[1] > 0
            #         print([all_cat_name[int(i)] for i in labels_per_image[idx2]])
            #         print([all_cat_name[int(i)] for i in othere_matched_labels[1][idx2]])
            if len(othere_matched_labels):
                for other_matches in othere_matched_labels:
                    one_hot = self.get_one_hot(other_matches)
                    one_hot_per_image = one_hot_per_image + one_hot
            one_hot_labels.append(one_hot_per_image)

            regression_targets_per_image_2 = self.box_coder.encode(
            matched_targets_2.bbox, proposals_per_image.bbox)
            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )
            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)
            labels_2.append(labels_per_image_2)
            regression_targets_2.append(regression_targets_per_image_2)

        return labels, regression_targets, labels_2, regression_targets_2, one_hot_labels

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets, labels_2, regression_targets_2, one_hot_labels = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image,\
                labels_per_image_2, regression_targets_per_image_2,\
                proposals_per_image, one_hot_per_image \
                in zip(labels, regression_targets, labels_2,\
                       regression_targets_2, proposals, one_hot_labels):

            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )
            proposals_per_image.add_field("labels_2", labels_per_image_2)
            proposals_per_image.add_field(
                "regression_targets_2", regression_targets_per_image_2
            )
            # if labels_per_image_2.sum() >0:
            #     print(labels_per_image.nonzero())
            #     print(labels_per_image_2.nonzero())
            #     print(labels_per_image.unique())
            #     print(labels_per_image_2.unique())
            #     #print(one_hot_per_image)
            #     import pdb; pdb.set_trace()
            proposals_per_image.add_field("one_hot_labels", one_hot_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """

        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        labels_2 = cat([proposal.get_field("labels_2") for proposal in proposals], dim=0)
        regression_targets_2 = cat(
            [proposal.get_field("regression_targets_2") for proposal in proposals], dim=0
        )

        one_hot_labels = cat([proposal.get_field("one_hot_labels") for proposal in proposals], dim=0)

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        classification_loss = F.cross_entropy(class_logits, labels)
        classification_loss_multi = F.binary_cross_entropy_with_logits(
            class_logits[sampled_pos_inds_subset], one_hot_labels[sampled_pos_inds_subset])
        classification_loss = classification_loss + classification_loss_multi
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing  
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss = box_loss / labels.numel()
        sampled_pos_inds_subset = torch.nonzero(labels_2 > 0).squeeze(1)
        labels_pos = labels_2[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor(
                [0, 1, 2, 3], device=device)
        box_loss_2 = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets_2[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        box_loss_2 = box_loss_2 / labels_2.numel()
        box_loss = box_loss + box_loss_2

        return classification_loss, box_loss


def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher,
        fg_bg_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )

    return loss_evaluator
