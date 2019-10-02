# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder


class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box_coder=None,
        cls_agnostic_bbox_reg=False,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box_coder (BoxCoder)
        """
        super(PostProcessor, self).__init__()
        self.score_thresh = score_thresh
        self.nms = nms
        self.detections_per_img = detections_per_img
        if box_coder is None:
            box_coder = BoxCoder(weights=(10., 10., 5., 5.))
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward(self, x, boxes, unique_labels):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logit, box_regression = x
        class_prob = F.softmax(class_logit, 1)
        class_prob = class_prob[:, 1]
        box_regression = box_regression[:, 4:8]
        num_classes = len(unique_labels)
        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [num_classes*len(box) for box in boxes]
        concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
        concat_boxes = concat_boxes.repeat(num_classes, 1)
        proposals = self.box_coder.decode(box_regression.view(sum(boxes_per_image), -1), concat_boxes)
        
        proposals = proposals.split(boxes_per_image, dim=0)
        class_prob = (class_prob, )
        results = []
        for boxes_per_img, image_shape, prob in zip(proposals, image_shapes, class_prob):
            boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            boxlist = self.filter_results(boxlist, num_classes, unique_labels)
            results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        # boxes = boxes.reshape(-1, 4)
        # scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        return boxlist

    def filter_results(self, boxlist, num_classes, unique_labels):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox#.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores")#.reshape(-1, num_classes)
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        # inds_all = scores > self.score_thresh
        k = scores.numel() // len(unique_labels)
        for j in range(0, num_classes):
            score_thresh = self.score_thresh
            scores_j = scores[j * k:(j + 1) * k]
            boxes_j = boxes[j * k:(j + 1) * k]
            inds = scores_j > (score_thresh)
            scores_j = scores_j[inds]
            boxes_j = boxes_j[inds]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            if scores_j.numel():
                box_len = len(boxlist_for_class)
                labels = unique_labels[j].repeat(box_len)
                proto_idx = (torch.zeros(box_len) + (j + 1)).long().cuda()
            else:  # assign empty tensor
                labels = scores_j.long()
                proto_idx = scores_j.long()

            boxlist_for_class.add_field("labels", labels)
            boxlist_for_class.add_field("proto_labels", proto_idx)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, self.nms, score_field="scores"
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result


def make_roi_box_post_processor(cfg):
    use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG
    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    bbox_aug_enabled = cfg.TEST.BBOX_AUG.ENABLED

    postprocessor = PostProcessor(
        score_thresh,
        nms_thresh,
        detections_per_img,
        box_coder,
        cls_agnostic_bbox_reg,
        bbox_aug_enabled
    )
    return postprocessor
