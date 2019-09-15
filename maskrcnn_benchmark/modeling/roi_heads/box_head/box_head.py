# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.make_layers import make_fc

class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
        self.label_set = None
        self.use_transfer = cfg.MODEL.ROI_BOX_HEAD.USE_TRANSFER
        self.transfer_fc_hidden = make_fc(776*5 + 1024, 1024)
        self.transfer_fc_cls = make_fc(1024, 915)
        self.transfer_fc_box = make_fc(1024, 915*4)
        # 776 freq, common
        # 915 common, rare
    def set_label_set(self, label_set):
        # self.label_set = label_set
        self.source_labels = label_set["cat_f"] + label_set["cat_c"]
        self.target_labels = label_set["cat_c"] + label_set["cat_r"]

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        if self.use_transfer:
            cls_source = class_logits[:, self.source_labels]
            map_inds = 4 * torch.tensor(self.source_labels)[:, None] + torch.tensor([0, 1, 2, 3])
            map_inds = map_inds.to(box_regression.device).view(-1)
            box_soucre = box_regression[:, map_inds]
            x = self.transfer_fc_hidden(torch.cat([x, cls_source, box_soucre], 1))
            class_logits[:, self.target_labels] = self.transfer_fc_cls(x)
            map_inds_target = 4 * torch.tensor(self.target_labels)[:, None] + torch.tensor([0, 1, 2, 3])
            map_inds_target = map_inds_target.to(box_regression.device).view(-1)
            box_regression[:, map_inds_target] = self.transfer_fc_box(x)

        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
