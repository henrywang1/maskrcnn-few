# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.make_layers import make_fc
from torch.nn import functional as F

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

        import pickle
        with open("relation.pickle", "rb") as f:
            relation = pickle.load(f)
        self.label_targets = relation["label_targets"]
        self.label_sources = relation["label_sources"]

    def cosine_distance(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        return F.cosine_similarity(a, b, 2)

    def get_map_inds(self, label):
        map_inds = 4 * torch.tensor(label)[:, None] + torch.tensor([0, 1, 2, 3])
        map_inds = map_inds.cuda().view(-1)
        return map_inds

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

        class_logits_stacked = class_logits.clone()
        box_regression_stacked = box_regression.clone()

        for label_targets, label_sources in zip(self.label_targets, self.label_sources):
            class_logits_prev = class_logits_stacked.clone()
            source = class_logits_prev[:, label_sources].clone()
            target = class_logits[:, label_targets].clone()
            class_logits_stacked[:, label_targets] = source + target

            box_regression_prev = box_regression_stacked.clone()
            map_ids_src = self.get_map_inds(label_targets)
            map_ids_tgt = self.get_map_inds(label_sources)
            source_box = box_regression_prev[:, map_ids_src].clone()
            target_box = box_regression[:, map_ids_tgt].clone()
            box_regression_stacked[:, map_ids_tgt] = source_box + target_box
            
        class_logits = class_logits_stacked
        box_regression = box_regression_stacked

        if not self.training:
            class_logits = class_logits[:, :1231]
            box_regression = box_regression[:, :1231*4]
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg = self.loss_evaluator(
            [class_logits], [box_regression]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier,loss_box_reg=loss_box_reg)
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
