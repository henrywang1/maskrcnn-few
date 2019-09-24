# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator
from maskrcnn_benchmark.modeling.make_layers import make_fc
from torch.nn import functional as F

class MLP(nn.Module): # MLP is used to transfer RoI features to Sentence embedding
    def __init__(self, input_size, output_size, representation_size, n_blk=3):
        super(MLP, self).__init__()
        self.models = []
        self.models += [make_fc(input_size, representation_size), nn.ReLU()]
        for i in range(n_blk - 2):
            self.models += [make_fc(representation_size, representation_size), nn.ReLU()]
        self.models += [make_fc(representation_size, output_size), nn.ReLU()]
        self.model = nn.Sequential(*self.models)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

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
        self.mlp = MLP(1024, 768, 1024)
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
        sentence_embedding = self.mlp(x)
        x = torch.cat([x, sentence_embedding], 1)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        if not self.training:
            result = self.post_processor((class_logits, box_regression), proposals)
            return x, result, {}

        loss_classifier, loss_box_reg, loss_sentence = self.loss_evaluator(
            [class_logits], [box_regression], [sentence_embedding]
        )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_sentence=loss_sentence)
        )


def build_roi_box_head(cfg, in_channels):
    """
    Constructs a new box head.
    By default, uses ROIBoxHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIBoxHead(cfg, in_channels)
