# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn.functional import one_hot

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.modeling.make_layers import make_fc
from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator


def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim=256, activation='reul'):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        assert x1.size() == x2.size()
        m_batchsize, C, width, height = x1.size()
        proj_query = self.query_conv(x1).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X(N)
        proj_key = self.key_conv(x2).view(
            m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x2).view(
            m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x1
        return out, attention

class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.use_corr = cfg.MODEL.ROI_MASK_HEAD.USE_CORR and cfg.MODEL.ROI_MASK_HEAD.USE_MIL_LOSS
        self.use_mlp = cfg.MODEL.ROI_MASK_HEAD.USE_MLP and cfg.MODEL.ROI_MASK_HEAD.USE_MIL_LOSS
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels, extract_type="avg")
        self.predictor = make_roi_mask_predictor(cfg, self.feature_extractor.out_channels)
        self.predictor_2 = make_roi_mask_predictor(cfg, self.feature_extractor.out_channels)
        if self.use_corr:
            self.feature_extractor_corr = make_roi_mask_feature_extractor(cfg, in_channels, extract_type="corr")
            self.predictor_corr = make_roi_mask_predictor(cfg, self.feature_extractor.out_channels)
        if self.use_mlp:
            self.feature_extractor_mlp = make_roi_mask_feature_extractor(cfg, in_channels, extract_type="mlp")
            self.predictor_mlp = make_fc(self.feature_extractor.out_channels*14*14, 2*28*28)

        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)
        self.use_box_mask = cfg.TEST.USE_BOX_MASK
        self.attn = Self_Attn( 256, 'relu')
    # def forward_support(self, roi_s):
    #     roi_s = self.feature_extractor.forward_ext(roi_s)
    #     mask_logits = self.predictor(roi_s)
    #     disc_maps = torch.stack(
    #         [one_hot(x[1].argmax(0), 28) + one_hot(x[1].argmax(1), 28) for x in mask_logits])
    #     return disc_maps

    def forward(self, features, proposals, targets=None, meta_data=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals, meta_data)

        if self.use_box_mask:
            mask_logits = torch.ones(x.shape[0], 2, 28, 28)
        else:
            mask_logits = self.predictor(x)

        if not self.training:
            result = self.post_processor(mask_logits, proposals)
            return x, result, {}

        loss_mask = self.loss_evaluator(proposals, mask_logits, targets)

        return x, all_proposals, dict(loss_mask=loss_mask)


def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
