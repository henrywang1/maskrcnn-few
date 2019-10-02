# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from ..box_head.roi_box_feature_extractors import ResNet50Conv5ROIFeatureExtractor
from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import make_conv3x3


registry.ROI_MASK_FEATURE_EXTRACTORS.register(
    "ResNet50Conv5ROIFeatureExtractor", ResNet50Conv5ROIFeatureExtractor
)


@registry.ROI_MASK_FEATURE_EXTRACTORS.register("MaskRCNNFPNFeatureExtractor")
class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_MASK_HEAD.USE_GN
        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS
        dilation = cfg.MODEL.ROI_MASK_HEAD.DILATION

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = make_conv3x3(
                next_feature, layer_features,
                dilation=dilation, stride=1, use_gn=use_gn
            )
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x, proposals, meta_data):
        x = self.pooler(x, proposals)
        roi_q, roi_s = meta_data["roi_mask"]
        roi_q = F.adaptive_avg_pool2d(roi_q, 1) if roi_q.numel() else roi_q
        roi_s = F.adaptive_avg_pool2d(roi_s, 1) if roi_s.numel() else roi_s
        proto_labels = [p.get_field("proto_labels") for p in proposals]
        
        if x.numel():
            proto_labels_q = proto_labels[0] - 1
            roi_s = roi_s[proto_labels_q]
            if self.training:
                proto_labels_s = proto_labels[1] - 1
                roi_q = roi_q[proto_labels_s]
                roi = torch.cat([roi_s, roi_q])
                x = x * roi
            else:
                x = x*roi_s

        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x


def make_roi_mask_feature_extractor(cfg, in_channels):
    func = registry.ROI_MASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
