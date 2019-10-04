# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch


def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_encode_label(labels, unique_labels):
    """
    Encode each label according to its position in the set of unique_labels
    """
    device = labels.device
    if unique_labels.numel() == 1:
        mask = (labels == unique_labels[0]).long()
    else:
        mask = (labels == unique_labels.view(-1, 1)).long() * \
            (torch.arange(unique_labels.numel()) + 1).view(-1, 1).to(device)
        mask = mask.sum(0)

    return mask
