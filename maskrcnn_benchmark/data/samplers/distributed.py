# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Code is copy-pasted exactly as in torch.utils.data.distributed.
# FIXME remove this once c10d fixes the bug it has
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from maskrcnn_benchmark.utils.comm import is_main_process

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.over_sampling = False
        self.uniform_sampling = False
        if self.dataset.is_train:
            self.weights = [v for k, v in dataset.img_repeat_factor.items()]
            self.weights_uniform = [v for k, v in dataset.img_repeat_factor_u.items()]
            # self.over_sampling = True

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            if self.over_sampling:
                indices = torch.multinomial(torch.tensor(self.weights), len(self.dataset), True).tolist()
            elif self.uniform_sampling:
                indices = torch.multinomial(torch.tensor(self.weights_uniform), len(self.dataset), True).tolist()
            else:
                indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_over_sampling(self):
        self.over_sampling = True

    def set_uniform_sampling(self):
        self.uniform_sampling = True
        self.over_sampling = False
