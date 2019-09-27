# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter
        # self.is_break = False

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                # if self.is_break:
                #    self.is_break = False
                #    break
                yield batch

    def set_over_sampling(self):
        self.batch_sampler.sampler.set_over_sampling()
    
    def set_uniform_sampling(self):
        self.batch_sampler.sampler.set_uniform_sampling()

    def __len__(self):
        return self.num_iterations
