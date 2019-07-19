import torch
import numpy as np
import logging
from abstract_classes import BatchDistractor, Pragmatics
from scipy.special import logsumexp


class NextExampleDistractor(BatchDistractor):
    """Use next example in batch as distractor"""
    def __init__(self, batch_size, logger=None):
        super().__init__(batch_size, logger)
        self._d_factor = 2

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src):
        new_src = []
        for batch in _chunks(src, self.orig_batch_size):
            for i, x in enumerate(batch):
                new_src.append(x)
                next_id = 0 if i == len(batch) - 1 else i + 1
                new_src.append(batch[next_id])
        return new_src, self.new_batch_size

class IdenticalDistractor(BatchDistractor):
    """Use the sample itself as distractor"""
    def __init__(self, batch_size, logger=None):
        super().__init__(batch_size, logger)
        self._d_factor = 2

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src):
        new_src = []
        for x in src:
            new_src.append(x)
            new_src.append(x)
        return new_src, self.new_batch_size

class NextNDistractor(BatchDistractor):
    """Use next N examples in batch as distractor"""
    def __init__(self, batch_size, N, logger=None):
        assert batch_size >= N+1, 'Invalid N!'
        super().__init__(batch_size, logger)
        self._d_factor = N+1

    @property
    def d_factor(self):
        return self._d_factor

    def generate(self, src):
        new_src = []
        for batch in _chunks(src, self.orig_batch_size):
            if len(batch) < self.d_factor:
                self._log('Dropping this batch that is too short', logging.WARNING)
                continue
            for i, x in enumerate(batch):
                ids = [j if j < len(batch) else j - len(batch) \
                    for j in range(i, i + self.d_factor)]
                new_src += [batch[j] for j in ids]

        return new_src, self.new_batch_size

class BasicPragmatics(Pragmatics):
    def __init__(self, alpha=1, logger=None):
        super().__init__(logger)
        self.alpha = alpha

    def l1(self, log_probs):
        """Pragmatic listener bases on top of s0"""
        # TODO: consider efficiency of not transposing?
        normalized = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
        normalized[torch.isnan(normalized)] = -np.log(normalized.shape[1])
        # print('has inf?', normalized[torch.isinf(normalized)].shape)

        return normalized

    def s1(self, s0_log_probs, l1_log_probs):
        """Pragmatic speaker"""
        adjusted = self.alpha * l1_log_probs
        isnan_mask = torch.isnan(adjusted)
        adjusted[isnan_mask] = float('-inf')

        log_probs = s0_log_probs + adjusted

        lse = torch.logsumexp(log_probs, dim=2, keepdim=True)
        normalized = log_probs - lse
        return normalized

    def inference(self, s0_log_probs):
        # issue is now fixed, problem was because s0_log_probs was not normalized
        s0_log_probs = s0_log_probs.type(torch.float)
        l1_log_probs = self.l1(s0_log_probs)
        res = self.s1(s0_log_probs, l1_log_probs).type(torch.float)
        return res


def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(l), n):
        yield l[i:i + n]

def scramble2tgt(idxs, d_factor):
    """Given scrambling indices, get indices of target examples"""
    # e.g. input idxs=[1,2,0,3], d_factor=2 -> out [2, 1],
    # elements 2 and 1 in list [1,2,0,3] are the indices of two targets
    # respectively, in correct order
    if isinstance(idxs, torch.Tensor):
        idxs_len = idxs.shape[0]
    elif isinstance(idxs, list):
        idxs_len = len(idxs)
    scrambled = idx_remap(idxs)
    return scrambled[torch.arange(0, idxs_len, d_factor)]

def idx_remap(idxs):
    # e.g. input idxs=[1,2,0,3] -> output [2,0,1,3]
    inds, perm = torch.sort(idxs)
    return perm
