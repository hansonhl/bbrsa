import torch
import numpy as np

class NextExampleDistractor(object):
    """Use next example in batch as distractor"""
    def __init__(self, batch_size):
        self.orig_batch_size = batch_size
        self.d_factor = 2
        self.new_batch_size = self.d_factor * batch_size

    def generate(self, src):
        new_src = []
        for batch in _chunks(src, self.orig_batch_size):
            for i, x in enumerate(batch):
                new_src.append(x)
                next_id = 0 if i == len(batch) - 1 else i + 1
                new_src.append(batch[next_id])
        return new_src, self.new_batch_size

class BasicPragmatics(object):
    def __init__(self, alpha=1):
        self.alpha = alpha

    def l1(self, log_probs):
        """Pragmatics listener bases on top of s0"""
        # TODO: consider efficiency of not transposing?
        normalized = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
        normalized[torch.isnan(normalized)] = -np.log(normalized.shape[1])
        return normalized

    def s1(self, s0_log_probs, l1_log_probs):
        log_probs = s0_log_probs + self.alpha * l1_log_probs
        normalized = log_probs - torch.logsumexp(log_probs, dim=2, keepdim=True)
        return normalized

    def inference(self, s0_log_probs):
        l1_log_probs = self.l1(s0_log_probs)
        return self.s1(s0_log_probs, l1_log_probs)




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
    if isinstance(idxs, torch.Tensor):
        idxs_len = idxs.shape[0]
    elif isinstance(idxs, list):
        idxs_len = len(idxs)

    scrambled = torch.LongTensor(idxs_len)
    for i, x in enumerate(idxs):
        scrambled[x] = i
    return scrambled
