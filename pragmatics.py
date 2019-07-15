import torch

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
    def __init__(self, batch_size, d_factor, beam_size):
        self.batch_size = batch_size
        self.d_factor = d_factor
        self.beam_size = beam_size

    def l1(self, log_probs):
        """Pragmatics listener bases on top of s0"""
        transposed = log_probs.transpose(1, 2) #[B*b, V, d]
        return transposed - torch.logsumexp(transposed, dim=2, keepdim=True)

        # need to deal with inf



def _chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(l), n):
        yield l[i:i + n]

def reorderidx2tgt(idxs, d_factor):
    """Given reordering indices, get indices of target examples"""
    # e.g. input idxs=[1,2,0,3], d_factor=2 -> out [2, 1],
    # elements 2 and 1 in list [1,2,0,3] are the indices of two targets
    # respectively, in correct order

    if isinstance(idxs, torch.Tensor):
        idxs = idxs.tolist()

    pairs = list(zip(range(len(idxs)), idxs))
    f = list(filter(lambda p: p[1] % d_factor == 0, pairs))
    return torch.tensor([p[0] for p in sorted(f, key=lambda q: q[1])])
