import torch

class NextExampleDistractor(object):
    """Use next example in batch as distractor"""
    def __init__(self, batch_size):
        self.orig_batch_size = batch_size
        self.d_factor = 2
        self.new_batch_size = self.d_factor * batch_size

    def generate(self, src):
        new_src = []
        for batch in chunks(src, self.orig_batch_size):
            for i, x in enumerate(batch):
                new_src.append(x)
                next_id = 0 if i == len(batch) - 1 else i + 1
                new_src.append(batch[next_id])
        return new_src, self.new_batch_size

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(l), n):
        yield l[i:i + n]
