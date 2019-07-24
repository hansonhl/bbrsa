import torch

from bbrsa.abstract_classes import BBRSAABC
from bbrsa.utils import idx_remap

class Evaluator(BBRSAABC):
    def __init__(self, s0, add_tags=True, logger=None):
        super().__init__(logger)
        self.s0 = s0
        self.add_tags = add_tags

    def evaluate(self, srcs, summary, truncate=None):
        """ Given target and distractors and a summary, get L1 probability for the target

        Currently only supports one target and one target+distractor set.

        Args:
            srcs: a python list of tokenized strings
            tgt: a string that is the summary
            truncate: (default None) whether to truncate the srcs
        """

        # some things to consider: summary may contain tokens that are not in
        # model vocab or src vocab
        num_distractors = len(srcs)
        summary = summary.strip()
        if self.add_tags:
            summary = '<t> ' + summary + ' </t>'
        tgts = [summary] * num_distractors

        s0 = self.s0
        batch_size = num_distractors if num_distractors > s0.default_batch_size \
            else s0.default_batch_size

        s0.init_batch_iterator(src=srcs, tgt=tgts, truncate=truncate,
            batch_size=batch_size)

        with torch.no_grad():

            for batch in s0.data_iter:
                s0.encode(batch)
                decoder_inputs = batch.tgt
                log_probs, _ = s0.decode(
                    input=decoder_inputs,
                    batch=batch,
                    step=None,
                    beam_batch_offset=None)

                reorder_idxs = idx_remap(batch.indices)
                reordered_probs = log_probs.index_select(1, reorder_idxs)
                range_idxs = torch.arange(reordered_probs.shape[0])
                word_idxs = batch.tgt[:, 0,0]
                word_probs = reordered_probs[range_idxs, :, word_idxs]
                sent_probs = word_probs.sum(dim=0)
                sent_probs = sent_probs - torch.logsumexp(sent_probs, dim=0, keepdim=True)
                sent_probs = torch.exp(sent_probs)

                return sent_probs
