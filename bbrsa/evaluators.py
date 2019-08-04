import torch

from tqdm import tqdm

from bbrsa.abstract_classes import BBRSAABC
from bbrsa.utils import idx_remap
from bbrsa.distractors import AsIsDistractor


class Evaluator(BBRSAABC):
    def __init__(self, eval_s0, add_tags=False, logger=None):
        super().__init__(logger)
        self.eval_s0 = eval_s0
        self.add_tags = add_tags

    def get_l1_probs(self, srcs, summary, truncate=None):
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

        s0 = self.eval_s0
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
                print('log_probs.shape', log_probs.shape)

                reorder_idxs = idx_remap(batch.indices)
                reordered_probs = log_probs.index_select(1, reorder_idxs)
                range_idxs = torch.arange(reordered_probs.shape[0])
                word_idxs = batch.tgt[:, 0,0]
                word_probs = reordered_probs[range_idxs, :, word_idxs]
                sent_probs = word_probs.sum(dim=0)
                sent_probs = sent_probs - torch.logsumexp(sent_probs, dim=0, keepdim=True)
                sent_probs = torch.exp(sent_probs)

                return sent_probs

    def split_evaluate(self, model, distractor, src, verbose=False):
        self._log('==== Starting Split Evaluation ====\n')
        model.distractor = AsIsDistractor(distractor.orig_batch_size,
                                          distractor.d_factor,
                                          self.logger)

        torch.manual_seed(3939)
        self._log('---- Evaluating base s0 ----\n')

        total_correct = 0.
        total_srcs = 0.

        for sent in tqdm(src):
            model_in, _ = distractor.generate([sent])
            model_out = model.summarize_s0(model_in, beam_size=10, n_best=1,
                                             diverse_beam='rank')
            hypothesis = model_out[0][0]
            probs = self.get_l1_probs(model_in, hypothesis)
            if probs.argmax() == 0:
                total_correct += 1
            total_srcs += 1
            if verbose:
                self._display_one_result(model_in, hypothesis, probs)
                self._log('Accuracy up to now: {:.4}'.format(total_correct/total_srcs))


        return total_correct / total_srcs


    def _display_one_result(self, src, hypothesis, probs):
        pb_list = probs.tolist()
        self._log('[{:.4}] tgt: {}\n'.format(pb_list[0], src[0]))
        for pb, s in zip(pb_list[1:], src[1:]):
            self._log('[{:.4}] distr: {}\n'.format(pb, s))
        self._log('hypothesis: {}\n\n'.format(hypothesis))
