import torch

from tqdm import tqdm

from bbrsa.abstract_classes import BBRSAABC
from bbrsa.utils import idx_remap, chunks
from bbrsa.distractors import AsIsDistractor, NextExampleDistractor


class Evaluator(BBRSAABC):
    def __init__(self, eval_s0, opts, logger=None):
        super().__init__(logger)
        self.eval_s0 = eval_s0
        self.gpu = opts.gpu
        self.device = torch.device('cuda') if self.gpu else torch.device('cpu')

    def _get_batch_l1_probs(self, srcs, summary, opts):
        """ Given target and distractors and a summary, get L1 probability for the target

        Currently only supports one target and one target+distractor set.

        Args:
            srcs: a python list of tokenized strings, shape
                ``[batch_size * d_factor, ...]``
            tgt: a string list of summaries for target articles, shape
                ``[<=batch_size,]``
            truncate: (default None) whether to truncate the srcs
        """
        add_tags = opts.add_tags
        truncate = opts.truncate
        assert len(srcs) % len(summary) == 0, 'must be divisible!'
        d_factor = len(srcs) // len(summary)

        tgts = []
        for s in summary:
            s = s.strip()
            if add_tags:
                s = '<t> ' + s + ' </t>'
            tgts += [s] * d_factor

        s0 = self.eval_s0
        batch_size = opts.batch_size * d_factor

        s0.init_batch_iterator(src=srcs, tgt=tgts, truncate=truncate,
            batch_size=batch_size)
        pad_token = self.eval_s0.pad_token

        all_sent_probs = []

        with torch.no_grad():
            for batch in s0.data_iter:
                scrm_idxs = batch.indices % batch_size
                s0.encode(batch)
                decoder_inputs = batch.tgt
                log_probs, _ = s0.decode(
                    input=decoder_inputs,
                    batch=batch,
                    step=None,
                    beam_batch_offset=None)

                reorder_idxs = idx_remap(scrm_idxs)
                # reordered_srcs = batch.src[0].index_select(1, reorder_idxs).transpose(0, 1)
                # print('reordered_srcs', reordered_srcs)
                reordered_probs = log_probs.index_select(1, reorder_idxs).transpose(0,1)
                tgt_range_idxs = torch.arange(batch.batch_size // d_factor,
                                              device=self.device) * d_factor
                word_idxs = batch.tgt[:, tgt_range_idxs, 0].transpose(0,1)
                max_seq_len = reordered_probs.shape[1]
                vocab_size = reordered_probs.shape[2]
                reordered_probs = reordered_probs.view(-1, d_factor, max_seq_len,
                                                       vocab_size)

                word_range_idxs = torch.arange(max_seq_len, device=self.device)

                batch_sent_probs = []
                for wd_idxs, probs in zip(word_idxs, reordered_probs):
                    mask = (wd_idxs != pad_token)
                    rng_idxs = word_range_idxs[mask]
                    wd_idxs = wd_idxs[mask]
                    word_probs = probs[:, rng_idxs, wd_idxs]
                    sent_probs = word_probs.sum(dim=1)
                    sent_probs = sent_probs - torch.logsumexp(sent_probs, dim=0, keepdim=True)
                    sent_probs = torch.exp(sent_probs)
                    batch_sent_probs.append(sent_probs)
                all_sent_probs += batch_sent_probs

        res = torch.stack(all_sent_probs)
        return res

    def split_evaluate(self, model, distractor, src, opts, mode='incr_s1', verbose=False):
        """Evaluate l1 accuracy of guessing target given summary and distractors.

        opts.eval_shard_size divides src into large shards, each of which is
        evaluated together as a big batch. This is to reduce memory constraints
        when src is very large, but also maintains efficienty by not initializing
        the batch_iter object every time for small batches in `incremental_s1()`
        or `summarize_s0()`

        Args:
            model: An instance of the `ONMTSummaryRSA` class that wraps together
                a base s0 speaker, a distractor generation method, and a
                pragmatics reasoning scheme. The distractor as defined in the
                model here is not actually used, as it is replaced with the
                `AsIsDistractor`.
            distractor: The distractor generator that's used.
            src: Target articles to summarize
            opts: A `ConfigOpts` object that encapsulates all parameter settings
            mode: which to evaluate, should be in the following list:
                ['incr_s1', 'global_s1', 's0']

        Returns:
            Accuracy, as a scalar
        """
        assert mode in ['incr_s1', 'global_s1', 's0']
        assert opts.eval_shard_size >= opts.batch_size

        self._log('==== Starting Split Evaluation ====\n')
        model.distractor = AsIsDistractor(distractor.d_factor, self.logger)

        torch.manual_seed(opts.random_seed)

        total_correct = 0.
        total_srcs = 0.

        for i, shard in enumerate(chunks(src, opts.eval_shard_size)):
            total_srcs += len(shard)
            model_in, _ = distractor.generate(shard, opts)
            if mode == 'incr_s1':
                model_out = model.incremental_s1(model_in, opts)
            else:
                raise NotImplementedError

            hypothesis = [s[0] for s in model_out]
            shard_probs = self._get_batch_l1_probs(model_in, hypothesis, opts)
            pred = shard_probs.argmax(dim=1)
            total_correct += (pred == 0).sum().item()

            if verbose:
                self._info('Shard {}, accuracy up to now, {:.4}'.format(i, total_correct / total_srcs))

            # if verbose:
            #     self._display_one_result(model_in, hypothesis, probs)
            #     self._log('Accuracy up to now: {:.4}'.format(total_correct/(i+1)))

        return total_correct / total_srcs


    def _display_one_result(self, src, hypothesis, probs):
        pb_list = probs.tolist()
        self._log('[{:.4}] tgt: {}\n'.format(pb_list[0], src[0]))
        for pb, s in zip(pb_list[1:], src[1:]):
            self._log('[{:.4}] distr: {}\n'.format(pb, s))
        self._log('hypothesis: {}\n\n'.format(hypothesis))
