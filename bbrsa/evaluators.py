import math, torch

from tqdm import tqdm

from bbrsa.abstract_classes import BBRSAABC
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.utils import idx_remap, chunks
from bbrsa.distractors import AsIsDistractor, NextExampleDistractor, BertDistractor


class Evaluator(BBRSAABC):
    def __init__(self, eval_s0, opts, logger=None):
        super().__init__(logger)
        self.eval_s0 = eval_s0
        self.gpu = opts.gpu
        self.device = torch.device('cuda') if self.gpu else torch.device('cpu')

    @classmethod
    def from_opts(cls, opts, eval_s0, logger=None):
        if isinstance(eval_s0, str):
            eval_s0_model = ONMTSummarizer(opts, eval_s0, logger)
        elif isinstance(eval_s0, ONMTSummarizer):
            eval_s0_model = eval_s0
        return cls(eval_s0_model, opts, logger)

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
        batch_size = 32 * d_factor# opts.batch_size * d_factor

        s0.init_batch_iterator(src=srcs, tgt=tgts, truncate=truncate,
            batch_size=batch_size)
        pad_token = self.eval_s0.pad_token

        all_sent_probs = []

        with torch.no_grad():
            for batch in s0.data_iter:
                scrm_idxs = batch.indices % batch_size
                s0.encode(batch)
                curr_tgt = batch.tgt[1:,:,:]
                log_probs, _ = s0.decode(input=curr_tgt, batch=batch,
                                         step=None, beam_batch_offset=None)
                # [max_seq_len, batch_size*d_factor, vocab]

                reorder_idxs = idx_remap(scrm_idxs)
                reordered_probs = log_probs.index_select(1, reorder_idxs).transpose(0,1)
                # [batch_size*d_factor, max_seq_len, vocab]

                tgt_range_idxs = torch.arange(batch.batch_size // d_factor,
                                              device=self.device) * d_factor
                reordered_tgts = curr_tgt.index_select(1, reorder_idxs)
                reordered_word_idxs = reordered_tgts[:, tgt_range_idxs, 0] \
                                      .transpose(0,1)
                max_seq_len = reordered_probs.shape[1]
                vocab_size = reordered_probs.shape[2]
                reordered_probs = reordered_probs.view(-1, d_factor, max_seq_len,
                                                       vocab_size)

                word_range_idxs = torch.arange(max_seq_len, device=self.device)

                batch_sent_probs = []
                count = 0
                for word_idxs, probs in zip(reordered_word_idxs, reordered_probs):
                    mask = (word_idxs != pad_token)
                    rng_idxs = word_range_idxs[mask]
                    wd_idxs = word_idxs[mask]

                    word_probs = probs[:, rng_idxs, wd_idxs]
                    # if count == 2:
                    #     print(wd_idxs)
                    #     print(word_probs)
                    #     print(word_idxs)
                    #     print(probs[:, word_range_idxs, word_idxs])
                    sent_probs = word_probs.sum(dim=1)
                    sent_probs = sent_probs - torch.logsumexp(sent_probs, dim=0, keepdim=True)
                    sent_probs = torch.exp(sent_probs)
                    batch_sent_probs.append(sent_probs)
                    count += 1
                all_sent_probs += batch_sent_probs

        res = torch.stack(all_sent_probs)
        return res

    def split_evaluate(self, model, src, mode, opts, output_distractors=False):
        """Evaluate l1 accuracy of guessing target given summary and distractors.

        opts.eval_shard_size divides src into large shards, each of which is
        evaluated together as a big batch. This is to reduce memory constraints
        when src is very large, but also maintains efficienty by not initializing
        the batch_iter object every time for small batches in `incremental_s1()`
        or `summarize_s0()`

        Args:
            model: An instance of the `ONMTRSAModel` class that wraps together
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
        assert opts.shard_size >= opts.batch_size
        #  get the distractor that is stored in the model
        distractor = model.distractor

        self._info('>>>> Starting Split Evaluation')
        if isinstance(distractor, BertDistractor):
            d_factor = opts.bert_distr_d_factor
        else:
            d_factor = distractor.d_factor
        model.distractor = AsIsDistractor(d_factor, self.logger)

        if opts.random_seed >= 0:
            torch.manual_seed(opts.random_seed)

        total_correct = 0.
        total_srcs = 0.

        if output_distractors:
            all_distractors = []
        all_hypotheses = []
        total_shards = math.ceil(len(src) / opts.shard_size)

        for i, shard in enumerate(chunks(src, opts.shard_size)):
            self._info('>>>> SplitEval shard {}/{}'.format(i+1, total_shards))
            total_srcs += len(shard)
            model_in, _ = distractor.generate(shard, opts)
            if opts.gpu:
                torch.cuda.empty_cache()
            if output_distractors:
                all_distractors.append(model_in)
            if mode == 'incr_s1':
                model_out = model.incremental_s1(model_in, opts)
            elif mode == 's0':
                model_out = model.summarize_s0(shard, opts)
            else:
                raise NotImplementedError

            hypothesis = [s[0] for s in model_out]
            all_hypotheses += hypothesis
            self._info('>>>> Evaluating listener accuracy')
            shard_probs = self._get_batch_l1_probs(model_in, hypothesis, opts)
            pred = shard_probs.argmax(dim=1)
            total_correct += (pred == 0).sum().item()

            if total_shards > 1:
                self._info('>>>> SplitEval accuracy up to now, {:.4}'\
                           .format(total_correct / total_srcs))

        # reset model back to use original distractor
        model.distractor = distractor
        if output_distractors:
            return total_correct / total_srcs, all_hypotheses, all_distractors
        else:
            return total_correct / total_srcs, all_hypotheses


    def _display_one_result(self, src, hypothesis, probs):
        pb_list = probs.tolist()
        self._log('[{:.4}] tgt: {}\n'.format(pb_list[0], src[0]))
        for pb, s in zip(pb_list[1:], src[1:]):
            self._log('[{:.4}] distr: {}\n'.format(pb, s))
        self._log('hypothesis: {}\n\n'.format(hypothesis))
