from bbrsa.abstract_classes import Beam
from bbrsa.utils import scramble2tgt, idx_remap

from onmt.translate.beam_search import BeamSearch

import torch

class ONMTBeam(Beam):
    def __init__(self, onmt_summarizer, batch_size, beam_size=None,
        n_best=None, diverse=None, distractor=None, scramble_idxs=None,
        logger=None):
        super().__init__(logger)

        assert diverse in ['rank', None], 'Invalid diverse beam type!'

        s = onmt_summarizer
        T = onmt_summarizer.translator

        # default values are given by T
        beam_size = T.beam_size if beam_size is None else beam_size
        n_best = T.n_best if n_best is None else n_best

        if distractor is None:
            memory_lengths = s.memory_lengths
        else:
            tgt_idx = scramble2tgt(scramble_idxs, distractor.d_factor)
            memory_lengths = s.memory_lengths.view(-1, beam_size) \
                .index_select(0, tgt_idx).view(-1)

        if diverse is None:
            self.beam = BeamSearch(
                beam_size,
                n_best=n_best,
                batch_size=batch_size, # actual batch size
                global_scorer=T.global_scorer,
                pad=T._tgt_pad_idx,
                eos=T._tgt_eos_idx,
                bos=T._tgt_bos_idx,
                min_length=T.min_length,
                ratio=T.ratio,
                max_length=T.max_length,
                mb_device=s.mb_device,
                return_attention=T.replace_unk,
                stepwise_penalty=T.stepwise_penalty,
                block_ngram_repeat=T.block_ngram_repeat,
                exclusion_tokens=T._exclusion_idxs,
                memory_lengths=s.memory_lengths)
        elif diverse == 'rank':
            self.beam = RankDiverseBeam(
                beam_size,
                n_best=n_best,
                batch_size=batch_size, # actual batch size
                global_scorer=T.global_scorer,
                pad=T._tgt_pad_idx,
                eos=T._tgt_eos_idx,
                bos=T._tgt_bos_idx,
                min_length=T.min_length,
                ratio=T.ratio,
                max_length=T.max_length,
                mb_device=s.mb_device,
                return_attention=T.replace_unk,
                stepwise_penalty=T.stepwise_penalty,
                block_ngram_repeat=T.block_ngram_repeat,
                exclusion_tokens=T._exclusion_idxs,
                memory_lengths=s.memory_lengths)


    def advance(self, log_probs, attn, verbose=False):
        self.beam.advance(log_probs, attn, verbose=verbose)

    def update_finished(self):
        self.beam.update_finished()

    @property
    def current_pred(self):
        return self.beam.current_predictions.view(1, -1, 1)

    @property
    def any_beam_is_finished(self):
        return self.beam.is_finished.any()

    @property
    def is_done(self):
        return self.beam.done

    @property
    def current_origin(self):
        return self.beam.current_origin

    @property
    def predictions(self):
        return self.beam.predictions

    @property
    def batch_offset(self):
        return self.beam._batch_offset

class RankDiverseBeam(BeamSearch):
    def advance(self, log_probs, attn, verbose=False):
        # log_probs has shape [beam_size * batch_size, ext_vocab_size]
        vocab_size = log_probs.size(-1)

        # using integer division to get an integer _B without casting
        # _B is batch size
        _B = log_probs.shape[0] // self.beam_size

        # _stepwise_cov_pen means coverage penalty ###
        if self._stepwise_cov_pen and self._prev_penalty is not None:
            # need to figure out what this is doing
            self.topk_log_probs += self._prev_penalty
            self.topk_log_probs -= self.global_scorer.cov_penalty(
                self._coverage + attn, self.global_scorer.beta).view(
                _B, self.beam_size)

        # force the output to be longer than self.min_length
        step = len(self)
        self.ensure_min_length(log_probs)

        # Multiply probs by the beam probability.
        # self.topk_log_probs has shape [batch_size, beam_size] ([2,10])
        # stores probs of top 10 candidates at current time step
        log_probs += self.topk_log_probs.view(_B * self.beam_size, 1)

        self.block_ngram_repeats(log_probs)

        # if the sequence ends now, then the penalty is the current
        # length + 1, to include the EOS token
        length_penalty = self.global_scorer.length_penalty(
            step + 1, alpha=self.global_scorer.alpha)
        if verbose:
            print('length_penalty', length_penalty)

        # Flatten probs into a list of possibilities.
        curr_scores = log_probs / length_penalty
        # curr_scores = log_probs
        _, idxs = torch.sort(curr_scores, dim=1, descending=True)
        ranking = idx_remap(idxs).float() + 1.
        curr_scores -= 1.5 * ranking

        curr_scores = curr_scores.reshape(_B, self.beam_size * vocab_size)

        torch.topk(curr_scores,  self.beam_size, dim=-1,
                   out=(self.topk_scores, self.topk_ids)) #update top k scores

        # Recover log probs.
        # Length penalty is just a scalar. It doesn't matter if it's applied
        # before or after the topk.
        torch.mul(self.topk_scores, length_penalty, out=self.topk_log_probs)

        # Resolve beam origin and map to batch index flat representation.
        torch.div(self.topk_ids, vocab_size, out=self._batch_index)
        self._batch_index += self._beam_offset[:_B].unsqueeze(1)
        self.select_indices = self._batch_index.view(_B * self.beam_size)

        # -- For each prediction, select_indices gives the index that prediction
        # originates from in the beam search
        """
         prev_idx  | select_indices
                0  | 0
                1  | 1
                2  | 0
                3  | 3
                ...|
                19 | 18  <-- this prediction is emitted from prev_idx 18

        """

        self.topk_ids.fmod_(vocab_size)  # resolve true word ids

        # Append last prediction.
        """
        if verbose:
            print('--- self.alive_seq.shape', self.alive_seq.shape) #[20, 11]
            print('--- self.alive_seq.index_select(0, self.select_indices).shape:',
                self.alive_seq.index_select(0, self.select_indices).shape) #[20, 11]
            print('--- self.topk_ids.view(_B * self.beam_size, 1).shape', #[20, 1]
                self.topk_ids.view(_B * self.beam_size, 1).shape)
        """
        self.alive_seq = torch.cat(
            [self.alive_seq.index_select(0, self.select_indices),
             self.topk_ids.view(_B * self.beam_size, 1)], -1)

        """
        if verbose:
            print('--- self.alive_seq.shape', self.alive_seq.shape) #[20, 12]

        New alive_seq:
                     |   # ensure that everything in alive_seq corresponds
                0  0 |     to indices given by select_indices
                1  1 |
                0  0 |
                3  3 |
                ...  |
                18 18|
        """

        if self.return_attention or self._cov_pen:
            current_attn = attn.index_select(1, self.select_indices)
            if step == 1:
                self.alive_attn = current_attn
                # update global state (step == 1)
                if self._cov_pen:  # coverage penalty, updated here
                    self._prev_penalty = torch.zeros_like(self.topk_log_probs)
                    self._coverage = current_attn
            else:
                self.alive_attn = self.alive_attn.index_select(
                    1, self.select_indices)
                self.alive_attn = torch.cat([self.alive_attn, current_attn], 0)
                # update global state (step > 1)
                if self._cov_pen:
                    self._coverage = self._coverage.index_select(
                        1, self.select_indices)
                    self._coverage += current_attn
                    self._prev_penalty = self.global_scorer.cov_penalty(
                        self._coverage, beta=self.global_scorer.beta).view(
                            _B, self.beam_size)

        if self._vanilla_cov_pen:
            # shape: (batch_size x beam_size, 1)
            cov_penalty = self.global_scorer.cov_penalty(
                self._coverage,
                beta=self.global_scorer.beta)
            self.topk_scores -= cov_penalty.view(_B, self.beam_size)

        self.is_finished = self.topk_ids.eq(self.eos)
        self.ensure_max_length()
