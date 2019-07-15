from abstract_classes import Beam
from pragmatics import reorderidx2tgt

class ONMTBeam(Beam):
    def __init__(self, onmt_summarizer, batch_size, beam_size=None,
        n_best=None, distractor=None, reorder_idx=None):
        from onmt.translate.beam_search import BeamSearch

        s = onmt_summarizer
        T = onmt_summarizer.translator

        # default values are given by T
        beam_size = T.beam_size if beam_size is None else beam_size
        n_best = T.n_best if n_best is None else n_best

        if distractor is None:
            memory_lengths = s.memory_lengths
        else:
            tgt_idx = reorderidx2tgt(reorder_idx, distractor.d_factor)
            memory_lengths = s.memory_lengths.view(-1, beam_size) \
                .index_select(0, tgt_idx).view(-1)

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


    def advance(self, log_probs, attn):
        self.beam.advance(log_probs, attn)

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
