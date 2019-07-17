import sys, os
import copy
import torch
from abc import ABC, abstractmethod

from beam import ONMTBeam
from pragmatics import NextExampleDistractor, BasicPragmatics, idx_remap, scramble2tgt
from torchtext.data.batch import Batch as TorchBatch

class BatchBeamRSA(ABC):
    pass

class ONMTSummaryRSA(BatchBeamRSA):
    def __init__(self, s0, pragmatics, distractor):

        self.s0 = s0
        self.distractor = distractor
        self.pragmatics = pragmatics

    def itos_single_array(self, idxs, beam_size):
        """Convert one single array to text"""
        base_itos = dict(self.s0.translator.fields)['tgt'].base_field.vocab.itos
        base_size = len(base_itos)
        tokens = []
        for i, tok in enumerate(idxs):
            b = i // beam_size
            ext_itos = self.s0.data.src_vocabs[b].itos
            ext_size = len(ext_itos)
            full_size = base_size + ext_size
            if tok < len(base_itos):
                tok_str = base_itos[tok]
                if tok_str != '</s>':
                    tokens.append(tok_str)
            elif tok < full_size:
                tok_str = ext_itos[tok - len(base_itos)]
                if tok_str != '</s>':
                    tokens.append(tok_str)
            else:
                token.append('<invalid>')
        return tokens

    def itos(self, idxs, batch, scramble_idxs=None):
        """Converts arrays of indices to text"""
        # scramble_idxs is not none: used when idx order is scrambled

        # idxs is a [batch_size, n_best] 2d list of tensors of ids
        # uses method defined in TranslationBuilder._build_target_tokens()
        base_itos = dict(self.s0.translator.fields)['tgt'].base_field.vocab.itos
        base_size = len(base_itos)
        preds = []
        for b, sent in enumerate(idxs):
            candidates = []
            ext_itos = self.s0.data.src_vocabs[b].itos
            ext_size = len(ext_itos)
            full_size = base_size + ext_size
            for i in sent:
                tokens = []
                for tok in i:
                    if tok < len(base_itos):
                        tok_str = base_itos[tok]
                        if tok_str != '</s>':
                            tokens.append(tok_str)
                    elif tok < full_size:
                        tok_str = ext_itos[tok - len(base_itos)]
                        if tok_str != '</s>':
                            tokens.append(tok_str)
                    else:
                        break
                candidates.append(' '.join(tokens))
            preds.append(candidates)
        if scramble_idxs is not None:
            reorder_idxs = idx_remap(scramble_idxs)
            preds = [preds[i] for i in reorder_idxs]
        return preds

    def summarize_with_s0(self, src, beam_size=1, n_best=1):
        preds = []
        with torch.no_grad():
            s0 = self.s0

            s0.set_configs(beam_size=beam_size, n_best=n_best)
            s0.init_batch_iterator(src)

            for batch in s0.data_iter:
                # result_dict = {
                #     "predictions": None,
                #     "scores": None,
                #     "attention": None,
                #     "batch": batch,
                #     "gold_score": [0] * batch.batch_size}

                s0.encode(batch)
                s0.batch_augment(batch, beam_size)
                s0.enc_states_augment(beam_size)
                s0.dec_states_augment(beam_size)
                max_length = s0.max_output_length
                beam_batch_size = batch.batch_size
                # actual batch size, if batch has fewer examples than max batch size

                beam = ONMTBeam(s0, batch_size=beam_batch_size, \
                    beam_size=beam_size)

                for step in range(max_length):
                    decoder_input = beam.current_pred
                    beam_batch_offset = beam.batch_offset

                    log_probs, attn = s0.decode(decoder_input, batch, step, \
                        beam_batch_offset)

                    beam.advance(log_probs, attn)

                    any_beam_is_finished = beam.any_beam_is_finished
                    if any_beam_is_finished:
                        beam.update_finished()
                        if beam.is_done:
                            break

                    select_indices = beam.current_origin

                    if any_beam_is_finished:
                        s0.enc_states_rearrange(select_indices)
                        s0.batch_rearrange(batch, select_indices)

                    s0.dec_states_rearrange(select_indices)


                # result_dict["scores"] = beam.beam.scores
                # result_dict["predictions"] = beam.predictions # [[tensor], [tensor]]
                # result_dict["attention"] = beam.beam.attention
                #
                # translations = s0.xlation_builder.from_batch(result_dict)
                # for trans in translations:
                #     n_best_preds = [" ".join(pred)
                #                     for pred in trans.pred_sents[:n_best]]
                #     preds += [n_best_preds]
            # end for batch in iter
        # end with no_grad
        return preds

    def summarize_with_distractor(self, src, beam_size=1, n_best=1):
        """Summarize source text using RSA."""
        assert self.distractor is not None, 'Must specify distractor!'
        preds = []
        with torch.no_grad():
            s0 = self.s0
            d_factor = self.distractor.d_factor
            s0.set_configs(beam_size=beam_size, n_best=n_best)
            src, batch_size = self.distractor.generate(src)
            s0.init_batch_iterator(src, batch_size)

            for batch in s0.data_iter:
                scramble_idxs = batch.indices

                result_dict = {
                    "predictions": None,
                    "scores": None,
                    "attention": None,
                    "batch": batch,
                    "gold_score": [0] * batch.batch_size}
                # batch.indices contains the scrambling index
                # original order -> index_select(batch.indices) -> current order
                s0.encode(batch)
                s0.batch_augment(batch, beam_size)
                s0.enc_states_augment(beam_size)
                s0.dec_states_augment(beam_size)
                max_length = s0.max_output_length

                # Beam only generates output for target text. The following
                # give the batch size from the perspective of the beam.
                # The elements in the beam are in their original unscrambled order
                beam_batch_size = batch.batch_size // d_factor

                beam = ONMTBeam(s0,
                    batch_size=beam_batch_size,
                    beam_size=beam_size,
                    n_best=n_best,
                    distractor=self.distractor,
                    scramble_idxs=batch.indices)

                for step in range(max_length):
                    decoder_input = _reshape_beam2dec(beam.current_pred,
                        beam_size, d_factor, scramble_idxs)
                    beam_batch_offset = list(range(len(beam.batch_offset) * \
                        self.distractor.d_factor))

                    log_probs, attn = s0.decode(
                        input=decoder_input,
                        batch=batch,
                        step=step,
                        beam_batch_offset=beam_batch_offset)

                    # log_probs_for_debug = torch.FloatTensor(log_probs)

                    s0_log_probs = _reshape_dec2prag(log_probs, beam_size,
                        d_factor, scramble_idxs) #[B*b, d, V]
                    attn = _reshape_attn(attn, beam_size, d_factor, scramble_idxs)

                    s1_log_probs = s0_log_probs
                    # s1_log_probs = self.pragmatics.inference(s0_log_probs) #[B*b, d, V]

                    beam_log_probs = _reshape_prag2beam(s1_log_probs, beam_size,
                        d_factor, scramble_idxs)

                    # _log_probs_debug(step, beam_size, d_factor, beam_log_probs, log_probs_for_debug, scramble_idxs)

                    beam.advance(beam_log_probs, attn)

                    any_beam_is_finished = beam.any_beam_is_finished
                    if any_beam_is_finished:
                        beam.update_finished()
                        if beam.is_done:
                            break

                    select_indices, scramble_idxs = \
                        _reshape_select_idxs_and_rescramble(
                        beam.current_origin, beam_size, d_factor, scramble_idxs, step=step)

                    # print('step', step, 'current_prediction:')
                    # print('    ', self.itos_single_array(beam.current_pred.squeeze(), beam_size))
                    if any_beam_is_finished:
                        s0.enc_states_rearrange(select_indices)
                        s0.batch_rearrange(batch, select_indices)
                    s0.dec_states_rearrange(select_indices)

                result_dict["scores"] = _dup_and_scramble(
                    beam.beam.scores, beam_size, d_factor, batch.indices)
                result_dict["predictions"] = _dup_and_scramble(
                    beam.predictions, beam_size, d_factor, batch.indices)
                print(result_dict["predictions"])
                print(batch.src[0])
                result_dict["attention"] = _dup_and_scramble(
                    beam.beam.attention, beam_size, d_factor, batch.indices)

                translations = s0.xlation_builder.from_batch(result_dict)
                for trans in translations:
                    n_best_preds = [" ".join(pred)
                                    for pred in trans.pred_sents[:n_best]]
                    preds += [n_best_preds]
            # end for batch in iter
        # end with no_grad
        return preds

def _dup_and_scramble(input, beam_size, d_factor, scramble_idxs):
    dup = []
    for x in input:
        dup += ([x] * d_factor)

    res2 = [dup[i] for i in scramble_idxs]
    return res2

def _reshape_beam2dec(input, beam_size, d_factor, scramble_idxs):
    """Given beam output for targets, repeat and scramble for decoder input"""
    # [1, B*b, 1] -> [B, b] -> [B*d, b] (repeat for distractors)
    #             -> [B*d, b] (dim 0 scrambled) -> [1, B*d*b, 1]
    res = input.view(-1, beam_size) \
               .repeat_interleave(d_factor, dim=0) \
               .index_select(0, scramble_idxs) \
               .view(1, -1, 1)
    return res

def _reshape_dec2prag(input, beam_size, d_factor, scramble_idxs):
    """Reshape decoder output for input into pragmatics"""
    # [B*d*b, V] -> [B*d, b, V] -> [B*d, b, V] (dim 0 reordered)
    #            -> [B, d, b, V] -> [B, b, d, V] -> [B*b, d, V]
    vocab_size = input.shape[-1]
    reorder_idxs = idx_remap(scramble_idxs)
    res = input.view(-1, beam_size, vocab_size) \
               .index_select(0, reorder_idxs) \
               .view(-1, d_factor, beam_size, vocab_size)\
               .permute(0,2,1,3).contiguous() \
               .view(-1, d_factor, vocab_size)
    return res

def _log_probs_debug(step, beam_size, d_factor, beam_log_probs, log_probs, scramble_idxs):
    print('----Step', step, 'debugging log probs')

    vocab_size = log_probs.shape[-1]
    reorder_idxs = idx_remap(scramble_idxs)
    res = log_probs.view(-1, beam_size, vocab_size)
    res = res.index_select(0, reorder_idxs)
    res = res.view(-1, d_factor, beam_size, vocab_size)
    res = res[:, 0, :, :].contiguous()
    res = res.view(-1, vocab_size)

    diff = torch.gt(torch.abs(torch.add(res, -beam_log_probs)), 1e-6)

    # diff = torch.ne(res, beam_log_probs)
    resdiff = res[diff]
    print('    how many are different?', resdiff.shape)

def _reshape_prag2beam(input, beam_size, d_factor, scramble_idxs):
    """Reshape pragmatics output for input into beam search"""
    # input [B*b, d, V] -> output [B*b, V]
    vocab_size = input.shape[-1]
    res = input[:, 0, :].reshape(-1, vocab_size)
    return res

def _reshape_attn(input, beam_size, d_factor, scramble_idxs):
    """Reshape attn from decoder for beam search"""
    # [1, B*d*b, L] -> [B*d, b, L] -> [B, b, L] -> [1, B*b, L]
    max_len = input.shape[-1]
    tgt_idxs = scramble2tgt(scramble_idxs, d_factor)
    res = input.view(-1, beam_size, max_len) \
               .index_select(0, tgt_idxs) \
               .view(1, -1, max_len)
    return res


def _reshape_select_idxs_and_rescramble(input, beam_size, d_factor, scramble_idxs, step=None):
    """Reshape select indices from beam for rearranging states"""
    # print('----Debugging scrabmle in step', step)
    res = input.view(-1, beam_size)    # [2,1]
    # print('    res.shape', res.shape)
    B = res.shape[0]

    if B * d_factor == scramble_idxs.shape[0]:
        offset = (torch.arange(B) * beam_size).view(-1, 1)
        res -= offset
        res = res.repeat_interleave(d_factor, dim=0) \
                 .index_select(0, scramble_idxs)
        offset = (torch.arange(res.shape[0]) * beam_size).view(-1, 1)
        res += offset
        res = res.view(-1)

        return res, scramble_idxs
    else:
        # get indices of remaining targets in original, unscrambled order
        rem_tgts = res[:, 0] // beam_size # remaining targets
        rem_tgts_idxs = ((rem_tgts.repeat_interleave(d_factor, dim=0) * d_factor) \
            .view(-1, d_factor) + torch.arange(d_factor)).view(-1)

        # get new scramble idxs given old scramble idxs and remaining targets
        rem_tgts_idx_map = torch.ones(scramble_idxs.shape[0], dtype=torch.long) * -1
        for i, x in enumerate(rem_tgts_idxs):
            rem_tgts_idx_map[x] = i
        new_scramble_idxs = []
        for i, x in enumerate(scramble_idxs):
            if rem_tgts_idx_map[x] != -1:
                new_scramble_idxs.append(rem_tgts_idx_map[x])
        new_scramble_idxs = torch.tensor(new_scramble_idxs)

        # filter out old scramble idxs of remaining targets
        scramble_mask = rem_tgts_idx_map[scramble_idxs] != -1
        filtered_scramble_idxs = scramble_idxs[scramble_mask]

        offset = (rem_tgts * beam_size).view(-1, 1)
        res -= offset
        res = res.repeat_interleave(d_factor, dim=0) \
                 .index_select(0, new_scramble_idxs)
        offset = (filtered_scramble_idxs * beam_size).view(-1, 1)
        res += offset
        res = res.view(-1)

        return res, new_scramble_idxs
