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

    def itos(self, idxs, batch, reordered=False, d_factor=None):
        """Converts arrays of indices to text"""
        # reordered is false when idx order is scrambled (for summarize_with_s0)
        # batch provides extended vocab and scrambled indices

        # idxs is a [batch_size, n_best] 2d list of tensors of ids
        # uses method defined in TranslationBuilder._build_target_tokens()
        base_itos = dict(self.s0.translator.fields)['tgt'].base_field.vocab.itos
        base_size = len(base_itos)
        preds = []
        scramble_idxs = batch.indices
        if reordered:
            tgt_idxs = scramble2tgt(scramble_idxs, d_factor)
            tgt_ex_vocabs = [batch.src_ex_vocab[i] for i in tgt_idxs]
        else:
            tgt_ex_vocabs = batch.src_ex_vocab

        for b, sent in enumerate(idxs):
            candidates = []
            ext_itos = tgt_ex_vocabs[b].itos
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
                        print('Invalid token!')
                candidates.append(' '.join(tokens))
            preds.append(candidates)

        if not reordered:
            reorder_idxs = idx_remap(scramble_idxs)
            preds = [preds[i] for i in reorder_idxs]
        return preds

    def summarize_with_s0(self, src, beam_size=1, n_best=1):
        preds = []
        #all_log_probs = []
        #curr_preds = []
        with torch.no_grad():
            s0 = self.s0


            s0.set_configs(beam_size=beam_size, n_best=n_best)
            s0.init_batch_iterator(src)

            for batch in s0.data_iter:
                s0.encode(batch)
                s0.batch_augment(batch, beam_size)
                s0.enc_states_augment(beam_size)
                s0.dec_states_augment(beam_size)
                max_length = s0.max_output_length
                beam_batch_size = batch.batch_size
                scramble_idxs = batch.indices
                # actual batch size, if batch has fewer examples than max batch size

                beam = ONMTBeam(s0, batch_size=beam_batch_size, \
                    beam_size=beam_size)

                for step in range(max_length):
                    decoder_input = beam.current_pred
                    beam_batch_offset = beam.batch_offset

                    log_probs, attn = s0.decode(decoder_input, batch, step, \
                        beam_batch_offset)

                    #print('----step', step)
                    #print(log_probs[:, :6])

                    # reorder_idxs = idx_remap(scramble_idxs)
                    # vocab_size = log_probs.shape[-1]
                    # reshaped = log_probs.view(-1, beam_size, vocab_size) \
                    #                     .index_select(0, reorder_idxs) \
                    #                     .view(-1, vocab_size)
                    # all_log_probs.append(reshaped)
                    beam.advance(log_probs, attn)

                    any_beam_is_finished = beam.any_beam_is_finished
                    if any_beam_is_finished:
                        beam.update_finished()
                        if beam.is_done:
                            break

                    select_indices = beam.current_origin
                    # print('select_idxs', select_indices)

                    # res = select_indices.view(-1, beam_size)    # [2,1]
                    # B = res.shape[0]
                    # if B != scramble_idxs.shape[0]:
                    #     rem_tgts = res[:, 0] // beam_size
                    #     rem_tgts_idx_map = torch.ones(scramble_idxs.shape[0], dtype=torch.long) * -1
                    #     for i, x in enumerate(rem_tgts):
                    #         rem_tgts_idx_map[x] = i
                    #     new_scramble_idxs = []
                    #     for i, x in enumerate(scramble_idxs):
                    #         if rem_tgts_idx_map[x] != -1:
                    #             new_scramble_idxs.append(rem_tgts_idx_map[x])
                    #     new_scramble_idxs = torch.tensor(new_scramble_idxs)
                    #     scramble_idxs = new_scramble_idxs
                    # reorder_idxs = idx_remap(scramble_idxs)

                    if any_beam_is_finished:
                        # print('----rearrange happened')
                        s0.enc_states_rearrange(select_indices)
                        s0.batch_rearrange(batch, select_indices)

                    s0.dec_states_rearrange(select_indices)
                    # curr_preds.append(beam.current_pred.view(-1).index_select(0, reorder_idxs))

                batch_preds = self.itos(beam.predictions, batch)
                preds += batch_preds

            # end for batch in iter
        # end with no_grad
        return preds #, all_log_probs, curr_preds

    def summarize_with_distractor(self, src, beam_size=1, n_best=1):
        """Summarize source text using RSA."""
        assert self.distractor is not None, 'Must specify distractor!'
        preds = []
        # all_log_probs = []
        # curr_preds = []
        with torch.no_grad():
            s0 = self.s0
            d_factor = self.distractor.d_factor
            s0.set_configs(beam_size=beam_size, n_best=n_best)
            src, batch_size = self.distractor.generate(src)
            s0.init_batch_iterator(src, batch_size)

            for batch in s0.data_iter:
                scramble_idxs = batch.indices
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
                    # print('----step', step)
                    # print(log_probs[:,:6])

                    # log_probs_for_debug = torch.FloatTensor(log_probs)
                    s0_log_probs = _reshape_dec2prag(log_probs, beam_size,
                        d_factor, scramble_idxs) #[B*b, d, V] #ok
                    attn = _reshape_attn(attn, beam_size, d_factor, scramble_idxs)

                    # s1_log_probs = s0_log_probs
                    s1_log_probs = self.pragmatics.inference(s0_log_probs) #[B*b, d, V]

                    beam_log_probs = _reshape_prag2beam(s1_log_probs, beam_size,
                        d_factor, scramble_idxs) #ok
                    # _log_probs_debug(step, beam_size, d_factor, beam_log_probs, log_probs_for_debug, scramble_idxs)

                    # all_log_probs.append(beam_log_probs)
                    beam.advance(beam_log_probs, attn)

                    any_beam_is_finished = beam.any_beam_is_finished
                    if any_beam_is_finished:
                        beam.update_finished()
                        if beam.is_done:
                            break

                    # print('select_idxs before scramble', beam.current_origin)

                    select_indices, scramble_idxs = \
                        _reshape_select_idxs_and_rescramble(
                        beam.current_origin, beam_size, d_factor, scramble_idxs,
                        step=step)
                    # print('select_idxs after scramble', select_indices)

                    # print('step', step, 'current_prediction:')
                    # print('    ', self.itos_single_array(beam.current_pred.squeeze(), beam_size))
                    if any_beam_is_finished:
                        # print('----rearrange happened')
                        s0.enc_states_rearrange(select_indices)
                        s0.batch_rearrange(batch, select_indices)
                    s0.dec_states_rearrange(select_indices)
                    # curr_preds.append(beam.current_pred.view(-1))
                # end for step
                batch_preds = self.itos(beam.predictions, batch, reordered=True, d_factor=d_factor)
                preds += batch_preds

            # end for batch
        # end with no_grad
        return preds #, all_log_probs, curr_preds

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

    diff = torch.gt(torch.abs(torch.add(res, -beam_log_probs)), 1e-10)

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
    # print('    select_idxs', input)
    B = res.shape[0] # B = batch_size

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
        # ***assume rem_tgts is in ascending order****
        # e.g. d_factor = 2, beam_size = 10
        # scramble_idxs = [4, 2, 0, 5, 1, 3]
        # res = [[0,2,4,5,6,7,8,7,8,9],[21,24,22,25,26,24,28,29,29,22]]
        rem_tgts = res[:, 0] // beam_size # remaining targets [0, 2]
        rem_tgts_idxs = ((rem_tgts.repeat_interleave(d_factor, dim=0) * d_factor) \
            .view(-1, d_factor) + torch.arange(d_factor)).view(-1) #[0,1,4,5]

        # get new scramble idxs given old scramble idxs and remaining targets
        #
        old_scr_len = scramble_idxs.shape[0]
        rem_tgts_idx_map = torch.ones(old_scr_len, dtype=torch.long) * -1
        for i, x in enumerate(rem_tgts_idxs):
            rem_tgts_idx_map[x] = i                           # [0,1,-1,-1,2,3]
        scrambled_rem_tgts = rem_tgts_idx_map[scramble_idxs]  # [2,-1,0,3,1,-1]
        rem_mask = rem_tgts_idx_map[scramble_idxs] != -1      # [1, 0,1,1,1, 0]
        new_scramble_idxs = scrambled_rem_tgts[rem_mask]          #[2,0,3,1]
        filtered_sel_idxs = rem_mask.nonzero().view(-1)  #[0,2,3,4]

        # new_scramble_idxs = []
        # for i, x in enumerate(scramble_idxs):
        #     if rem_tgts_idx_map[x] != -1:
        #         new_scramble_idxs.append(rem_tgts_idx_map[x])
        # new_scramble_idxs = torch.tensor(new_scramble_idxs)

        # e.g.     res = [[0,2,4,5,6,7,8,7,8,9],[21,24,22,25,26,24,28,29,29,22]]
        offset = (rem_tgts * beam_size).view(-1, 1)                # [[0], [20]]
        res -= offset             #[[0,2,4,5,6,7,8,7,8,9],[1,4,2,5,6,4,8,9,9,2]]
        res = res.repeat_interleave(d_factor, dim=0) \
                 .index_select(0, new_scramble_idxs)
                                          #[[0,2...],[0,2...],[1,4...],[1,4...]]
                                          #[[1,4...],[0,2...],[1,4...],[0,2...]]
        offset = (filtered_sel_idxs * beam_size).view(-1, 1)
                                                #[0,2,3,4]->[[0],[20],[30],[40]]
        res += offset
        res = res.view(-1)

        return res, new_scramble_idxs
