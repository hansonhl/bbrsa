import torch
import sys, os
from abc import ABC, abstractmethod
from models import ONMTSummarizer
from beam import ONMTBeam
from pragmatics import NextExampleDistractor, BasicPragmatics, idx_remap, scramble2tgt

ONMT_DIR = '../myOpenNMT'

class BatchBeamRSA(ABC):
    pass

class ONMTSummaryRSA(BatchBeamRSA):
    def __init__(self, onmt_dir=ONMT_DIR):
        # managing package import for OpenNMT
        sys.path.insert(0, os.path.abspath(ONMT_DIR)) # ../myOpenNMT
        # https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath
        self.s0 = ONMTSummarizer()
        default_batch_size = self.s0.opt.batch_size
        default_beam_size = self.s0.translator.beam_size

        self.distractor = NextExampleDistractor(
            batch_size=default_batch_size)
        self.pragmatics = BasicPragmatics()

    def itos(self, idxs):
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
                        tokens.append(base_itos[tok])
                    elif tok < full_size:
                        tokens.append(ext_itos[tok - len(base_itos)])
                    else:
                        break
                candidates.append(' '.join(tokens))
            preds.append(candidates)
        return preds

    def summarize_with_distractor(self, src, beam_size=1, n_best=1):
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

                    s0_log_probs = _reshape_dec2prag(log_probs, beam_size,
                        d_factor, scramble_idxs) #[B*b, d, V]
                    attn = _reshape_attn(attn, beam_size, d_factor, scramble_idxs)

                    s1_log_probs = self.pragmatics.inference(s0_log_probs) #[B*b, d, V]

                    log_probs = _reshape_prag2beam(s1_log_probs, beam_size,
                        d_factor, scramble_idxs)

                    beam.advance(log_probs, attn)

                    any_beam_is_finished = beam.any_beam_is_finished
                    if any_beam_is_finished:
                        beam.update_finished()
                        if beam.is_done:
                            break

                    select_indices, scramble_idxs = \
                        _reshape_select_idxs_and_rescramble(
                        beam.current_origin, beam_size, d_factor, scramble_idxs)

                    if any_beam_is_finished:
                        s0.enc_states_rearrange(select_indices)
                        s0.batch_rearrange(batch, select_indices)

                    s0.dec_states_rearrange(select_indices)

                batch_preds = self.itos(beam.predictions)
                preds += batch_preds
            # end for batch in iter
        # end with no_grad
        return preds

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

def _reshape_prag2beam(input, beam_size, d_factor, scramble_idxs):
    """Reshape pragmatics output for input into beam search"""
    # input [B*b, d, V] -> output [B*b, V]
    return input[:, 0, :].squeeze()

def _reshape_attn(input, beam_size, d_factor, scramble_idxs):
    """Reshape attn from decoder for beam search"""
    # [1, B*d*b, L] -> [B*d, b, L] -> [B, b, L] -> [1, B*b, L]
    max_len = input.shape[-1]
    tgt_idxs = scramble2tgt(scramble_idxs, d_factor)
    res = input.view(-1, beam_size, max_len) \
               .index_select(0, tgt_idxs) \
               .view(1, -1, max_len)
    return res


def _reshape_select_idxs_and_rescramble(input, beam_size, d_factor, scramble_idxs):
    """Reshape select indices from beam for rearranging states"""
    res = input.view(-1, beam_size)
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
        rem_tgts = res[:, 0] // beam_size # remaining targets
        num_rem_tgts = rem_tgts.shape[0]
        num_rem_tot = rem_tgts.shape[0] * d_factor
        print('rem_tgts', rem_tgts)
        rem_tgts_idxs = ((rem_tgts.repeat_interleave(d_factor, dim=0) * d_factor) \
            .view(-1, d_factor) + torch.arange(d_factor)).view(-1)
        print('rem_tgts_idxs', rem_tgts_idxs)

        rem_tgts_idx_map = torch.ones(scramble_idxs.shape[0], dtype=torch.long) * -1
        for i, x in enumerate(rem_tgts_idxs):
            rem_tgts_idx_map[x] = i
        print('rem_tgts_idx_map', rem_tgts_idx_map)
        new_scramble_idxs = []
        for i, x in enumerate(scramble_idxs):
            if rem_tgts_idx_map[x] != -1:
                new_scramble_idxs.append(rem_tgts_idx_map[x])
        new_scramble_idxs = torch.tensor(new_scramble_idxs)

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
