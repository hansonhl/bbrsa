import torch
import sys, os
from abc import ABC, abstractmethod
from models import ONMTSummarizer
from beam import ONMTBeam
from pragmatics import NextExampleDistractor, BasicPragmatics

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
        self.pragmatics = BasicPragmatics(
            batch_size=default_beam_size,
            d_factor=self.distractor.d_factor,
            beam_size=default_beam_size)

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
            s0.set_configs(beam_size=beam_size, n_best=n_best)

            src, batch_size = self.distractor.generate(src)

            s0.init_batch_iterator(src, batch_size)

            for batch in s0.data_iter:
                # batch.indices contains the reordering index
                s0.encode(batch)
                s0.batch_augment(batch, beam_size)
                s0.enc_states_augment(beam_size)
                s0.dec_states_augment(beam_size)
                max_length = s0.max_output_length

                beam_batch_size = batch.batch_size // self.distractor.d_factor
                # actual batch size, if batch has fewer examples than max batch size

                beam = ONMTBeam(s0,
                    batch_size=beam_batch_size,
                    beam_size=beam_size,
                    n_best=n_best,
                    distractor=self.distractor,
                    reorder_idx=batch.indices)

                for step in range(max_length):
                    decoder_input = augment_dec_input(beam.current_pred,
                        beam_size, self.distractor.d_factor, batch.indices)

                    beam_batch_offset = list(range(len(beam.batch_offset) * \
                        self.distractor.d_factor))

                    log_probs, attn = s0.decode(
                        input=decoder_input,
                        batch=batch,
                        step=step,
                        beam_batch_offset=beam_batch_offset)

                    log_probs = reshape_log_probs(
                        log_probs=log_probs,
                        beam_size=beam_size,
                        d_factor=self.distractor.d_factor,
                        idxs=batch.indices)
                    print('log_probs.shape', log_probs.shape)
                    print('log_probs[:6,:,:10]', log_probs[:6,:,:10])

                    return

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

                batch_preds = self.itos(beam.predictions)
                preds += batch_preds
            # end for batch in iter
        # end with no_grad
        return preds

def augment_dec_input(dec_input, beam_size, d_factor, idxs):
    """Given beam output for 2 targets, repeat and scramble for decoder input"""
    res = dec_input.view(-1, beam_size).repeat_interleave(d_factor, dim=0)
    res = res.index_select(0, idxs).view(1, -1, 1)
    return res

def reshape_log_probs(log_probs, beam_size, d_factor, idxs):
    vocab_size = log_probs.shape[-1]
    restore_order_idxs = torch.tensor([p[0] for p in \
        sorted(list(zip(range(len(idxs)),idxs)), key=lambda q: q[1])])
    log_probs = log_probs.view(-1, beam_size, vocab_size) \
                         .index_select(0, restore_order_idxs) \
                         .view(-1, d_factor, beam_size, vocab_size)\
                         .permute(0,2,1,3).contiguous() \
                         .view(-1, d_factor, vocab_size)
    #[B*d*b, V] -> [B*d, b, V] -> [B*d, b, V] (distractors grouped with tgts)
    #           -> [B, d, b, V] -> [B, b, d, V] -> [B*b, d, V]
    return log_probs
