import torch
import sys, os
from abc import ABC, abstractmethod
from models import ONMTSummarizer
from beam import ONMTBeam
from distractors import NextExampleDistractor

ONMT_DIR = '../myOpenNMT'

class BatchBeamRSA(ABC):
    pass

class ONMTSummaryRSA(BatchBeamRSA):
    def __init__(self, onmt_dir=ONMT_DIR):
        # managing package import for OpenNMT
        sys.path.insert(0, os.path.abspath(ONMT_DIR)) # ../myOpenNMT
        # https://askubuntu.com/questions/470982/how-to-add-a-python-module-to-syspath

        print('Initializing ONMT summarizer')
        self.s0 = ONMTSummarizer()
        self.distractor = NextExampleDistractor(batch_size=self.s0.batch_size)
        print('Finished initialization')

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


    def summarize(self, src, beam_size=1, n_best=1):
        preds = []
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
                # actual batch size, if batch has fewer examples than max batch size

                # TODO: Setup beam search considering reordering
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

                batch_preds = self.itos(beam.predictions)
                preds += batch_preds
            # end for batch in iter
        # end with no_grad
        return preds

def scrambled2tgt(idxs, d_factor):
    """Given reordering indices, get indices of target examples"""
    # e.g. input [1,2,0,3]: out [2, 1], i.e. elements with id 2 and 1 in
    # list [1,2,0,3] refer to two targets respectively, in correct order
    if isinstance(idxs, torch.Tensor):
        idxs = idxs.tolist()

    pairs = list(zip(range(len(idxs)), idxs))
    f = list(filter(lambda p: p[1] % d_factor == 0, pairs))
    return [p[0] for p in sorted(f, key=lambda q: q[1])]

def augment_for_dec_input(dec_input, d_factor, idxs):
    """Given beam output for 2 targets, repeat and scramble for decoder input"""
    new_input = torch.repeat_interleave(dec_input, d_factor, dim=1)
    return(new_input.index_select(1, idxs))
