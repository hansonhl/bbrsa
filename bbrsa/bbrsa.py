import sys, os
import torch
import logging
import bbrsa

from bbrsa.abstract_classes import BBRSAABC
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.beam import ONMTBeam
from bbrsa.distractors import NextExampleDistractor, BertDistractor
from bbrsa.pragmatics import BasicPragmatics, GrowingAlphaPragmatics, MemoizedListener
from bbrsa.utils import idx_remap, scramble2tgt, chunks
from torchtext.data.batch import Batch as TorchBatch


class ONMTRSAModel(BBRSAABC):
    def __init__(self, s0, pragmatics, distractor, opts, logger=None):
        super().__init__(logger)

        self.s0 = s0
        self.distractor = distractor
        self.pragmatics = pragmatics
        self.gpu = opts.gpu
        self.device = torch.device('cuda') if self.gpu else torch.device('cpu')

    @classmethod
    def from_opts(cls, opts, s0, logger=None):
        if isinstance(s0, string):
            s0_model = ONMTSummarizer(opts, s0, logger)
        elif isinstance(s0, ONMTSummarizer):
            s0_model = s0
        pragmatics = bbrsa.str2prag[opts.pragmatics](opts, logger)
        distractor = bbrsa.str2distr[opts.distractor](opts, logger)
        return cls(s0_model, pragmatics, distractor, logger, opts)

    def _log(self, message, level=None):
        if self.logger is None:
            print(message)
        else:
            level = logging.DEBUG if level is None else level
            self.logger.log(level, message)

    def itos_single(self, idxs, idx_in_batch, src=True):
        """Convert one single array to text"""
        field_name = 'src' if src else 'tgt'
        base_itos = dict(self.s0.translator.fields)[field_name].base_field.vocab.itos
        base_size = len(base_itos)
        tokens = []
        for tok in idxs:
            if src:
                ext_itos = self.s0.data.src_vocabs[idx_in_batch].itos
            else:
                ext_itos = self.s0.data.tgt_vocabs[idx_in_batch].itos
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
        # reordered is false when idx order is scrambled (for s0)
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


    def _run_s0(self, src, opts, dump=None):
        with torch.no_grad():
            beam_size = opts.beam_size
            n_best = opts.n_best
            truncate = opts.truncate
            diverse_beam = opts.diverse_beam

            preds = []
            # all_log_probs = []
            # all_curr_preds = []
            s0 = self.s0
            s0.set_configs(beam_size=beam_size, n_best=n_best)
            s0.init_batch_iterator(src=src, truncate=truncate)
            if dump is not None:
                dump.init(src[0])

            for i, batch in enumerate(s0.data_iter):
                if i % 10 == 9:
                    self._info('S0 batch {}'.format(i+1))
                s0.encode(batch)
                s0.batch_augment(batch, beam_size)
                s0.enc_states_augment(beam_size)
                s0.dec_states_augment(beam_size)

                max_length = s0.max_output_length
                beam_batch_size = batch.batch_size

                # actual batch size, batch may be smaller than max batch size

                beam = ONMTBeam(s0,
                    batch_size=beam_batch_size,
                    beam_size=beam_size,
                    diverse=diverse_beam)

                for step in range(max_length):
                    decoder_input = beam.current_pred
                    beam_batch_offset = beam.batch_offset

                    log_probs, attn = s0.decode(
                        input=decoder_input,
                        batch=batch,
                        step=step,
                        beam_batch_offset=beam_batch_offset)

                    # self._log('attn_shape: {}'.format(attn.shape))
                    # self._log('log_probs.shape: {}'.format(log_probs.shape))
                    # self._log('argmax of log_probs: {}'.format())

                    if diverse_beam is not None:
                        beam.advance(log_probs, attn, opts=opts)
                    else:
                        beam.advance(log_probs, attn)

                    # self._log('step {}, topk_probs {}'.format(step, beam.beam.topk_log_probs))
                    # self._log('step {}, attn {}'.format(step, attn))

                    any_beam_is_finished = beam.any_beam_is_finished
                    if any_beam_is_finished:
                        beam.update_finished()
                        if beam.is_done:
                            break

                    if dump is not None:
                        dump.advance(attn, beam.current_pred)

                    # NOTE: max select_idx is bound by num of remaining
                    # unfinished sentences
                    select_indices = beam.current_origin

                    # self._log('Step {}, curr_preds: {}'.format(step, beam.current_pred.view(-1)))

                    if any_beam_is_finished:
                        s0.enc_states_rearrange(select_indices)
                        s0.batch_rearrange(batch, select_indices)

                    s0.dec_states_rearrange(select_indices)
                # end for step
                # self._log('Final predictions: {}'.format(beam.predictions[0][0]))

                batch_preds = self.itos(beam.predictions, batch)
                preds += batch_preds
            # end for batch
            if dump is not None:
                dump.finalize(preds[0][0])
            return preds, beam
        # end with torch.no_grad()

    def summarize_s0(self, src, opts, dump=None):
        beam_size = opts.beam_size
        n_best = opts.n_best
        truncate = opts.truncate
        diverse_beam = opts.diverse_beam

        self._info('>> Summary with S0')

        preds, _ = self._run_s0(src, opts)

        if dump is not None:
            self._log('dump.tgt len: {}, dump.attns len:{}' \
                .format(len(dump.tgt), len(dump.attns)))
            self._log('src: {}, tgt: {}'.format(dump.src, dump.tgt))
        return preds #, all_log_probs, curr_preds

    def _get_s0_log_probs(self, srcs, summaries, d_factor, truncate):
        """Get S0(u|w) for different w's and u's.

        Args:
            srcs: original articles and their distractors, list of strings
                ``[num_srcs * d_factor,]``
                This should be output from distractor.generate().
            summaries: possible utterances for each original article, 2-D str list
                ``[num_srcs, num_candidates]``
                This should be output from _run_s0(). (num_candidates = beam_size)
            d_factor: total number of world states, i.e. num_distractors + 1
            truncate: whether to truncate src article. # possibly change

        Returns:
            ``tensor([num_srcs, d_factor, num_candidates])``
            Matrix of log probabilities, for S0(u|w), last dimension normalized
        """
        # reshape srcs into (num_srcs, d_factor)
        src_chunks = list(chunks(srcs, d_factor))
        srcs, tgts = [], []
        num_candidates = -1
        for ws, us in zip(src_chunks, summaries):
            # TODO: for cnndm, need to add tags to tgts
            if num_candidates == -1:
                num_candidates = len(us)
            for w in ws:
                # for each w in {art, distractors} set, get a batch
                srcs += [w] * len(us)
            tgts += us * len(ws)
        # now srcs, tgts have shape [num_srcs * d_factor * num_candidates,]

        s0 = self.s0
        s0.init_batch_iterator(src=srcs, tgt=tgts, truncate=truncate,
            batch_size=num_candidates)
        # TODO: probably have a more flexible batch size?
        pad_token = s0.pad_token

        all_sent_probs = []

        with torch.no_grad():
            for batch in s0.data_iter:
                s0.encode(batch)
                curr_tgt = batch.tgt[1:, :, :]
                log_probs, _ = s0.decode(
                    input=curr_tgt,
                    batch=batch,
                    step=None,
                    beam_batch_offset=None)

                reorder_idxs = idx_remap(batch.indices)
                reordered_probs = log_probs.index_select(1, reorder_idxs)
                reordered_tgts = curr_tgt.index_select(1, reorder_idxs)
                masks = (reordered_tgts != pad_token)

                sent_probs = torch.zeros(num_candidates, device=self.device)
                range_idxs = torch.arange(reordered_probs.shape[0],
                                          device=self.device)
                for i in range(num_candidates):
                    mask = masks[:, i, 0]
                    rng = range_idxs[mask]
                    word_idxs = reordered_tgts[:, i, 0][mask]
                    word_probs = reordered_probs[rng, i, word_idxs]

                    # NORMALIZATION: divide by length
                    sent_probs[i] = word_probs.sum(dim=0) / ((word_probs.shape[0]) ** 1.2)

                sent_probs = sent_probs - torch.logsumexp(sent_probs, dim=0,
                                                          keepdim=True)
                all_sent_probs.append(sent_probs)

        all_sent_probs = torch.stack(all_sent_probs) \
                              .view(-1, d_factor, num_candidates)
        # now all_sent_probs has shape [num_srcs, d_factor, num_candidates]
        return all_sent_probs


    def global_s1(self, src, opts):
        """Summarize source text using global RSA

        Args:
            src: articles to summarize, list of strings

            ### below are in opts
            beam_size: beam size during beam search of s0, equivalent to number
                of candidate utterances for each article, used as the set of
                possible utterances for the pragmatic l1 and s1, default 5.
            n_best: number of articles returned at the end, ranked by s1's
                probabilities, default 1.
            truncate: length of src article that is truncated when read by s0,
                default None, which does not truncate it
            diverse_beam: name of strategy for diverse beam search in s0
        """
        beam_size = opts.beam_size
        n_best = opts.n_best
        truncate = opts.truncate
        diverse_beam = opts.diverse_beam
        shard_size = opts.shard_size

        assert (not isinstance(self.pragmatics, GrowingAlphaPragmatics)) \
            and (not isinstance(self.pragmatics, MemoizedListener)), \
            'Must use BasicPragmaitcs for global S1!'
        assert (n_best <= beam_size), 'n_best must be less than beam size!'

        self._info('>> Summary with global S1')

        candidates, _ = self._run_s0(src, opts, dump=None)
        # candidates has shape [num_srcs, num_candidates]

        d_factor = self.distractor.d_factor
        srcs, batch_size = self.distractor.generate(src, opts)

        s0_probs = self._get_s0_log_probs(srcs, candidates, d_factor, truncate)
        s1_probs = self.pragmatics.inference(s0_probs, opts)
        # both have shape [num_srcs, d_factor, num_candidates]

        tgt_probs = s1_probs[:, 0, :]
        n_best_probs, n_best_idxs = tgt_probs.topk(n_best, dim=1)

        res = [[candidates[i][j] for j in idxs] for i, idxs in enumerate(n_best_idxs)]
        return res


    def incremental_s1(self, src, opts):
        """Summarize source text using incremental RSA."""
        beam_size = opts.beam_size
        n_best = opts.n_best
        truncate = opts.truncate
        diverse_beam = opts.diverse_beam

        self._info('>> Summary with incremental s1')
        assert self.distractor is not None, 'Must specify distractor!'
        assert self.pragmatics is not None, 'Must specify pragmatics'
        preds = []
        self.pragmatics.clear_mem()

        with torch.no_grad():
            s0 = self.s0

            # for BertDistractor, d_factor may be variable
            if isinstance(self.distractor, BertDistractor):
                d_factor = opts.bert_distr_d_factor
            else:
                d_factor = self.distractor.d_factor

            s0.set_configs(beam_size=beam_size, n_best=n_best)
            self._debug('Generating distractors')
            src, batch_size = self.distractor.generate(src, opts)

            self._debug('Initializing batch iterator')
            s0.init_batch_iterator(
                src=src,
                batch_size=batch_size,
                truncate=truncate)
            self._debug('Finished initializing batch iterator')

            for i, batch in enumerate(s0.data_iter):
                if i % 10 == 9:
                    self._info('Incr. S1 batch {}'.format(i+1))
                scramble_idxs = batch.indices % batch_size

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
                # self._log('batch.batch_size is {}'.format(batch.batch_size), logging.INFO)

                beam = ONMTBeam(s0,
                    batch_size=beam_batch_size,
                    beam_size=beam_size,
                    n_best=n_best,
                    diverse=diverse_beam,
                    distractor=self.distractor,
                    scramble_idxs=batch.indices)

                batch_offset = list(range(len(beam.batch_offset) * d_factor))

                for step in range(max_length):
                    # self._log('----step {}----'.format(step))
                    decoder_input = _reshape_beam2dec(beam.current_pred,
                        beam_size, d_factor, scramble_idxs)

                    log_probs, attn = s0.decode(
                        input=decoder_input,
                        batch=batch,
                        step=step,
                        beam_batch_offset=batch_offset)
                    # log_probs.shape = [B*d*b, V]

                    s0_log_probs = _reshape_dec2prag(log_probs, beam_size,
                        d_factor, scramble_idxs)
                    # s0_log_probs.shape = [B*b, d, V] (B*b ordered)

                    # s1_log_probs = s0_log_probs # for debug
                    if isinstance(self.pragmatics, GrowingAlphaPragmatics):
                        s1_log_probs = self.pragmatics.inference(
                            s0_log_probs, opts, step)
                    elif isinstance(self.pragmatics, MemoizedListener):
                        s1_log_probs = self.pragmatics.inference(
                            s0_log_probs, opts,
                            beam.current_origin,
                            beam.current_pred.view(-1))
                    else:
                        s1_log_probs = self.pragmatics.inference(
                            s0_log_probs, opts)

                    # s1_log_probs.shape = [B*b, d, V]

                    beam_log_probs = _reshape_prag2beam(s1_log_probs, beam_size,
                        d_factor, scramble_idxs)
                    # beam_log_probs.shape = [B*b, V]

                    attn = _reshape_attn(attn, beam_size, d_factor, scramble_idxs)
                    # attn.shape = [1, B*b, L]

                    if diverse_beam is not None:
                        beam.advance(beam_log_probs, attn, opts=opts)
                    else:
                        beam.advance(beam_log_probs, attn)

                    any_beam_is_finished = beam.any_beam_is_finished
                    if any_beam_is_finished:
                        beam.update_finished()
                        if beam.is_done:
                            break

                    select_indices, scramble_idxs, batch_offset = \
                        _reshape_select_idxs_and_rescramble(
                            beam.current_origin, beam_size, d_factor,
                            scramble_idxs, batch_offset, step=step)

                    if any_beam_is_finished:
                        s0.enc_states_rearrange(select_indices)
                        s0.batch_rearrange(batch, select_indices)
                    s0.dec_states_rearrange(select_indices)
                # end for step

                batch_preds = self.itos(beam.predictions, batch, reordered=True,
                                        d_factor=d_factor)
                preds += batch_preds
            # end for batch
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


def _reshape_select_idxs_and_rescramble(input, beam_size, d_factor,
    scramble_idxs, batch_offset, step=None):
    """Reshape select indices from beam for rearranging states"""
    res = input.view(-1, beam_size)    # [2,1]
    B = res.shape[0] # B = batch_size

    if B * d_factor == scramble_idxs.shape[0]:
        B_range = torch.arange(B, device=input.device)
        offset = (B_range * beam_size).view(-1, 1)
        res -= offset
        res = res.repeat_interleave(d_factor, dim=0) \
                 .index_select(0, scramble_idxs)
        offset_range = torch.arange(res.shape[0], device=input.device)
        offset = (offset_range * beam_size).view(-1, 1)
        res += offset
        res = res.view(-1)

        return res, scramble_idxs, batch_offset
    else:
        # get indices of remaining targets in original, unscrambled order
        # ***assume rem_tgts is in ascending order****

        # e.g. d_factor = 2, beam_size = 10
        #      scramble_idxs = [4,2,0,5,1,3]
        #      res = [[0,2,4,5,6,7,8,7,8,9],[21,24,22,25,26,24,28,29,29,22]]
        #      batch_offset = [0,1,2,3,4,5]
        rem_tgts = res[:, 0] // beam_size # remaining targets [0, 2]
        d_factor_range = torch.arange(d_factor, device=input.device)
        rem_ts_and_ds = ((rem_tgts.repeat_interleave(d_factor, dim=0)
            * d_factor).view(-1, d_factor) + d_factor_range).view(-1) #[0,1,4,5]

        # get new scramble idxs given old scramble idxs and remaining targets
        prev_scramble_len = scramble_idxs.shape[0]
        rem_reorder_map = -1 * torch.ones(prev_scramble_len, dtype=torch.long,
                                          device=input.device)
        rem_reorder_map[rem_ts_and_ds] = torch.arange(rem_ts_and_ds.shape[0],
                                                      device=input.device)
                                                                #[0,1,-1,-1,2,3]
        scrmbd_reorder_map = rem_reorder_map[scramble_idxs]     #[2,-1,0,3,1,-1]
        scrmbd_rem_mask = rem_reorder_map[scramble_idxs] != -1  #[1, 0,1,1,1, 0]
        new_scramble_idxs = scrmbd_reorder_map[scrmbd_rem_mask]       #[2,0,3,1]
        rem_idxs_in_batch = scrmbd_rem_mask.nonzero().view(-1)        #[0,2,3,4]

        # e.g.     res = [[0,2,4,5,6,7,8,7,8,9],[21,24,22,25,26,24,28,29,29,22]]
        offset = (rem_tgts * beam_size).view(-1, 1)                # [[0], [20]]
        res -= offset             #[[0,2,4,5,6,7,8,7,8,9],[1,4,2,5,6,4,8,9,9,2]]
        res = res.repeat_interleave(d_factor, dim=0) \
                 .index_select(0, new_scramble_idxs)
                                          #[[0,2...],[0,2...],[1,4...],[1,4...]]
                                          #[[1,4...],[0,2...],[1,4...],[0,2...]]
        offset = (rem_idxs_in_batch * beam_size).view(-1, 1)
                                                #[0,2,3,4]->[[0],[20],[30],[40]]
        res += offset
        res = res.view(-1)

        batch_offset = [batch_offset[i] for i in rem_idxs_in_batch]   #[0,2,3,4]

        return res, new_scramble_idxs, batch_offset
