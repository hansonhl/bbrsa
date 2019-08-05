import torch
import logging
from bbrsa.abstract_classes import LiteralSpeaker
from bbrsa.utils import onmt_translator_builder

import onmt.inputters as inputters
from onmt.utils.misc import tile
from onmt.translate import TranslationBuilder

INFO = logging.INFO
DEBUG = logging.DEBUG

class ONMTSummarizer(LiteralSpeaker):
    def __init__(self, my_opts, model_ckpt_path, logger=None):
        """build summarizer from config"""
        super().__init__(logger)

        self.translator = onmt_translator_builder(model_ckpt_path, my_opts, logger)

        # for batch
        self.data, self.data_iter = None, None
        self.default_batch_size = my_opts.batch_size

        # Encoder representations
        self.src, self.enc_states, self.memory_bank, self.memory_lengths = \
            None, None, None, None
        self.memory_lengths = None
        self.mb_device = None
        self.src_map = None

        # for augmentation
        self.tile = tile

    def init_batch_iterator(self, src, tgt=None, batch_size=None, truncate=None):
        """ Initiates batch iterator. The iterator maps raw text to idx's.

        Args:
            src: a python list of raw input text to be summarized
        """
        if truncate != -1:
            src = _truncate(src, truncate)
            self._log('Truncated src to length {}'.format(truncate), logging.INFO)
            self._log('len of first element is {}'.format(len(src[0].split())))

        batch_size = self.default_batch_size if batch_size is None else batch_size

        T = self.translator
        self.data = inputters.Dataset(
            T.fields,
            readers=[T.src_reader, T.tgt_reader] if tgt else [T.src_reader],
            data=[("src", src), ("tgt", tgt)] if tgt else [("src", src)],
            dirs=[None, None] if tgt else [None],
            sort_key=inputters.str2sortkey[T.data_type],
            filter_pred=T._filter_pred
        )

        self.data_iter = inputters.OrderedIterator(
            dataset=self.data,
            device=T._dev,
            batch_size=batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

    def encode(self, batch):
        T = self.translator
        src, enc_states, memory_bank, src_lengths = T._run_encoder(batch)

        T.model.decoder.init_state(src, memory_bank, enc_states)

        self.src, self.enc_states, self.memory_bank, self.memory_lengths = \
            src, enc_states, memory_bank, src_lengths

    def batch_augment(self, batch, beam_size):
        T = self.translator
        self.src_map = (self.tile(batch.src_map, beam_size, dim=1)
                   if T.copy_attn else None)


    def enc_states_augment(self, beam_size):
        T = self.translator
        if isinstance(self.memory_bank, tuple):
            self.memory_bank = tuple(self.tile(x, beam_size, dim=1) \
                for x in self.memory_bank)
            self.mb_device = self.memory_bank[0].device
        else:
            self.memory_bank = self.tile(self.memory_bank, beam_size, dim=1)
            self.mb_device = self.memory_bank.device
        self.memory_lengths = self.tile(self.memory_lengths, beam_size)

    def dec_states_augment(self, beam_size):
        T = self.translator
        T.model.decoder.map_state(
            lambda state, dim: self.tile(state, beam_size, dim=dim))

    @property
    def min_output_length(self):
        return self.translator.min_length

    @property
    def max_output_length(self):
        return self.translator.max_length

    @property
    def pad_token(self):
        str_pad_token = self.translator.fields['tgt'].base_field.pad_token
        return self.translator.fields['tgt'].base_field.vocab.stoi[str_pad_token]

    def decode(self, input, batch, step=None, beam_batch_offset=None):
        T = self.translator

        log_probs, attn = T._decode_and_generate(
            input,
            self.memory_bank,
            batch,
            self.data.src_vocabs,
            memory_lengths=self.memory_lengths,
            src_map=self.src_map if step is not None else batch.src_map,
            step=step,
            batch_offset=beam_batch_offset)
        # normalize
        if step is not None:
            lse = torch.logsumexp(log_probs, dim=1, keepdim=True)
            log_probs = log_probs - lse

        return log_probs, attn


    def batch_rearrange(self, batch, select_indices):
        if self.src_map is not None:
            self.src_map = self.src_map.index_select(1, select_indices)

    def enc_states_rearrange(self, select_indices):
        if isinstance(self.memory_bank, tuple):
            self.memory_bank = tuple(x.index_select(1, select_indices)
                                for x in self.memory_bank)
        else:
            self.memory_bank = self.memory_bank.index_select(1, select_indices)

        self.memory_lengths = self.memory_lengths.index_select(0, select_indices)

    def dec_states_rearrange(self, select_indices):
        T = self.translator
        T.model.decoder.map_state(
            lambda state, dim: state.index_select(dim, select_indices))

    def set_configs(self, beam_size, n_best):
        self.translator.beam_size = beam_size
        self.translator.n_best = n_best

def _truncate(src, len):
    """truncate each string in src to a given length

    Assume each element in src is tokenized, seperated by spaces
    """
    return [' '.join(l.split()[:len]) for l in src]
