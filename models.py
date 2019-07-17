import torch
from abstract_classes import LiteralSpeaker

DEFAULT_CONFIG_DIR = 'summary_inference2.yml'

class ONMTSummarizer(LiteralSpeaker):
    def __init__(self, config_dir=DEFAULT_CONFIG_DIR):

        print('Configuring summary model...')

        from onmt.utils.parse import ArgumentParser
        import onmt.opts as opts
        from onmt.translate.translator import build_translator
        from onmt.utils.misc import tile


        parser = ArgumentParser(default_config_files=[config_dir])
        opts.translate_opts(parser)
        self.opt = parser.parse_args()
        ArgumentParser.validate_translate_opts(self.opt)
        self.translator = build_translator(self.opt, report_score=True)
        print('Finished configuration.\n')

        # for batch
        self.data, self.data_iter = None, None
        self.batch_size = self.opt.batch_size

        # Encoder representations
        self.src, self.enc_states, self.memory_bank, self.src_lengths = \
            None, None, None, None
        self.memory_lengths = None
        self.mb_device = None
        self.src_map = None

        # for augmentation
        self.tile = tile

    def init_batch_iterator(self, src, batch_size=None):
        """ Initiates batch iterator. The iterator also maps raw text to idx's.

        Args:
            src: a python list of raw input text to be summarized
        """
        import onmt.inputters as inputters
        from onmt.translate import TranslationBuilder

        T = self.translator
        self.data = inputters.Dataset(
            T.fields,
            readers=[T.src_reader],
            data=[("src", src)],
            dirs=[None],
            sort_key=inputters.str2sortkey[T.data_type],
            filter_pred=T._filter_pred
        )

        if batch_size is not None:
            self.batch_size = batch_size

        self.data_iter = inputters.OrderedIterator(
            dataset=self.data,
            device=T._dev,
            batch_size=self.batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        # build translator
        self.xlation_builder = TranslationBuilder(
            self.data, T.fields, T.n_best, T.replace_unk, None,
            T.phrase_table)

    def encode(self, batch):
        T = self.translator
        src, enc_states, memory_bank, src_lengths = T._run_encoder(batch)

        T.model.decoder.init_state(src, memory_bank, enc_states)

        self.src, self.enc_states, self.memory_bank, self.src_lengths = \
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
        self.memory_lengths = self.tile(self.src_lengths, beam_size)

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

    def decode(self, input, batch, step, beam_batch_offset=None):
        T = self.translator
        log_probs, attn = T._decode_and_generate(
            input,
            self.memory_bank,
            batch,
            self.data.src_vocabs,
            memory_lengths=self.memory_lengths,
            src_map=self.src_map,
            step=step,
            batch_offset=beam_batch_offset)
        # normalize
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

    def set_configs(self, beam_size=1, n_best=1):
        self.translator.beam_size = beam_size
        self.translator.n_best = n_best
