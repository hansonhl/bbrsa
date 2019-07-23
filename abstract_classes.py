from abc import ABC, abstractmethod, abstractproperty

class BBRSAABC(ABC):
    def __init__(self, logger=None):
        self.logger = logger

    def _log(self, message, level=None):
        if self.logger is None:
            print(message)
        else:
            level = logging.DEBUG if level is None else level
            self.logger.log(level, message)


class LiteralSpeaker(BBRSAABC):
    def __init__(self, logger=None):
        super().__init__(logger)

    @abstractmethod
    def init_batch_iterator(self, src):
        pass

    @abstractmethod
    def encode(self, batch):
        pass

    @abstractmethod
    def decode(self, input, batch, step):
        pass

    @abstractmethod
    def batch_augment(self, batch, beam_size):
        pass

    @abstractmethod
    def enc_states_augment(self, beam_size):
        pass

    @abstractmethod
    def dec_states_augment(self, beam_size):
        pass

    @abstractmethod
    def batch_rearrange(self, batch, select_indices):
        pass

    @abstractmethod
    def enc_states_rearrange(self, select_indices):
        pass

    @abstractmethod
    def dec_states_rearrange(self, select_indices):
        pass

    @abstractproperty
    def min_output_length(self):
        pass

    @abstractproperty
    def max_output_length(self):
        pass

    @abstractmethod
    def set_configs(self, **kwargs):
        pass

class Beam(BBRSAABC):
    def __init__(self, logger=None):
        super().__init__(logger)

    @abstractmethod
    def advance(self, log_probs, attn):
        pass

    @abstractmethod
    def update_finished(self):
        pass

    @abstractproperty
    def current_pred(self):
        pass

    @abstractproperty
    def any_beam_is_finished(self):
        pass

    @abstractproperty
    def is_done(self):
        pass

    @abstractproperty
    def current_origin(self):
        pass

    @abstractproperty
    def predictions(self):
        pass

    @abstractproperty
    def batch_offset(self):
        pass

class BatchDistractor(BBRSAABC):
    def __init__(self, batch_size, logger=None):
        super().__init__(logger)
        self.orig_batch_size = batch_size

    @abstractmethod
    def generate(self, src):
        return src

    # I make this an abstract property so that it is mandatory
    @abstractproperty
    def d_factor(self):
        return 1

    @property
    def new_batch_size(self):
        return self.d_factor * self.orig_batch_size

class Pragmatics(BBRSAABC):
    def __init__(self, logger=None):
        super().__init__(logger)

    @abstractproperty
    def inference(self, probs):
        return probs

    def clear_mem(self):
        pass
