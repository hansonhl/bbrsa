from abc import ABC, abstractmethod, abstractproperty

class LiteralSpeaker(ABC):
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

class Beam(ABC):
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
