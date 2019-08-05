import torch
import numpy as np
import logging
from bbrsa.abstract_classes import Pragmatics

class BasicPragmatics(Pragmatics):
    def __init__(self, logger=None):
        super().__init__(logger)

    def l1(self, log_probs, opts, *args):
        """Pragmatic listener based on top of s0"""
        normalized = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True)
        normalized[torch.isnan(normalized)] = -np.log(normalized.shape[1])

        return normalized

    def s1(self, s0_log_probs, l1_log_probs, opts, *args):
        """Pragmatic speaker"""
        adjusted = opts.prag_alpha * l1_log_probs
        isnan_mask = torch.isnan(adjusted)
        adjusted[isnan_mask] = float('-inf')

        log_probs = s0_log_probs + adjusted

        lse = torch.logsumexp(log_probs, dim=2, keepdim=True)
        normalized = log_probs - lse
        return normalized

    def inference(self, s0_log_probs, opts, *args):
        """Do pragmatic inference based on log probs generated by s0
        Args:
            s0_log_probs: `tensor([*, num_world_states, vocab_size])`
                `num_word_states` is also called `d_factor`
            rem_idxs: only used for memoized listener

        Returns:
            `tensor([*, num_world_states, vocab_size])`
        """
        s0_log_probs = s0_log_probs.type(torch.double)
        l1_log_probs = self.l1(s0_log_probs, opts, *args)
        res = self.s1(s0_log_probs, l1_log_probs, opts, *args).type(torch.float)
        return res

class GrowingAlphaPragmatics(BasicPragmatics):
    def __init__(self, logger=None):
        super().__init__(logger)

    def s1(self, s0_log_probs, l1_log_probs, opts, step):
        """Pragmatic speaker, alpha grows incrementally until it reaches self.alpha"""
        grow_steps = opts.prag_alpha_grow_steps
        if step is not None:
            alpha = min(step, grow_steps) / grow_steps * opts.prag_alpha # grows in  steps

        adjusted = alpha * l1_log_probs
        isnan_mask = torch.isnan(adjusted)
        adjusted[isnan_mask] = float('-inf')

        log_probs = s0_log_probs + adjusted

        lse = torch.logsumexp(log_probs, dim=2, keepdim=True)
        normalized = log_probs - lse
        return normalized

class MemoizedListener(BasicPragmatics):
    # TODO: Need to fix - l1_prev_prob should have dimension of [B*b, d]
    #    instead of [B*b, d, V]: need to use the output word that's chosen in
    #    last timestep. For this I need to add one more argument in l1.
    #    probably use *args in base definition?
    def __init__(self, logger=None):
        super().__init__(logger)
        self.l1_prev_prob = None

    def clear_mem(self):
        del self.l1_prev_prob
        self.l1_prev_prob = None

    def l1(self, log_probs, opts, rem_idxs, curr_pred):
        """Pragmatic listener that uses prob of previous time step as prior
        Args:
            log_probs: tensor((batch_size * beam_size), d_factor, vocab_size)
            rem_idxs: indices of partial sentences chosen in latest time step,
                tensor((batch_size * beam_size), )
            curr_pred: predicted words for each partial sentence in latest time
                step, tensor((batch_size * beam_size), )
        """
        num_world_states = log_probs.shape[1]

        if self.l1_prev_prob is None:
            self.l1_prev_prob = torch.zeros(log_probs.shape, dtype=torch.double)
            self.l1_prev_prob -= np.log(num_world_states)
            priors = torch.zeros(log_probs.shape[:-1], dtype=torch.double)
            priors -= np.log(num_world_states)
            priors = priors.unsqueeze(2)

        if rem_idxs is not None:
            priors = self.l1_prev_prob.index_select(0, rem_idxs)      # [B*b', d, V]
            priors = priors[torch.arange(priors.shape[0]), :, curr_pred] # [B*b', d]
            priors = priors - torch.logsumexp(priors, dim=1, keepdim=True)
            priors = priors.unsqueeze(2)                           #[B*b', d, 1]

        new_log_probs = log_probs + priors
        normalized = new_log_probs - torch.logsumexp(new_log_probs, dim=1, keepdim=True)
        normalized[torch.isnan(normalized)] = -np.log(num_world_states)
        self.l1_prev_prob = normalized

        return normalized
