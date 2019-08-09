from bbrsa.configopts import ConfigOpts

from bbrsa.distractors import *
from bbrsa.pragmatics import *

str2distr = {
    'next_1': NextExampleDistractor,
    'identical': IdenticalDistractor,
    'next_n': NextNDistractor,
    'bert': BertDistractor
}
distr_list = list(str2distr.keys())

str2prag = {
    'basic': BasicPragmatics,
    'growing_alpha': GrowingAlphaPragmatics,
    'memoized_l1': MemoizedListener
}
prag_list = list(str2prag.keys())

mode_list = ['incr_s1', 's0']

ONMT_DIR = '../myOpenNMT'

_s0_fixed_opts = ['s0_model_path', 'batch_size', 'beam_size', 'dummy_src',
                  'coverage_penalty', 'coverage_penalty_beta', 'length_penalty',
                  'length_penalty_alpha', 'block_ngram_repeat', 'min_length',
                  'stepwise_penalty', 'gpu']

_default_opts_dict = {
    'gpu': True,
    'random_seed': 39831, # used for bert distractor generation
    'dummy_src': 'data/giga_small_input.txt', # need this for opennmt model to be loaded

    'distractor': ('next_1', distr_list), # use bert
    'pragmatics': ('basic', prag_list),
    'mode': ('incr_s1', mode_list),

    # these are fixed
    'batch_size': 32,
    'shard_size': 1000,
    'truncate': -1,
    'add_tags': False,

    # fix these for incremetal. all of these are relevant to both s0 and s1
    'n_best': 1,
    'beam_size': 10,
    'model_verbose': False,
    'diverse_beam': (None, [None, 'rank']),
    'diverse_beam_rank_lambda': 1.5,

    # the following are used in init setting and are fixed
    'coverage_penalty': (None, [None, 'summary', 'wu']),
    'coverage_penalty_beta': 0, # recommend 5 for cnndm
    'length_penalty': (None, [None, 'wu', 'avg']),
    'length_penalty_alpha': 0, # recommend 9 for cnndm
    'block_ngram_repeat': 0,
    'min_length': 0,
    'stepwise_penalty': False,

    # these can be changed
    'prag_alpha': 2.,        # change, relevant to s1 only
    'prag_alpha_grow_steps': 5, # fix

    'nextn_distr_N': 3,
    'bert_distr_d_factor': 2, # change, relevant to s1 only
    'bert_distr_method': ('unmasked_surprisal', ['unmasked_surprisal', 'layer0_attn']),
    'bert_distr_salient_topk': 15, #fix
    'bert_distr_mask_topk': 5, #fix
    'bert_distr_repl_search_top': 0, #change, relevant to s1 only
    'bert_distr_repl_search_bottom': 5, #change, relevant to s1 only
    'bert_distr_ensure_different': True #fix
}

DEFAULT_OPTS = ConfigOpts(_default_opts_dict)
