from bbrsa.configopts import ConfigOpts

ONMT_DIR = '../myOpenNMT'

_default_opts_dict = {
    'gpu': False,
    'random_seed': 3939, # used for bert distractor generation
    'dummy_src': 'data/giga_small_input.txt', # need this for opennmt model to be loaded
    'batch_size': 20,
    'truncate': -1,
    'add_tags': False,
    'n_best': 1,
    'beam_size': 5,
    'model_verbose': False,
    'diverse_beam': (None, [None, 'rank']),
    'diverse_beam_rank_lambda': 1.5,
    'coverage_penalty': (None, [None, 'summary', 'wu']),
    'coverage_penalty_beta': 0, # recommend 5 for cnndm
    'length_penalty': (None, [None, 'wu', 'avg']),
    'length_penalty_alpha': 0, # recommend 9 for cnndm
    'block_ngram_repeat': 0,
    'min_length': 0,
    'stepwise_penalty': False,
    'prag_alpha': 2.,
    'prag_alpha_grow_steps': 5,
    'nextn_distr_N': 3,
    'bert_distr_d_factor': 2,
    'bert_distr_method': ('unmasked_surprisal', ['unmasked_surprisal', 'layer0_attn']),
    'bert_distr_salient_topk': 15,
    'bert_distr_mask_topk': 5,
    'bert_distr_repl_search_topk': 5,
    'bert_distr_ensure_different': True,
    'eval_shard_size': 512
}

DEFAULT_OPTS = ConfigOpts(_default_opts_dict)
