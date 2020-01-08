import sys, os
import torch
import bbrsa

from bbrsa.distractors import BertDistractor, NextExampleDistractor, NextNDistractor, AsIsDistractor
from bbrsa.bbrsa import ONMTRSAModel
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.pragmatics import BasicPragmatics, GrowingAlphaPragmatics, MemoizedListener
from bbrsa.configopts import ConfigOpts

from bbrsa.utils import opts_to_list

# default_opts_dict = {
#     'random_seed': 3939, # used for bert distractor generation
#     'dummy_src': 'data/giga_small_input.txt', # need this for opennmt model to be loaded
#     'batch_size': 20,
#     'truncate': -1,
#     'add_tags': False,
#     'n_best': 1,
#     'beam_size': 5,
#     'model_verbose': False,
#     'diverse_beam': (None, [None, 'rank']),
#     'diverse_beam_rank_lambda': 1.5,
#     'coverage_penalty': (None, [None, 'summary', 'wu']),
#     'coverage_penalty_beta': 0, # recommend 5 for cnndm
#     'length_penalty': (None, [None, 'wu', 'avg']),
#     'length_penalty_alpha': 0, # recommend 9 for cnndm
#     'block_ngram_repeat': 0,
#     'min_length': 0,
#     'stepwise_penalty': False,
#     'prag_alpha': 2.,
#     'prag_alpha_grow_steps': 5,
#     'nextn_distr_N': 3,
#     'bert_distr_d_factor': 2,
#     'bert_distr_method': ('unmasked_surprisal', ['unmasked_surprisal', 'layer0_attn']),
#     'bert_distr_salient_topk': 15,
#     'bert_distr_mask_topk': 5,
#     'bert_distr_repl_search_topk': 5,
#     'bert_distr_ensure_different': True,
# }

base_opts = bbrsa.DEFAULT_OPTS
clone = base_opts.clone()
clone.beam_size = 10000
clone.distractor = 'bert'

for k, v in clone:
    print(k, v)
print(base_opts.beam_size)
print(base_opts.distractor)

# s0 = ONMTSummarizer(base_opts, model_ckpt_path='/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_nocov_step_59156_valacc48.57_ppl15.51.pt')
#
# pragmatics = BasicPragmatics()
#
# distr = NextNDistractor()
#
# model = ONMTRSAModel(s0, pragmatics, distr)
#
# src = ['police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a french nuclear research and supply vessel in antarctica , a spokesman for the protesters said .',
#     'police arrested five climate change activists monday after they sought to disrupt loading of a french oil research and supply ship in antarctica , a spokesman for the protesters said .',
#     'police arrested seven anti-corruption protesters tuesday after they sought to disrupt loading of a french luxurious tourist boat in the maldives , a spokesman for the protesters said .',
#     'military arrested five anti-nuclear protesters thursday after they sought to disrupt loading of an american nuclear research and supply vessel , a spokesman for the protesters said .']
# # print(distr.generate(src, repl_search_topk=10, ensure_different=True))
#
# # print('\nNormal beam search:')

newconfigs = {
    'diverse_beam': 'rank',
    'beam_size': 10,
    'prag_alpha': 15.,
    'n_best': 3,
    'nextn_distr_N': 2
}

base_opts.set_values(newconfigs)

new_base_opts_dict = base_opts.default_dict()
new_base_opts = ConfigOpts(new_base_opts_dict)
new_base_opts.set_as_default(newconfigs)
base_opts.reset()
# print('base_opts')
# print(base_opts)
# print('new_base_opts')
# print(new_base_opts)

# print(model.incremental_s1(src, base_opts))

# print('\nSummarizing with diverse beam search:')
# print(model.incremental_s1(src, n_best=10, diverse_beam='rank', beam_size=10))

# print(model.summarize_s0(src, n_best=10, diverse_beam='rank', beam_size=10))
#
# print(model.global_s1(src, base_opts))
#
# print('\n-----Trying as-is distractor-----')
#
# distr2 = AsIsDistractor(d_factor=distr.d_factor)
# model2 = ONMTRSAModel(s0, pragmatics, distr2)
# src2, _ = distr.generate(src, new_base_opts)

# print(model2.incremental_s1(src2, new_base_opts))
