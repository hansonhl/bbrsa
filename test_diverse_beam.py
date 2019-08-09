import sys, os
import torch
import bbrsa

from bbrsa.distractors import BertDistractor, NextExampleDistractor, AsIsDistractor
from bbrsa.bbrsa import ONMTRSAModel
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.pragmatics import BasicPragmatics, GrowingAlphaPragmatics

opts = bbrsa.DEFAULT_OPTS

s0 = ONMTSummarizer(opts, '/home/hansonlu/myOpenNMT/models/gigaword_copy_acc_51.78_ppl_11.71_e20.pt')
pragmatics = BasicPragmatics(opts)
distr = BertDistractor(opts)
model = ONMTRSAModel(s0, pragmatics, distr, opts)

src = ['police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a french nuclear research and supply vessel in antarctica , a spokesman for the protesters said .']
    # 'military arrested five anti-nuclear protesters thursday after they sought to disrupt loading of an american nuclear research and supply vessel , a spokesman for the protesters said .']
# print(distr.generate(src, repl_search_topk=10, ensure_different=True))

# print('\nNormal beam search:')
# print(model.incremental_s1(src, n_best=10, beam_size=10))

print('\nSummarizing with diverse beam search:')
# print(model.incremental_s1(src, n_best=10, diverse_beam='rank', beam_size=10))

# print(model.summarize_s0(src, n_best=10, diverse_beam='rank', beam_size=10))

# opts.set_values({'prag_alpha': 15., 'beam_size': 10, 'n_best': 3, 'diverse_beam': None})
#
# print(model.global_s1(src, opts))

opts.set_values({'prag_alpha': 2., 'beam_size': 10, 'n_best': 1,
                 'diverse_beam': 'rank',
                 'bert_distr_repl_search_top': 0,
                 'bert_distr_repl_search_bottom': 10,
                 'bert_distr_d_factor': 5})

"""
new result
['police arrested five anti-nuclear protesters thursday after they
sought to disrupt loading of a french nuclear research and supply vessel
in antarctica , a spokesman for the protesters said .',
'france arrested five anti-nuclear protesters . after they responded to stop
 loading of a french nuclear research and supply station in antarctica ,
  a spokesman for the protesters said .']

['police arrested five anti-nuclear protesters thursday after they
sought to disrupt loading of a french nuclear research and supply vessel
in antarctica , a spokesman for the protesters said .',
'france arrested five anti-nuclear protesters . after they responded to stop
loading of a french nuclear research and supply station in antarctica ,
a spokesman for the protesters said .']

['police arrested five anti-nuclear protesters thursday after they
sought to disrupt loading of a french nuclear research and supply vessel
in antarctica , a spokesman for the protesters said .',
'france arrested five anti-nuclear protesters . after they responded to stop
loading of a french nuclear research and supply station in antarctica ,
a spokesman for the protesters said .']


"""

print(model.incremental_s1(src, opts))

print('\n-----Trying as-is distractor-----')

# distr2 = AsIsDistractor(d_factor=distr.d_factor)
# model2 = ONMTRSAModel(s0, pragmatics, distr2, opts)
# src2, _ = distr.generate(src)
# print('src2', src2)
# print(model2.incremental_s1(src2, opts))
