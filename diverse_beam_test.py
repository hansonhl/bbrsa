import sys, os
import torch
import bbrsa
sys.path.append(os.path.abspath(bbrsa.ONMT_DIR))

from bbrsa.distractors import BertDistractor, NextExampleDistractor, AsIsDistractor
from bbrsa.bbrsa import ONMTSummaryRSA
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.pragmatics import BasicPragmatics, GrowingAlphaPragmatics

s0 = ONMTSummarizer(config_path='onmt_configs/giga.yml')
pragmatics = BasicPragmatics(alpha=3)
distr = BertDistractor(batch_size=s0.default_batch_size)
model = ONMTSummaryRSA(s0, pragmatics, distr)

src = ['police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a french nuclear research and supply vessel in antarctica , a spokesman for the protesters said .']
    # 'military arrested five anti-nuclear protesters thursday after they sought to disrupt loading of an american nuclear research and supply vessel , a spokesman for the protesters said .']
# print(distr.generate(src, repl_search_topk=10, ensure_different=True))

# print('\nNormal beam search:')
# print(model.incremental_s1(src, n_best=10, beam_size=10))

print('\nSummarizing with diverse beam search:')
# print(model.incremental_s1(src, n_best=10, diverse_beam='rank', beam_size=10))

# print(model.summarize_s0(src, n_best=10, diverse_beam='rank', beam_size=10))

print(model.global_s1(src, beam_size=10, n_best=3, diverse_beam=None))

model.set_alpha(3)

print(model.incremental_s1(src, beam_size=10, n_best=1, diverse_beam='rank'))

print('\n-----Trying as-is distractor-----')

distr2 = AsIsDistractor(d_factor=distr.d_factor, batch_size=s0.default_batch_size)
model2 = ONMTSummaryRSA(s0, pragmatics, distr2)
src2, _ = distr.generate(src)
print('src2', src2)
print(model2.incremental_s1(src2, beam_size=10, n_best=1, diverse_beam='rank'))
