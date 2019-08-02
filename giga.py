import time
import sys, os
import torch
import logging
import bbrsa

sys.path.append(os.path.abspath(bbrsa.ONMT_DIR))

from bbrsa.bbrsa import ONMTSummaryRSA
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.distractors import NextExampleDistractor, IdenticalDistractor, NextNDistractor
from bbrsa.pragmatics import BasicPragmatics, GrowingAlphaPragmatics, MemoizedListener
from bbrsa.utils import init_logger, display, ProbAttnDump

logger = init_logger(no_format=True, print_level=logging.DEBUG)

giga_config_path = 'onmt_configs/giga.yml'
s0 = ONMTSummarizer(config_path=giga_config_path, logger=logger)
pragmatics = GrowingAlphaPragmatics(alpha=1, steps=3, logger=logger)
distractor = NextNDistractor(batch_size=s0.default_batch_size, N=1, logger=logger)
model = ONMTSummaryRSA(s0, pragmatics, distractor, logger=logger)


src = ['police arrested five anti-nuclear protesters friday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said .',
    'police arrested five anti-nuclear protesters friday after they sought to disrupt loading of an american antarctic research and supply vessel , a spokesman for the protesters said .',
    'french police arrested five anti-nuclear protesters friday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said .',
    'american police arrested five anti-nuclear protesters friday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said .']

# src = ['a sample survey , launched by a taipei-based human resources bank , showed that ## percent of taiwanese office workers were willing to working on the chinese mainland , up #.# percent from #### .',
#     'a sample survey , launched by a taipei-based human resources bank , showed that ## percent of taiwanese office workers were willing to working on the chinese mainland , down #.# percent from #### .']
"""
'French police arrested five anti-nuclear protesters friday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said .',
'French police arrested five anti-nuclear protesters friday after they sought to disrupt loading of an american antarctic research and supply vessel , a spokesman for the protesters said .'
"""
# dump = ProbAttnDump()
model.set_alpha(5)
s0_pred = model.summarize_s0(src, n_best=1, beam_size=20)
s1_pred = model.incremental_s1(src, beam_size=20)
#json_file = 'results/test2.json'
#dump.to_attn_vis_json(json_file)
print(s0_pred)

display(['s0', 's1'], [s0_pred, s1_pred])
"""
[['anti-nuclear protesters arrested',
'police arrest anti-nuclear protesters',
'five anti-nuclear protesters arrested',
'anti-nuclear protesters arrested in paris',
'anti-nuclear protesters arrested in france',
'police arrest five anti-nuclear protesters',
'anti-nuclear protesters arrested in french antarctic',
'police arrest # anti-nuclear protesters',
'five anti-nuclear protesters arrested in paris',
'five anti-nuclear protesters arrested in france'],
['anti-nuclear protesters arrested',
'police arrest anti-nuclear protesters',
'five anti-nuclear protesters arrested',
'police arrest five protesters in antarctic protest',
'police arrest five protesters at antarctic research',
'police arrest five protesters at antarctic site',
'police arrest five anti-nuclear protesters',
'police arrest # anti-nuclear protesters',
'five anti-nuclear protesters arrested at antarctic',
'police arrest five anti-nuclear protesters in antarctica'], ['anti-nuclear protesters arrested', 'anti-nuclear protesters arrested in paris', 'police arrest anti-nuclear protesters', 'five anti-nuclear protesters arrested', 'anti-nuclear protesters arrested in france', 'anti-nuclear protesters arrested in french antarctic', 'police arrest five anti-nuclear protesters', 'police arrest # anti-nuclear protesters', 'five anti-nuclear protesters arrested in paris', 'five anti-nuclear protesters arrested in france'], ['anti-nuclear protesters arrested', 'police arrest anti-nuclear protesters', 'five anti-nuclear protesters arrested', 'police arrest five protesters in antarctic protest', 'police arrest five protesters at antarctic research', 'police arrest five anti-nuclear protesters', 'police arrest five protesters at antarctic site', 'police arrest # anti-nuclear protesters', 'five anti-nuclear protesters arrested at antarctic', 'police arrest five anti-nuclear protesters in brazil']]
"""
