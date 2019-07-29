import time
import sys, os
import torch
import logging
import bbrsa

sys.path.append(os.path.abspath(bbrsa.ONMT_DIR))

from bbrsa.bbrsa import ONMTSummaryRSA
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.evaluators import Evaluator
from bbrsa.pragmatics import BasicPragmatics
from bbrsa.distractors import NextExampleDistractor
from bbrsa.utils import init_logger, display
# from models import ONMTSummarizer
# from evaluators import Evaluator
# from utils import init_logger, display

logger = init_logger(no_format=True, print_level=logging.DEBUG)
s0 = ONMTSummarizer(config_path='onmt_configs/giga.yml', logger=logger)
evaluator = Evaluator(s0, add_tags=False, logger=logger)

pragmatics = BasicPragmatics(alpha=1)
distractor = NextExampleDistractor(batch_size=s0.default_batch_size)
model = ONMTSummaryRSA(s0, pragmatics, distractor)

src = ['police arrested seven climate-change protesters friday after they sought to disrupt loading of a french arctic research and supply vessel , a spokesman for the protesters said .',
    'police arrested six climate-change protesters saturday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said .',
    'police arrested eighteen climate-change protesters friday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said .']

pred = model.summarize_with_s0(src, beam_size=10)

tgt = 'police arrest climate-change protesters on saturday'
# [0.9315, 0.0500, 0.0185], cnndm
# [0.0010, 0.9974, 0.0016], giga

tgt = 'climate-change protesters arrested for disrupting french research'
# [0.9186, 1.3e-06, 0.0814], cnndm
# [0.2232, 6.9e-05, 0.7768], giga


res = evaluator.evaluate(src, tgt)
print(res.tolist())
