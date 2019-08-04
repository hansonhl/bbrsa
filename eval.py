import time
import sys, os
import torch
import logging
import bbrsa

# sys.path.append(os.path.abspath(bbrsa.ONMT_DIR))

from bbrsa.bbrsa import ONMTSummaryRSA
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.evaluators import Evaluator
from bbrsa.pragmatics import BasicPragmatics
from bbrsa.distractors import NextExampleDistractor, BertDistractor
from bbrsa.utils import init_logger, display
# from models import ONMTSummarizer
# from evaluators import Evaluator
# from utils import init_logger, display

logger = init_logger(no_format=True, print_level=logging.DEBUG)
eval_s0 = ONMTSummarizer(config_path='onmt_configs/giga_split2.yml', logger=logger)
evaluator = Evaluator(eval_s0, add_tags=False, logger=logger)

pragmatics = BasicPragmatics(alpha=3)
summ_s0 = ONMTSummarizer(config_path='onmt_configs/giga_split1.yml', logger=logger)
distractor = BertDistractor(batch_size=summ_s0.default_batch_size)
model = ONMTSummaryRSA(summ_s0, pragmatics, distractor)


# srcs = ['police arrested five climate-change protesters friday after they sought to disrupt loading of a french arctic research and supply vessel , a spokesman for the protesters said .']

with open('/home/hansonlu/myOpenNMT/data/giga/input_clean.txt', 'r') as f:
    srcs = [s.strip() for s in f.readlines()]


acc = evaluator.split_evaluate(model, distractor, srcs, verbose=True)
print('Accuracy: {:.5}'.format(acc))

# pred = model.summarize_s0(src, beam_size=10)

tgt = 'police arrest climate-change protesters on saturday'
# [0.9315, 0.0500, 0.0185], cnndm
# [0.0010, 0.9974, 0.0016], giga

tgt = 'climate-change protesters arrested for disrupting french research'
# [0.9186, 1.3e-06, 0.0814], cnndm
# [0.2232, 6.9e-05, 0.7768], giga
