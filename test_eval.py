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

opts = bbrsa.DEFAULT_OPTS
part1_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_nocov_step_59156_valacc48.57_ppl15.51.pt'
part2_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt2_nocov_step_59156.pt'


logger = init_logger(no_format=True, print_level=logging.DEBUG)
eval_s0 = ONMTSummarizer(opts, part1_model_path)
evaluator = Evaluator(eval_s0, logger=logger)

summ_s0 = ONMTSummarizer(opts, part2_model_path, logger=logger)
pragmatics = BasicPragmatics()
distractor = NextExampleDistractor()
model = ONMTSummaryRSA(summ_s0, pragmatics, distractor)


# srcs = ['police arrested five climate-change protesters friday after they sought to disrupt loading of a french arctic research and supply vessel , a spokesman for the protesters said .']

with open('/home/hansonlu/myOpenNMT/data/giga/input_clean.txt', 'r') as f:
    srcs = [s.strip() for s in f.readlines()]


acc = evaluator.split_evaluate(model, distractor, srcs, opts, verbose=True)
print('Accuracy: {:.5}'.format(acc))

# pred = model.summarize_s0(src, beam_size=10)

tgt = 'police arrest climate-change protesters on saturday'
# [0.9315, 0.0500, 0.0185], cnndm
# [0.0010, 0.9974, 0.0016], giga

tgt = 'climate-change protesters arrested for disrupting french research'
# [0.9186, 1.3e-06, 0.0814], cnndm
# [0.2232, 6.9e-05, 0.7768], giga
