import sys, os, time, logging, argparse
import torch
import bbrsa
from tqdm import tqdm

import cProfile, pstats

# sys.path.append(os.path.abspath(bbrsa.ONMT_DIR))

from bbrsa.bbrsa import ONMTSummaryRSA
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.evaluators import Evaluator
from bbrsa.pragmatics import BasicPragmatics, GrowingAlphaPragmatics
from bbrsa.distractors import NextExampleDistractor, BertDistractor
from bbrsa.utils import init_logger, display
# from models import ONMTSummarizer
# from evaluators import Evaluator
# from utils import init_logger, display

opts = bbrsa.DEFAULT_OPTS
large_clean_input_path = '/home/hansonlu/myOpenNMT/data/giga/input_clean.txt'
small_clean_input_path = 'data/giga_test_50.txt'
medium_clean_input_path = 'data/giga_test_500.txt'
part1_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_nocov_step_59156_valacc48.57_ppl15.51.pt'
# part2_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt2_nocov_step_59156.pt'

parser = argparse.ArgumentParser()
parser.add_argument('model_path', default=part1_model_path)
args = parser.parse_args()

logger = init_logger(no_format=True, print_level=logging.DEBUG)
# eval_s0 = ONMTSummarizer(opts, part1_model_path)
# evaluator = Evaluator(eval_s0, logger=logger)

opts.gpu = False
opts.batch_size = 16
opts.beam_size = 10
opts.n_best = 1
opts.prag_alpha = 3.

summ_s0 = ONMTSummarizer(opts, args.model_path, logger=logger)
pragmatics = BasicPragmatics(opts) # ok
distractor = BertDistractor(opts)
model = ONMTSummaryRSA(summ_s0, pragmatics, distractor, opts)

# src = ['police arrested five climate-change protesters friday after they sought to disrupt loading of a french arctic research and supply vessel , a spokesman for the protesters said .']

with open(medium_clean_input_path, 'r') as f:
    src = [s.strip() for s in f.readlines()]

print('Running incremental s1 on gpu')
pr = cProfile.Profile()
start_time = time.time()
pr.enable()
pred = model.incremental_s1(src, opts)
pr.disable()
print('=============================')
print('Finished running, took {:.4} s'.format(time.time() - start_time))

sortby = 'cumulative'
ps = pstats.Stats(pr).sort_stats(sortby)
ps.print_stats()

# print('Writing results')
# with open('results/inc_s0_output_alpha4.txt', 'w') as f:
#     for s in tqdm(pred):
#         f.write('{}\n'.format(s[0]))
