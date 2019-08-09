import sys, os, time, logging, argparse
import torch
import bbrsa
from tqdm import tqdm

import cProfile, pstats

# sys.path.append(os.path.abspath(bbrsa.ONMT_DIR))

from bbrsa.bbrsa import ONMTRSAModel
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.evaluators import Evaluator
from bbrsa.pragmatics import BasicPragmatics, GrowingAlphaPragmatics
from bbrsa.distractors import NextExampleDistractor, BertDistractor
from bbrsa.utils import init_logger, display
# from models import ONMTSummarizer
# from evaluators import Evaluator
# from utils import init_logger, display

opts = bbrsa.DEFAULT_OPTS
large_clean_input_path = 'data/giga_2000valid_art.txt'
small_clean_input_path = 'data/giga_test_50.txt'
medium_clean_input_path = 'data/giga_test_500.txt'
part1_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_nocov_step_59156_valacc48.57_ppl15.51.pt'
part2_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt2_nocov_step_59156.pt'
diff_output_path = 'results/s0s1diff'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=part1_model_path)
parser.add_argument('-o', '--output', default=diff_output_path)
args = parser.parse_args()

logger = init_logger(no_format=True, print_level=logging.DEBUG)
# eval_s0 = ONMTSummarizer(opts, part1_model_path)
# evaluator = Evaluator(eval_s0, logger=logger)

opts.gpu = True
opts.batch_size = 32
opts.beam_size = 10
opts.n_best = 1

opts.distractor = 'next_1'

print(opts)

model = ONMTRSAModel.from_opts(opts, part1_model_path)

# summ_s0 = ONMTSummarizer(opts, args.model, logger=logger)
# pragmatics = BasicPragmatics(opts) # ok
# distractor = NextExampleDistractor(opts)
# model = ONMTRSAModel(summ_s0, pragmatics, distractor, opts)

evaluator = Evaluator.from_opts(opts, part2_model_path, logger=logger)

with open(large_clean_input_path, 'r') as f:
    src = [s.strip() for s in f.readlines()]

"""
print('Running incremental s1')
start_time = time.time()
s1pred = model.incremental_s1(src, opts)
print('=============================')
print('Finished running, took {:.4} s'.format(time.time() - start_time))
print('Running incremental s0')
start_time = time.time()
s0pred = model.summarize_s0(src, opts)
print('=============================')
print('Finished running, took {:.4} s'.format(time.time() - start_time))

s1outf = open(args.output + '_s1.txt', 'w')
s0outf = open(args.output + '_s0.txt', 'w')

for x, y in zip(s1pred, s0pred):
    s1sent = x[0]
    s0sent = y[0]
    s1outf.write(s1sent+'\n')
    s0outf.write(s0sent+'\n')

s0outf.close()
s1outf.close()
"""

print('Evaluating incremental s1')
start_time = time.time()
acc1, __ = evaluator.split_evaluate(model, src, 'incr_s1', opts)
print('=============================')
print('Finished evaluating, took {:.4} s, acc={:.4}'.format(time.time() - start_time,
                                                            acc1))
print('Evaluating s0')
start_time = time.time()
acc0, _ = evaluator.split_evaluate(model, src, 's0', opts)
print('=============================')
print('Finished evaluating, took {:.4} s, acc={:.4}'.format(time.time() - start_time,
                                                            acc0))

# ps.print_stats()

# print('Writing results')
# with open('results/inc_s0_output_alpha4.txt', 'w') as f:
#     for s in tqdm(pred):
#         f.write('{}\n'.format(s[0]))
