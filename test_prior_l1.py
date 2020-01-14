import sys, os, time, logging, argparse
import torch
import bbrsa
from tqdm import tqdm

import cProfile, pstats

# sys.path.append(os.path.abspath(bbrsa.ONMT_DIR))

from bbrsa.bbrsa import ONMTRSAModel
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.evaluators import Evaluator
from bbrsa.pragmatics import BasicPragmatics, MemoizedListener
from bbrsa.distractors import NextExampleDistractor, BertDistractor
from bbrsa.utils import init_logger, display
# from models import ONMTSummarizer
# from evaluators import Evaluator
# from utils import init_logger, display

opts = bbrsa.DEFAULT_OPTS
part1_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_shuf_step_73945.pt'
src_path = '/home/hansonlu/CSLI/bbrsa/data/valid.head20.art.txt'
tgt_path = '/home/hansonlu/CSLI/bbrsa/data/valid.head20.tgt.txt'

# part2_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt2_nocov_step_59156.pt'
stats_dump = 'results/profiler_l1_valid.prof'

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=part1_model_path)
parser.add_argument('-o', '--out', default=stats_dump)
parser.add_argument('-p', '--profiler', action="store_true")
args = parser.parse_args()

logger = init_logger(no_format=True, print_level=logging.DEBUG)
# eval_s0 = ONMTSummarizer(opts, part1_model_path)
# evaluator = Evaluator(eval_s0, logger=logger)

opts.gpu = False
opts.batch_size = 1
opts.beam_size = 20
opts.n_best = 1
opts.prag_alpha = 1.5
opts.bert_distr_repl_search_top = 5
opts.bert_distr_repl_search_bottom = 10
opts.bert_distr_no_subword_repl = True
opts.bert_distr_exclusion_set = 3

summ_s0 = ONMTSummarizer(opts, args.model, logger=logger)
pragmatics = MemoizedListener(opts) # ok
distractor = BertDistractor(opts)
model = ONMTRSAModel(summ_s0, pragmatics, distractor, opts)

# src = ['police arrested five climate-change protesters friday after they sought to disrupt loading of a french arctic research and supply vessel , a spokesman for the protesters said .']

# with open(src_path, 'r') as f:
#     src = [s.strip() for s in f.readlines()]
# with open(tgt_path, 'r') as f:
#     tgt = [s.strip() for s in f.readlines()]

src = ["china 's army has won a war against a toy maker without firing a shot ."]
tgt = ["chinese army wins legal war with toy maker"]

print('Running incremental s1 on %s' % ('gpu' if opts.gpu else 'cpu'))

s0_pred = model.summarize_s0(src, opts)
memo_l1_pred = model.incremental_s1(src, opts)

pragmatics = BasicPragmatics(opts)
model = ONMTRSAModel(summ_s0, pragmatics, distractor, opts)
basic_pred = model.incremental_s1(src, opts)

display(['src', 'tgt', 's0', 'basic_s1', 'memo_l1'],
        [src, tgt, s0_pred, basic_pred, memo_l1_pred])
# ps.print_stats()

# print('Writing results')
# with open('results/inc_s0_output_alpha4.txt', 'w') as f:
#     for s in tqdm(pred):
#         f.write('{}\n'.format(s[0]))
