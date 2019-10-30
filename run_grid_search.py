import bbrsa, logging
from bbrsa.grid_search import *
from bbrsa.utils import init_logger

logging.basicConfig(level=logging.WARNING)
logger = init_logger(print_level=logging.INFO)

default_opts = bbrsa.DEFAULT_OPTS

part1_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_shuf_step_73945.pt'
part2_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt2_shuf_step_73945.pt'
src_path = 'data/valid.rsa.art.txt'
tgt_path = 'data/valid.rsa.tgt.txt'
db_path = 'tables/2031res.db'
pred_save_file = '/home/hansonlu/links/data/2031val/2031val' #'/home/hansonlu/CSLI/bbrsa/results/tests0'

test_opts = default_opts.clone()
test_opts.set_as_default({'beam_size': 10,
                          'batch_size': 32,
                          'prag_alpha': 1.,
                          'mode': 'incr_s1',
                          'gpu': True,
                          'shard_size': 2000,
                          'distractor': 'bert',
                          'bert_distr_no_subword_repl': True,
                          'bert_distr_d_factor': 2})

gs = GridSearch(part1_model_path, part2_model_path, src_path, tgt_path,
                        db_path, test_opts, logger=logger)

# s0_grid_dict = {'mode': ['s0'],
#                 'bert_distr_repl_search': [(5,10), (10, 20)],
#                 'bert_distr_exclusion_set': [3],
#                 'bert_distr_d_factor': [2, 5, 10]}
# gs.execute(s0_grid_dict, pred_save_file=pred_save_file)

# test_grid_dict = {'mode': ['s0'],
#                   'bert_distr_d_factor': [2],
#                   'bert_distr_repl_search': [(10, 20)],
#                   'bert_distr_no_subword_repl': [True],
#                   'bert_distr_exclusion_set': [0, 3]}
# gs.execute(test_grid_dict, pred_save_file=pred_save_file)
#

test_grid_dict = {'mode': ['incr_s1'],
                  'batch_size': [8],
                  'shard_size': [256],
                  'bert_distr_d_factor': [10],
                  'bert_distr_repl_search': [(5,10)],
                  'prag_alpha': [2.5, 3.],
                  'bert_distr_no_subword_repl': [True],
                  'bert_distr_exclusion_set': [3]}

gs.execute(test_grid_dict, pred_save_file=pred_save_file)

# """Test to get variance"""
# test_grid_dict = {'mode': ['incr_s1'],
#                   'random_seed': list(range(39831, 39851)),
#                   'bert_distr_d_factor': [5],
#                   'bert_distr_repl_search': [(5,10)], # not sure about this?
#                   'prag_alpha': [1.5],
#                   'bert_distr_no_subword_repl': [True],
#                   'bert_distr_exclusion_set': [3]}

gs.execute(test_grid_dict, pred_save_file=pred_save_file)
