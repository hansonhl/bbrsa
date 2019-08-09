import bbrsa, logging
from bbrsa.grid_search import *
from bbrsa.utils import init_logger

logging.basicConfig(level=logging.WARNING)
logger = init_logger(print_level=logging.INFO)

default_opts = bbrsa.DEFAULT_OPTS

part1_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_nocov_step_59156_valacc48.57_ppl15.51.pt'
part2_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt2_nocov_step_59156.pt'
src_path = 'data/giga_2000valid_art.txt'
tgt_path = 'data/giga_2000valid_tgt.txt'
db_path = 'tables/2000res.db'
pred_save_file = '/home/hansonlu/links/data/2000val/2000val'

test_opts = default_opts.clone()
test_opts.set_as_default({'beam_size': 10,
                          'batch_size': 32,
                          'prag_alpha': 1.,
                          'mode': 'incr_s1',
                          'gpu': True,
                          'shard_size': 2000,
                          'distractor': 'bert',
                          'bert_distr_d_factor': 2})

gs = GridSearch(part1_model_path, part2_model_path, src_path, tgt_path,
                        db_path, test_opts, logger=logger)

# s0_grid_dict = {'mode': ['s0'],
#                 'bert_distr_repl_search': [(5, 10), (10, 20)]}
# gs.execute(s0_grid_dict, pred_save_file=pred_save_file)

test_grid_dict = {'mode': ['incr_s1'],
                  'bert_distr_repl_search': [(20, 50)],
                  'prag_alpha': [0.5, 1.0, 1.5, 2., 2.5, 3., 3.5]}
gs.execute(test_grid_dict, pred_save_file=pred_save_file)
