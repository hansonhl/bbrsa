import bbrsa, logging
from bbrsa.grid_search import *
from bbrsa.utils import init_logger

default_opts = bbrsa.DEFAULT_OPTS
logging.basicConfig(level=logging.WARNING)
logger = init_logger(print_level=logging.INFO)

part1_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_shuf_step_73945.pt'
part2_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt2_shuf_step_73945.pt'
src_path = 'data/valid.head50.art.txt'
tgt_path = 'data/valid.head50.tgt.txt'
db_path = 'tables/debug.db'
verbose_output = '/home/hansonlu/links/data/debug/debug'

test_opts = default_opts.clone()
test_opts.set_as_default({'mode': 'incr_s1',
                          'pragmatics': 'basic',
                          'beam_size': 5,
                          'batch_size': 5,
                          'prag_alpha': 1.,
                          'mode': 'incr_s1',
                          'gpu': True,
                          'shard_size': 2000,
                          'distractor': 'bert',
                          'bert_distr_no_subword_repl': True,
                          'bert_distr_d_factor': 5,
                          'bert_distr_exclusion_set': 3,
                          'bert_distr_repl_search_top': 5,
                          'bert_distr_repl_search_bottom': 10})

gs = GridSearch(part1_model_path, part2_model_path, src_path, tgt_path,
                        db_path, test_opts, logger=logger)

grid_dict = {'batch_size': [3, 5, 10]}
gs.execute(grid_dict, verbose_output=verbose_output)
