import bbrsa, logging
from bbrsa.grid_search import *
from bbrsa.utils import init_logger

logging.basicConfig(level=logging.WARNING)
logger = init_logger(print_level=logging.DEBUG)

default_opts = bbrsa.DEFAULT_OPTS

part1_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt1_nocov_step_59156_valacc48.57_ppl15.51.pt'
part2_model_path = '/home/hansonlu/links/data/giga-models/giga_halfsplit_pt2_nocov_step_59156.pt'
src_path = 'data/giga_50valid_art.txt'
tgt_path = 'data/giga_50valid_tgt.txt'
db_path = 'tables/50testres.db'

test_opts = default_opts.clone()
test_opts.set_as_default({'beam_size': 5,
                          'batch_size': 10,
                          'prag_alpha': 1.,
                          'distractor': 'bert',
                          'gpu': True,
                          'mode': 'incr_s1'})

gs = GridSearch(part1_model_path, part2_model_path, src_path, tgt_path,
                        db_path, test_opts, logger=logger)


test_res_dict = {
    'pred_out_file': 'path/testres.txt',
    'listener_acc': 0.60,
    'ROUGE_1R': 0.3,
    'ROUGE_1P': 0.1,
    'ROUGE_1F': 0.2,
    'ROUGE_2R': 0.3,
    'ROUGE_2P': 0.1,
    'ROUGE_2F': 0.2,
    'ROUGE_LR': 0.3,
    'ROUGE_LP': 0.1,
    'ROUGE_LF': 0.2
}

# newid = gridsearch.db_insert(test_opts, test_res_dict)

test_grid_dict = {'mode': ['incr_s1'],
                  'bert_distr_d_factor': [5],
                  'bert_distr_repl_search': [(0, 5)],
                  'prag_alpha': [2.]}
gs.execute(test_grid_dict, pred_save_file='gs_results/50test')

# s0_grid_dict = {'mode': ['s0']}
# gs.execute(s0_grid_dict, pred_save_file='gs_results/50test')
