import logging
import json
import torch
import sqlite3

from onmt.utils.parse import ArgumentParser
import onmt.opts as opts
from onmt.translate.translator import build_translator

class ProbAttnDump(object):
    # works only for batch size of one, greedy search
    def __init__(self):
        self.attns = []
        self.curr_preds = []

    def init(self, src):
        self.src = src.split()
        self.attns = []
        self.curr_preds = []

    def advance(self, attn, curr_preds):
        assert len(attn.shape) == 3
        # beam_size = attn.shape[1]
        # orig_len = attn.shape[2]
        # self.attns.append(attn.view(beam_size, orig_len))
        self.attns.append(attn.view(-1).tolist())
        self.curr_preds.append(curr_preds.view(-1))

    def finalize(self, tgt):
        self.tgt = tgt.split()

    def to_attn_vis_json(self, file_name, ref=None):
        data = {}
        data['article_lst'] = self.src
        data['decoded_lst'] = self.tgt
        data['abstract_str'] = ref if ref is not None else ' '.join(self.tgt)
        data['attn_dists'] = self.attns

        with open(file_name, 'w') as f:
            json.dump(data, f)


def init_logger(no_format=False, print_level=logging.DEBUG, log_file=None,
    log_file_level=logging.DEBUG, log_mode='a'):
    """Initialize logger""" # modified from onmt/utils/logging.py
    default_level = logging.DEBUG
    if no_format:
        log_format = logging.Formatter("%(message)s")
    else:
        log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(default_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(print_level)
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file, mode=log_mode)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def display(names, preds):
    """
    Display predictions given by bbrsa summarization module

    Args:
        names (list[str]): the name of each set of predictions.
        preds (list): model output
    """
    num_examples = len(preds[0])
    for i in range(num_examples):
        for j, s in enumerate(preds):
            if isinstance(s[i], list):
                s = s[i][0]
            elif isinstance(s[i], str):
                s = s[i]
            else:
                print('Error in pred type!')
            print(names[j] + ': ' + s.strip())
        print('')

def onmt_translator_builder(my_opts, s0_model_path, logger=None):
    if logger is not None:
        logger.info('Building ONMT translator with model from ' + s0_model_path)
    else:
        print('Building ONMT translator with model from ' + s0_model_path)

    parser = ArgumentParser()
    opts.config_opts(parser)
    opts.translate_opts(parser)

    arglist = ['-model', s0_model_path] + opts_to_list(my_opts)
    opt = parser.parse_args(arglist)

    ArgumentParser.validate_translate_opts(opt)
    translator = build_translator(opt, report_score=True, logger=logger)
    if logger is not None:
        logger.info('Finished building ONMT translator.')
    else:
        print('Finished building ONMT translator.')

    return translator

def opts_to_list(opts):
    res = []
    lookup = {
        's0_model_path': '-model',
        'batch_size': '-batch_size',
        'beam_size': '-beam_size',
        'dummy_src': '-src',
        'coverage_penalty': '-coverage_penalty',
        'coverage_penalty_beta': '-beta',
        'length_penalty': '-length_penalty',
        'length_penalty_alpha': '-alpha',
        'block_ngram_repeat': '-block_ngram_repeat',
        'min_length': '-min_length',
        'stepwise_penalty': '-stepwise_penalty'
    }
    for k, v in opts:
        if k not in lookup:
            continue
        val = str(v.value)
        if val == 'False':
            continue
        res.append(lookup[k])
        if val == 'True':
            continue
        if val == 'None':
            val = 'none'
        res.append(val)
        if k == 'gpu':
            res += ['-gpu', '0']
        if k == 's0_block_ngram_repeat':
            res += ['-ignore_when_blocking', ".", "</t>", "<t>"]

    res += ['-max_length', '80']
    return res


def scramble2tgt(idxs, d_factor, device=None):
    """Given scrambling indices, get indices of target examples"""
    # e.g. input idxs=[1,2,0,3], d_factor=2 -> out [2, 1],
    # elements 2 and 1 in list [1,2,0,3] are the indices of two targets
    # respectively, in correct order

    if isinstance(idxs, torch.Tensor):
        idxs_len = idxs.shape[0]
        if device is None:
            device = idxs.device
    elif isinstance(idxs, list):
        idxs_len = len(idxs)
        if device is None:
            device = torch.device('cpu')
    scrambled = idx_remap(idxs)
    return scrambled[torch.arange(0, idxs_len, d_factor, device=device)]

def idx_remap(idxs):
    # e.g. input idxs=[1,2,0,3] -> output [2,0,1,3]
    _, perm = torch.sort(idxs)
    return perm

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(l), n):
        yield l[i:i + n]

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    else:
        return text

def db_connect(db_file):
    """ create a database connection to the SQLite database

    Args:
        db_file: database file

    Returns: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return None
