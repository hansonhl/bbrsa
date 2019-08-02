import logging
import json
import torch

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
    """Display predictions given by bbrsa summarization module
    Args:
        names: a list of strings indicating the name of each set of predictions.
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

def onmt_translator_builder(config_path, logger=None):
    if logger is not None:
        logger.info('Building ONMT translator with configs from ' + config_path)
    else:
        print('Building ONMT translator with configs from ' + config_path)

    parser = ArgumentParser(default_config_files=[config_path])
    opts.config_opts(parser)
    opts.translate_opts(parser)

    opt = parser.parse_args(['-config', config_path])

    ArgumentParser.validate_translate_opts(opt)
    translator = build_translator(opt, report_score=True, logger=logger)
    if logger is not None:
        logger.info('Finished building ONMT translator.')
    else:
        print('Finished building ONMT translator.')

    return translator, opt


def scramble2tgt(idxs, d_factor):
    """Given scrambling indices, get indices of target examples"""
    # e.g. input idxs=[1,2,0,3], d_factor=2 -> out [2, 1],
    # elements 2 and 1 in list [1,2,0,3] are the indices of two targets
    # respectively, in correct order
    if isinstance(idxs, torch.Tensor):
        idxs_len = idxs.shape[0]
    elif isinstance(idxs, list):
        idxs_len = len(idxs)
    scrambled = idx_remap(idxs)
    return scrambled[torch.arange(0, idxs_len, d_factor)]

def idx_remap(idxs):
    # e.g. input idxs=[1,2,0,3] -> output [2,0,1,3]
    _, perm = torch.sort(idxs)
    return perm

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    # from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    for i in range(0, len(l), n):
        yield l[i:i + n]
