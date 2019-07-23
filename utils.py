import logging
import json

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
