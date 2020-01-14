import time, torch, sqlite3, bbrsa, rouge

from bbrsa.abstract_classes import BBRSAABC
from bbrsa.bbrsa import ONMTRSAModel
from bbrsa.configopts import ConfigOpts
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.evaluators import Evaluator
from bbrsa.utils import remove_prefix, chunks, db_connect

from itertools import product
from datetime import datetime

SAMPLE_RES_DICT = {
    'pred_save_file': 'path/testres.txt',
    'listener_acc': 0.,
    'eval_duration': 0.,
    'rouge_1r': 0.,
    'rouge_1p': 0.,
    'rouge_1f': 0.,
    'rouge_2r': 0.,
    'rouge_2p': 0.,
    'rouge_2f': 0.,
    'rouge_lr': 0.,
    'rouge_lp': 0.,
    'rouge_lf': 0.
}

class GridSearch(BBRSAABC):
    def __init__(self, summ_model_path, eval_model_path,
                 src_path, tgt_path, db_path, opts, logger=None):
        """Initializes the GridSearch class.

        Assumes that parameters of the s0 models are not changed
        """
        super().__init__(logger)
        # initialize the following only once, with their init code constant
        self.summ_s0 = ONMTSummarizer(opts, summ_model_path, logger)
        self.eval_s0 = ONMTSummarizer(opts, eval_model_path, logger)
        # self.distractor = bbrsa.str2distr[opts.distractor](opts, logger)
        # for now, do not initialize distractor here as it may be changed
        # for now, do not initialize pragmatics here as it may be changed

        self.evaluator = Evaluator(self.eval_s0, opts, logger)
        # for now, do not initialize model as pragmatics is fluid.
        self.rouge_scorer = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                                        max_n=2,
                                        limit_length=False,
                                        stemming=True)

        with open(src_path, 'r') as srcf:
            self.src = [s.strip() for s in srcf.readlines()]
        with open(tgt_path, 'r') as tgtf:
            self.tgt = [s.strip() for s in tgtf.readlines()]

        self.default_opts = opts.clone()

        self.db_path = db_path
        if self.db_path:
            conn = db_connect(self.db_path)
            cmd, opts_cols, res_cols = create_cmd(self.default_opts)
            with conn:
                cur = conn.cursor()
                cur.execute(cmd)
            self.db_opts_cols = opts_cols
            self.db_res_cols = res_cols

    def execute(self, grid_dict, pred_save_file=None, verbose_output=None, verbose=True):
        var_opt_names = list(grid_dict.keys())
        var_opt_values = list(grid_dict.values())

        # treat elements in list as separate args to fxn
        for tup in product(*var_opt_values):
            value_dict = {}
            for name, val in zip(var_opt_names, tup):
                if name == 'bert_distr_repl_search':
                    value_dict['bert_distr_repl_search_top'] = val[0]
                    value_dict['bert_distr_repl_search_bottom'] = val[1]
                else:
                    value_dict[name] = val

            curr_opts = self.default_opts.clone()
            curr_opts.set_values(value_dict)
            mode = curr_opts.mode
            self.run_once(curr_opts, mode, pred_save_file, verbose_output, verbose)

    def run_once(self, opts, mode, pred_save_file, verbose_output, verbose):
        logger = self.logger
        summ_s0 = self.summ_s0
        pragmatics = bbrsa.str2prag[opts.pragmatics](opts, logger)
        distractor = bbrsa.str2distr[opts.distractor](opts, logger)
        model = ONMTRSAModel(summ_s0, pragmatics, distractor, opts, logger)

        start_time = time.time()
        if verbose_output is None:
            acc, preds = self.evaluator.split_evaluate(model, self.src, mode,
                                                       opts)
        else:
            acc, preds, distractors = self.evaluator.split_evaluate(
                model, self.src, mode, opts, output_distractors=True)
        duration = time.time() - start_time

        res_dict = {'eval_duration': duration, 'listener_acc': acc}
        self._calculate_rouge(preds, res_dict)
        if verbose:
            self._display_results(opts, res_dict)
        if verbose_output:
            self._verbose_output(opts, res_dict, distractors, preds, verbose_output)
        if pred_save_file:
            self._record_results(opts, res_dict, preds, pred_save_file)
        del model

    def _calculate_rouge(self, preds, res_dict):
        scores = self.rouge_scorer.get_scores(preds, self.tgt)
        rpf = ['r', 'p', 'f']
        for metric, results in scores.items():
            metric = metric.lower().replace('-', '_')
            for x in rpf:
                res_dict[metric+x] = results[x]

    def _display_results(self, opts, res_dict):
        val_strings = []
        for k, v in opts.value_dict().items():
            name = remove_prefix(k, 'bert_distr_')
            name = remove_prefix(name, 'prag_')
            val_strings.append('{}={}'.format(name, v))
        self._info(' '.join(val_strings))
        self._info('>>>> Acc={:.4}, Rouge-1/2/L-F={:.4}/{:.4}/{:.4}, Eval Time={:.4}'\
              .format(100. * res_dict['listener_acc'],
                      100. * res_dict['rouge_1f'],
                      100. * res_dict['rouge_2f'],
                      100. * res_dict['rouge_lf'],
                      res_dict['eval_duration']))

    def _verbose_output(self, opts, res_dict, distractors, preds, out_file):
        distrs = distractors[0]
        date_str = datetime.now().strftime("%m%d_%H%M%S")
        pred_save_file = '{}_{}_acc{:.4}_rglf{:.4}.txt' \
                        .format(out_file, date_str,
                                res_dict['listener_acc']*100.,
                                res_dict['rouge_lf']*100.)
        with open(pred_save_file, 'w') as f:
            f.write(datetime.now().strftime("# %Y/%m/%d %H:%M Results\n"))
            f.write('\n## Parameters\n')
            f.write(str(opts))

            f.write('\n## Results\n')
            for k, v in res_dict.items():
                f.write('- {}: {}\n'.format(k, v))

            if opts.distractor == 'bert':
                d_factor = opts.bert_distr_d_factor
            else:
                d_factor = self.distractor.d_factor

            f.write('\n## Distractors and outputs\n')
            for pred, ref, distrs in zip(preds, self.tgt,
                                         chunks(distrs, d_factor)):
                f.write('__{}__  \n'.format(distrs[0].replace('``', "\'\'")))
                for d in distrs[1:]:
                    f.write(d.replace('``', "\'\'") + '  \n')
                f.write('\n__REFR__: {}  \n'.format(ref))
                f.write('__PRED__: {}\n\n'.format(pred))
        self._info('>>>> Saved to {}'.format(pred_save_file))

    def _record_results(self, opts, res_dict, preds, pred_save_file):
        date_str = datetime.now().strftime("%m%d_%H%M%S")
        pred_save_file = '{}_{}_acc{:.4}_rglf{:.4}.txt' \
                        .format(pred_save_file, date_str,
                                res_dict['listener_acc']*100.,
                                res_dict['rouge_lf']*100.)
        with open(pred_save_file, 'w') as f:
            for line in preds:
                f.write(line+'\n')
        res_dict['pred_save_file'] = pred_save_file
        self._info('>>>> Saved to {}'.format(res_dict['pred_save_file']))
        if self.db_path:
            self.db_insert(opts, res_dict)
            self._info('>>>> Added entry to database {}'.format(self.db_path))

    def db_insert(self, opts, res_dict):
        assert len(opts) >= len(self.db_opts_cols)
        assert len(res_dict) >= len(self.db_res_cols)

        all_cols = self.db_opts_cols + self.db_res_cols
        cmd = insert_cmd(all_cols)
        l = []
        for item in self.db_opts_cols:
            l.append(opts[item])
        for item in self.db_res_cols:
            l.append(res_dict[item])

        cmd_tuple = tuple(l)

        conn = db_connect(self.db_path)
        with conn:
            cur = conn.cursor()
            cur.execute(cmd, cmd_tuple)
            new_id = cur.lastrowid
        return new_id


def insert_cmd(cols):
    cols_string = '(' + ','.join(cols) + ')'
    vals_string = '(' + ','.join(['?' for x in cols]) + ')'
    cmd = 'INSERT INTO results {} VALUES {}'.format(cols_string, vals_string)

    return cmd

def create_cmd(opts):
    cmd = "CREATE TABLE IF NOT EXISTS results(\nid integer PRIMARY KEY,\n"

    type2str = {float: 'real', int: 'integer', list: 'text', str: 'text',
                bool: 'bool'}

    opts_cols = []

    for k, v in opts:
        col = k
        opts_cols.append(col)
        type_is_list = isinstance(v.type, list)
        data_type = v.type if not type_is_list else list
        type_str = type2str[data_type]
        default = str(v.value)
        if type_str == 'text':
            default = '\'' + default + '\''
        if default == '\'None\'':
            default = 'NULL'

        s = '{} {} DEFAULT {}'.format(col, type_str, default)

        if not type_is_list or (type_is_list and None not in v.type):
            s += ' NOT NULL'
        s += ',\n'
        cmd += s

    res_cols = []
    for k, v in SAMPLE_RES_DICT.items():
        col = k
        res_cols.append(col)
        type_str = type2str[type(v)]
        s = '{} {} NOT NULL'.format(col, type_str)
        if k == 'pred_save_file':
            s += ' UNIQUE'
        s += ',\n'
        cmd += s


    cmd = cmd.strip(',\n') + '\n);'

    return cmd, opts_cols, res_cols
