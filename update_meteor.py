# add new meteor scores to database
import sqlite3
import pandas as pd
import bbrsa
import argparse
from subprocess import check_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_list_path', type=str)
    parser.add_argument('-r', '--ref_path', type=str,
                        default="/home/hansonlu/links/data/giga/valid.rsa.tgt.txt")
    parser.add_argument('-d', '--db_path', type=str, default="tables/2031res.db")
    parser.add_argument('-j', '--meteor_jar_path', type=str,
                        default="analysis/METEOR/meteor-1.5/meteor-1.5.jar")
    args = parser.parse_args()

    file_list_path = args.file_list_path
    ref_path = args.ref_path
    db_path = args.db_path
    meteor_jar_path = args.meteor_jar_path

    if file_list_path:
        with open(file_list_path, "r") as f:
            files = [s.strip() for s in f.readlines()]
    else:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        cmd = "SELECT pred_save_file FROM results WHERE METEOR IS NULL"
        c.execute(cmd)
        rows = c.fetchall()
        files = [row[0] for row in rows]
        conn.close()

    print("Going to evaluate the following files:")
    print(files)

    res = []

    for pred_path in files:
        pred_path = pred_path.strip()
        print("Evaluating " + pred_path)
        cmd = "java -Xmx2G -jar %s %s %s -l en -norm \
    | tail -n 1 | tr -dc '.0-9'" % (meteor_jar_path, pred_path, ref_path)
        meteor_score = float(check_output(cmd, shell=True))
        # need shell=True to accept piped commands
        res.append((pred_path, meteor_score))

    conn = sqlite3.connect(db_path)

    for path, meteor_score in res:
        cmd = "UPDATE results SET METEOR = {} WHERE pred_save_file = \"{}\";".format(
            meteor_score, path)
        print(cmd)
        c = conn.cursor()
        c.execute(cmd)

    conn.commit()
    conn.close()

# create new column in table
def add_new_column(db_path, name, type):
    new_col_cmd = "ALTER TABLE results ADD " + name + " " + type + ";"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(new_col_cmd)
    conn.close()

def create_cmd(opts):
    cmd = "CREATE TABLE IF NOT EXISTS new_results(\nid integer PRIMARY KEY,\n"

    type2str = {float: 'real', int: 'integer', list: 'text', str: 'text',
                bool: 'bool'}

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
        'rouge_lf': 0.,
        'METEOR': 0.
    }

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

def revert_to_original_columns(db_path):
    test_opts = bbrsa.DEFAULT_OPTS.clone()
    test_opts.set_as_default({'beam_size': 10,
                              'batch_size': 32,
                              'prag_alpha': 1.,
                              'mode': 'incr_s1',
                              'gpu': True,
                              'shard_size': 2000,
                              'distractor': 'bert',
                              'bert_distr_no_subword_repl': True,
                              'bert_distr_d_factor': 2})

    begin_cmd = "BEGIN TRANSACTION;"
    crt_cmd, opts, res = create_cmd(test_opts)
    colnames = ', '.join(opts + res)
    print(len(opts+res))
    copy_cmd = "INSERT INTO new_results(" + colnames + ") SELECT " + colnames + " FROM results;"
    drop_cmd = "DROP TABLE IF EXISTS results;"
    alter_cmd = "ALTER TABLE new_results RENAME TO results;"
    commit_cmd = "COMMIT;"

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(begin_cmd)
    # c.execute(crt_cmd)
    # c.execute(copy_cmd)
    c.execute(drop_cmd)
    c.execute(alter_cmd)
    c.execute(commit_cmd)

    conn.close()

# revert_to_original_columns(db_path)
# add_new_column(db_path, 'METEOR', 'real')

if __name__ == '__main__':
    main()
