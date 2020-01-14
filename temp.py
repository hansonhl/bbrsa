with open(out_file_name, 'w') as f:
    for distr_set in chunks(all_results, opts.bert_distr_d_factor):
        for s in distr_set:
            f.write(s+'\n')
        f.write('\n'
