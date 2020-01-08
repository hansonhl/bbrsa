import rouge, argparse, sys
import numpy as np
from collections import defaultdict as dd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pred', type=argparse.FileType('r'))
    parser.add_argument('ref', type=argparse.FileType('r'))
    parser.add_argument('-s', '--src', type=argparse.FileType('r'))

    args = parser.parse_args()

    all_hypotheses = args.pred.read().splitlines()
    all_references = args.ref.read().splitlines()
    all_sources = None if args.src is None else args.src.read().splitlines()

    args.pred.close()
    args.ref.close()

    res = get_sorted(all_hypotheses, all_references, all_sources)

    print('num before filter:', len(res))
    res = res_filter(res)
    print('num after filter:', len(res))
    res = res[1000:1100]

    with open('results/cleaned_giga_1000-1100.txt', 'w') as file:
        fancy_display(file, res, start_rank=1)

def res_filter(res):
    def filter_func(r):
        return r['src'] != 'UNK' and word_overlap(r['src'], r['ref']) != 0 \
            and r['ref'] != 'afp world news summary'

    def word_overlap(s1, s2):
        s1_dict = dd(int)
        for wd in s1.split():
            s1_dict[wd] = 1
        s2_list = s2.split()
        total_words = len(s2_list)
        relevant_words = 0
        for wd in s2_list:
            if wd in s1_dict:
                relevant_words += 1
        return relevant_words / total_words

    return list(filter(filter_func, res))

def get_sorted(preds, refs, srcs):
    rouge_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                           max_n=3,
                           limit_length=True,
                           length_limit=100,
                           length_limit_type='words',
                           apply_avg=False,
                           apply_best=False,
                           alpha=0.5, # Default F1_score
                           weight_factor=1.2,
                           stemming=True)

    scores = rouge_evaluator.get_scores(preds, refs)
    all_f_scores = []
    for type in scores.keys():
        score_list = scores[type]
        f_scores = [x['f'] for x in score_list]
        all_f_scores.append(f_scores)
    all_f_scores = np.array(all_f_scores)
    mean_f_scores = np.mean(all_f_scores, axis=0)

    if srcs is not None:
        res = [{'mean_f': f.item(), 'pred': p, 'ref': r, 'src': s} \
               for f, p, r, s in zip(mean_f_scores, preds, refs, srcs)]
    else:
        res = [{'mean_f': f.item(), 'pred': p, 'ref': r} \
               for f, p, r in zip(mean_f_scores, preds, refs)]

    sorted_res = sorted(res, key=lambda x : x['mean_f'])
    return sorted_res

def fancy_display(file, res, start_rank=1):
    for i, item in enumerate(res):
        file.write('\n')
        if 'src' in item:
            file.write('SRC: ' + item['src'] + '\n')
        file.write('RANK: {}, MEAN_F SCORE: {:.4}\n'.format(i + start_rank, item['mean_f']))
        file.write('PRED: ' + item['pred'] + '\n')
        file.write('REFR: ' + item['ref'] + '\n')


if __name__ == '__main__':
    main()
