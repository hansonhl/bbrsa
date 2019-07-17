import sys, os, time

from bbrsa import ONMTSummaryRSA

from models import ONMTSummarizer
from pragmatics import NextExampleDistractor, BasicPragmatics

src_file = 'data/cnndm_val_first5_src.txt'
tgt_file = 'data/cnndm_val_first5_tgt.txt'
res_file = 'res1.txt'
ONMT_DIR = '../myOpenNMT'

def main():
    sys.path.insert(0, os.path.abspath(ONMT_DIR))
    with open(src_file, 'r') as f:
        src = f.readlines()
    with open(tgt_file, 'r') as f:
        tgt = f.readlines()

    s0 = ONMTSummarizer()
    pragmatics = BasicPragmatics(alpha=1)
    distractor = NextExampleDistractor(batch_size=s0.opt.batch_size)
    model = ONMTSummaryRSA(s0, pragmatics, distractor)

    print('----- Starting summary with distractor')
    start_time = time.time()
    pred1 = model.summarize_with_distractor(src, beam_size=5)
    print('----- Finished summary. Duration:', time.time() - start_time)
    print(pred1)

    print('----- Starting summary without distractor')
    start_time = time.time()
    pred2 = model.summarize_with_s0(src, beam_size=5)
    print('----- Finished summary. Duration:', time.time() - start_time)
    print(pred2)

    with open(res_file, 'w') as f:
        for s, l1, l2 in zip(tgt, pred1, pred2):
            f.write('\n====================\n[REFERENCE] ')
            f.write(s)
            f.write('\n--------------------\n[PRAG] ')
            f.write(l1[0])
            f.write('\n--------------------\n[BASE] ')
            f.write(l2[0])



if __name__ == '__main__':
    main()
