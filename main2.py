import sys, os, time

from bbrsa import ONMTSummaryRSA

from models import ONMTSummarizer
from pragmatics import NextExampleDistractor, BasicPragmatics
from pragmatics import IdenticalDistractor, NextNDistractor

src_file = 'data/cnndm_val_first5_src.txt'
tgt_file = 'data/cnndm_val_first5_tgt.txt'
res_file = 'results/5_articles.txt'
ONMT_DIR = '../myOpenNMT'

def main():
    sys.path.insert(0, os.path.abspath(ONMT_DIR))
    with open(src_file, 'r') as f:
        src = f.readlines()
    with open(tgt_file, 'r') as f:
        tgt = f.readlines()

    s0 = ONMTSummarizer()
    pragmatics = BasicPragmatics(alpha=2)
    distractor = NextNDistractor(batch_size=s0.opt.batch_size, N=2)
    model = ONMTSummaryRSA(s0, pragmatics, distractor)

    print('----- Starting summary with distractor')
    start_time = time.time()
    pred1 = model.summarize_with_distractor(src, beam_size=3)
    print('----- Finished summary. Duration:', time.time() - start_time)
    print(pred1)

    # print('----- Starting summary without distractor')
    # start_time = time.time()
    # pred2 = model.summarize_with_s0(src, beam_size=10)
    # print('----- Finished summary. Duration:', time.time() - start_time)
    # print(pred2)
    #
    # s0_2 = ONMTSummarizer()
    # pragmatics = BasicPragmatics(alpha=2)
    # distractor = IdenticalDistractor(batch_size=s0.opt.batch_size)
    # model = ONMTSummaryRSA(s0_2, pragmatics, distractor)
    # print('----- Starting summary with identical distractor')
    # start_time = time.time()
    # pred3 = model.summarize_with_s0(src, beam_size=10)
    # print('----- Finished summary. Duration:', time.time() - start_time)
    # print(pred3)

    # with open(res_file, 'w') as f:
    #     for s, l1, l2, l3 in zip(tgt, pred1, pred3, pred2):
    #         f.write('\n====================\n[REFR] ')
    #         f.write(s)
    #         f.write('\n--------------------\n[NEXT] ')
    #         f.write(l1[0])
    #         f.write('\n--------------------\n[IDEN] ')
    #         f.write(l2[0])
    #         f.write('\n--------------------\n[BASE] ')
    #         f.write(l3[0])



if __name__ == '__main__':
    main()
