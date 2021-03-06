import sys, os, time, logging
import bbrsa

from bbrsa.bbrsa import ONMTRSAModel
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.pragmatics import BasicPragmatics
from bbrsa.distractors import IdenticalDistractor, NextNDistractor, NextExampleDistractor
from bbrsa.utils import init_logger, display

src_file = 'data/cnndm_val_first5_src.txt'
tgt_file = 'data/cnndm_val_first5_tgt.txt'
res_file = 'results/5_articles.txt'

def main():
    sys.path.insert(0, os.path.abspath(bbrsa.ONMT_DIR))
    with open(src_file, 'r') as f:
        src = f.readlines()
    with open(tgt_file, 'r') as f:
        tgt = f.readlines()

    s0 = ONMTSummarizer()
    pragmatics = BasicPragmatics(alpha=2)
    distractor = NextNDistractor(batch_size=s0.default_batch_size, N=1)
    model = ONMTRSAModel(s0, pragmatics, distractor)

    print('----- Starting summary with distractor')
    start_time = time.time()
    pred1 = model.incremental_s1(src, beam_size=10, truncate=400)
    print('----- Finished summary. Duration:', time.time() - start_time)

    print('----- Starting summary without distractor')
    start_time = time.time()
    pred2 = model.summarize_s0(src, beam_size=10, truncate=400)
    print('----- Finished summary. Duration:', time.time() - start_time)

    display(['s1', 's0'], [pred1, pred2])
    #
    # s0_2 = ONMTSummarizer()
    # pragmatics = BasicPragmatics(alpha=2)
    # distractor = IdenticalDistractor(batch_size=s0.default_batch_size)
    # model = ONMTRSAModel(s0_2, pragmatics, distractor)
    # print('----- Starting summary with identical distractor')
    # start_time = time.time()
    # pred3 = model.summarize_s0(src, beam_size=10)
    # print('----- Finished summary. Duration:', time.time() - start_time)
    # print(pred3)





if __name__ == '__main__':
    main()
