import sys, os, time
import bbrsa

from bbrsa.bbrsa import ONMTRSAModel
from bbrsa.summarizers import ONMTSummarizer
from bbrsa.pragmatics import BasicPragmatics
from bbrsa.distractors import NextExampleDistractor, IdenticalDistractor

src_file = 'data/cnndm_val_first_manip2.txt'
res_file = 'monday_1000000.txt'

def main():
    sys.path.insert(0, os.path.abspath(bbrsa.ONMT_DIR))
    with open(src_file, 'r') as f:
        src = f.readlines()

    s0 = ONMTSummarizer()
    pragmatics = BasicPragmatics(alpha=5)
    distractor = NextExampleDistractor(batch_size=s0.default_batch_size)
    model = ONMTRSAModel(s0, pragmatics, distractor)

    print('----- Starting summary with distractor')
    start_time = time.time()
    pred1 = model.incremental_s1(src, beam_size=10, truncate=400)
    print('----- Finished summary. Duration:', time.time() - start_time)

    print('----- Starting summary without distractor')
    start_time = time.time()
    pred2 = model.summarize_s0(src, n_best=30, beam_size=30, truncate=400)
    print('----- Finished summary. Duration:', time.time() - start_time)
    for i, l in enumerate(pred2):
        print('#### Sentence', i, '####')
        for j, x in enumerate(l):
            print('('+str(j)+')', x)

    # with open(res_file, 'w') as f:
    #     for s, l1, l2 in zip(src, pred1, pred2):
    #         f.write('\n====================\n[SOURCE] ')
    #         f.write(s)
    #         f.write('\n--------------------\n[S1] ')
    #         f.write(l1[0])
    #         f.write('\n--------------------\n[S0] ')
    #         f.write(l2[0])

if __name__ == '__main__':
    main()
