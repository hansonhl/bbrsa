import sys, os
import torch
import bbrsa
sys.path.append(os.path.abspath(bbrsa.ONMT_DIR))

from bbrsa.distractors import BertDistractor
from bbrsa.bbrsa import ONMTSummaryRSA
from bbrsa.summarizers import ONMTSummarizer

s0 = ONMTSummarizer(config_path='onmt_configs/giga.yml')
# pragmatics = BasicPragmatics(alpha=1)
distr = BertDistractor(batch_size=s0.default_batch_size)
# model = ONMTSummaryRSA(s0, pragmatics, distractor)

src = ['police arrested five anti-nuclear protesters thursday after they sought to disrupt loading of a french antarctic research and supply vessel , a spokesman for the protesters said .',
    'indian prime minister p.v. narasimha rao\'s promise of more autonomy for troubled kashmir and his plea for early state elections has sparked a violent reaction from provincial moslem and opposition parties .']

print(distr.generate(src, method='unmasked_surprisal', ensure_different=False))
