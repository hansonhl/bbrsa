import numpy as np
from subprocess import check_output

ART_FILE = "../data/valid.rsa.art.txt"
TGT_FILE = "../data/valid.rsa.tgt.txt"

# s0 output
LIT_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1023_173511_acc26.59_rglf42.62.txt"#10
#"/home/hansonlu/links/data/2031val/2031val_1023_172927_acc34.81_rglf42.62.txt"#5
#"/home/hansonlu/links/data/2031val/2031val_1023_172437_acc55.74_rglf42.62.txt"#2

# alpha = 0.5
LOW_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1013_140725_acc55.24_rglf42.58.txt"#2
#"/home/hansonlu/links/data/2031val/2031val_1014_212030_acc35.11_rglf42.3.txt"#5
#"/home/hansonlu/links/data/2031val/2031val_1018_005633_acc26.29_rglf42.02.txt"#10

# alpha = 1.5
MID_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1015_002551_acc36.93_rglf40.91.txt"#5
#"/home/hansonlu/links/data/2031val/2031val_1018_034120_acc30.28_rglf38.66.txt"#10
#"/home/hansonlu/links/data/2031val/2031val_1013_153233_acc56.43_rglf42.02.txt"#2

# alpha = 3
HI_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1023_192339_acc58.84_rglf40.79.txt"
#"/home/hansonlu/links/data/2031val/2031val_1023_222423_acc43.13_rglf34.75.txt"#5
#"/home/hansonlu/links/data/2031val/2031val_1030_010438_acc37.81_rglf27.11.txt"#10

S0_B20_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1106_150510_acc35.3_rglf42.27.txt"

# alpha = 1.5, beam=20
S1_B20_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1110_133158_acc39.19_rglf40.57.txt"#5
#"/home/hansonlu/links/data/2031val/2031val_1106_005815_acc33.04_rglf38.44.txt"#10

GROW_P1_B10_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1106_214353_acc35.99_rglf41.63.txt"#5

GROW_P1_B20_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1106_222500_acc35.45_rglf41.39.txt"#5

MEMOIZED_L1_B10_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1109_235527_acc36.24_rglf41.89.txt"#5

MEMOIZED_L1_B20_FILE = \
"/home/hansonlu/links/data/2031val/2031val_1110_010214_acc42.49_rglf34.77.txt"#5

readable = True
sample = 100
OUT_FILE = "memo_l1/memol1_qual_sample_Nov10_4distr.txt"

examples = []

# files = [open(ART_FILE, "r"),
#          open(TGT_FILE, "r"),
#          open(LIT_FILE, "r"),
#          open(LOW_FILE, "r"),
#          open(MID_FILE, "r"),
#          open(HI_FILE, "r")]
# columns = ['source','gold','s0','alpha0.5','alpha1.5','alpha3']
# files = [open(ART_FILE, "r"),
#          open(TGT_FILE, "r"),
#          open(LIT_FILE, "r"),
#          open(S0_B20_FILE, "r"),
#          open(MID_FILE, "r"),
#          open(S1_B20_FILE, "r")]
# columns = ['source','gold','s0_beam10','s0_beam20','s1_beam10','s1_beam20']
files = [open(ART_FILE, "r"),
         open(TGT_FILE, "r"),
         open(LIT_FILE, "r"),
         open(MID_FILE, "r"),
         open(S1_B20_FILE, "r"),
         open(MEMOIZED_L1_B10_FILE, "r"),
         open(MEMOIZED_L1_B20_FILE, "r")]
columns = ['      source', '        gold','      s0_b10','s1_basic_b10','s1_basic_b20',' s1_memo_b10',' s1_memo_b20']

assert len(files) == len(columns)

if sample:
    num_lines = int(check_output(['wc', '-l', TGT_FILE]).split()[0])
    idxs = np.random.choice(num_lines, sample, replace=False).tolist()
    idxs.sort(reverse=True)

with open(OUT_FILE, "w") as f:
    if readable:
        f.write('\n'.join(columns) + '\n\n')
    else:
        f.write('\t'.join(columns) + '\n')
    for i, ss in enumerate(zip(*files)):
        if sample:
            if len(idxs) > 0 and i == idxs[-1]: idxs.pop()
            else: continue
        ss = [s.rstrip() for s in ss]
        if readable:
            f.write('\n'.join([tag + ': ' + s for tag, s in
                               zip(columns, ss) ]) + '\n\n')
        else:
            f.write('\t'.join(ss) + '\n')
    # for src, gold, s0, low, mid, hi in zip(art_f, tgt_f, lit_f, low_f, mid_f, hi_f):
    #     if CSV:
    #         f.write("%s,\n%s,\n%s,\n%s,\n%s,\n%s\n\n"
    #                 % (src.rstrip(), gold.rstrip(), s0.rstrip(), low.rstrip(), mid.rstrip(), hi.rstrip()))
    #     else:
    #         f.write("%s\t%s\t%s\t%s\t%s\t%s\n"
    #                 % (src.rstrip(), gold.rstrip(), s0.rstrip(), low.rstrip(), mid.rstrip(), hi.rstrip()))

for f in files:
    f.close()
