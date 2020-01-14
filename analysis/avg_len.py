file = "/home/hansonlu/links/data/2031val/2031val_1015_002551_acc36.93_rglf40.91.txt"

avglen = 0
num_sentences = 0
with open(file, "r") as f:
    for s in f:
        avglen += len(s.split())
        num_sentences += 1
print("avglen=", avglen/num_sentences)
