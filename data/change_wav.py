import sys
import os
src_wav = sys.argv[1]


tgt_lines = []
with open(src_wav, 'r') as f:
    for l in f.readlines():
        utt, path = l.strip().split()
        wavname = os.path.basename(path)
        tgt_lines.append(f"{utt} /mnt/lustre/sjtu/home/ywg12/dataset/HiFiTTS_8k_small/{wavname}\n")

with open(src_wav, 'w') as f:
    for l in tgt_lines:
        f.write(l)


