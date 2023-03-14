#!/usr/bin/env python3

import sys
fr_ali, fr_phn, fw_text, fw_duration = sys.argv[1:]
fr_ali = open(fr_ali, 'r')
fr_phn = open(fr_phn, 'r')
fw_text = open(fw_text, 'w')
fw_duration = open(fw_duration, 'w')

phnid2phn = {}
for line in fr_phn.readlines():
    phn, phnid = line.strip().split()
    phnid2phn[phnid] = phn

for line in fr_ali.readlines():
    space_idx = line.find(" ")
    uttid = line[:space_idx]
    phnid_dur_seq = line[space_idx+1:-1].split(" ; ")
    phn_seq = []
    dur_seq = []
    for phnid_dur in phnid_dur_seq:
        phnid, dur = phnid_dur.split()
        phn = phnid2phn[phnid].split("_")[0]
        if phn == 'sil':
            if int(dur) <= 5:
                phn = 'SIL0'
            elif int(dur) <= 10:
                phn = 'SIL1'
            elif int(dur) <= 20:
                phn = 'SIL2'
            else:
                phn = 'SIL3'
        phn_seq.append(phn)
        dur_seq.append(dur)
    fw_text.write(uttid + " " + " ".join(phn_seq) + "\n")
    fw_duration.write(uttid + " " + " ".join(dur_seq) + "\n")
