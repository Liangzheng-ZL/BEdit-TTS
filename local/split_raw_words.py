#!/usr/bin/env python3
import numpy as np
import scipy.io.wavfile as wf
import os
import json
import argparse
from tqdm import tqdm


def split_raw(args):
    f_shift, f_len = args.f_shift, args.f_len
    with open(args.json_path, 'r') as jf:
        info = json.load(jf)
    for utt in tqdm(info.keys()):
        os.mkdir(os.path.join(args.target_path, utt))
        fs, wav = wf.read(os.path.join(args.wav_path, utt + '.wav'))
        word_lens, dur = eval(info[utt]['word_lens']), eval(info[utt]['phone_dur'])
        word_dur = []
        end = 0
        for w in word_lens:
            start = end
            end += w
            word_dur.append(sum(dur[start:end]))
        assert end == len(dur), 'duration and word_lens mismatch'
        end = 0  # Now becomes the digital index
        for i, d in enumerate(word_dur):
            start = end
            end += f_shift * d
            this_word = wav[start:min(end + f_len - f_shift, len(wav))]
            wf.write(os.path.join(args.target_path, utt, str(i) + '.wav'), fs,
                     this_word)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_path', type=str)
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--json_path', type=str,
                        help="the output_info json which contains word_lens and phone duration")
    parser.add_argument("--f_shift", type=int, help='frame shift point', default=200)
    parser.add_argument("--f_len", type=int, help='frame length point', default=800)
    args = parser.parse_args()
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)
    split_raw(args)
