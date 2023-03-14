#!/usr/bin/env python3
import scipy.io.wavfile as wf
import argparse
import os
import shutil
import numpy as np
import json
from tqdm import tqdm


def split_idx(args):
    f_shift = args.frame_shift
    f_len = args.frame_len
    if not os.path.exists(args.target_path):
        os.mkdir(args.target_path)
    else:
        shutil.rmtree(args.target_path)
        os.mkdir(args.target_path)
    for i in range(args.num_gaussian):
        os.mkdir(os.path.join(args.target_path, str(i)))
    num_each = np.zeros(args.num_gaussian)
    with open(os.path.join(args.output_js_path, 'output_info.json'), 'r') as f:
        jf = json.load(f)
    for id in tqdm(jf.keys()):
        indices, word_lens, dur = eval(jf[id]['idx_info']), eval(jf[id]['word_lens']), eval(jf[id]['phone_dur'])
        word_dur = []
        end = 0
        for w in word_lens:
            start = end
            end += w
            word_dur.append(sum(dur[start:end]))
        assert end == len(dur), 'duration and word_lens mismatch'
        fs, wav = wf.read(os.path.join(args.wav_path,  id + '.wav'))
        end = 0  # Now becomes the digital index
        for i, d in enumerate(word_dur):
            start = end
            end += f_shift * d
            if d <= 17:
                continue
            this_word = wav[start:min(end + f_len - f_shift, len(wav))]
            this_idx = indices[i]
            wf.write(os.path.join(args.target_path, str(this_idx), str(int(num_each[this_idx]))+'.wav'), fs, this_word)
            num_each[this_idx] += 1
        print(id+' Done')
    for i in range(args.num_gaussian):
        if os.listdir(os.path.join(args.target_path, str(i))):
            root = os.path.join(args.target_path, str(i))
            res = []
            for w in tqdm(os.listdir(root)):
                fs, f = wf.read(os.path.join(root, w))
                res.append(f)
                res.append(np.zeros(2500, dtype=np.int16))
            res = np.concatenate(res)
            wf.write(os.path.join(args.target_path,str(i)+'.wav'), fs, res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frame_shift', type=int, default=200)
    parser.add_argument('--frame_len', type=int, default=800)
    parser.add_argument('--wav_path', type=str,
                        default='/home/cantabile-kwok/code/word_level/exp/phn_train_no_eval_pytorch_train_fastspeech/outputs_model.last1.avg.best_decode_denorm/phn_eval/seed1729_k3_rescale1')
    parser.add_argument('--output_js_path', type=str,
                        default="/home/cantabile-kwok/code/word_level/exp/phn_train_no_eval_pytorch_train_fastspeech/outputs_model.last1.avg.best_decode/phn_eval/output_info",
                        help='does not include json file name')
    parser.add_argument('--target_path', type=str,
                        default='/home/cantabile-kwok/code/word_level/exp/phn_train_no_eval_pytorch_train_fastspeech/outputs_model.last1.avg.best_decode_denorm/phn_eval/split_words')
    parser.add_argument('--num_gaussian', type=int, default=20)
    args = parser.parse_args()
    split_idx(args)
