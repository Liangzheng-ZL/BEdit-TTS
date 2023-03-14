#!/usr/bin/env python3

import numpy as np
import argparse
import os
import json
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition_path', type=str)
    parser.add_argument('--idx_json_path', type=str,)
    parser.add_argument('--target_path', type=str,)
    args = parser.parse_args()
    if not os.path.exists(args.target_path):
        os.makedirs(args.target_path)
    with open(args.idx_json_path, 'r') as jf:
        info = json.load(jf)
    labels = []
    data = []
    for utt in tqdm(info.keys()):
        if not os.path.exists(os.path.join(args.condition_path, utt + '.npy')):
            continue
        labels += eval(info[utt]['idx_info'])
        utt_data = np.load(os.path.join(args.condition_path, utt + '.npy'))
        data.append(utt_data)
    labels = np.array(labels, dtype=np.int16)
    data = np.concatenate(data, axis=0)
    np.save(os.path.join(args.target_path, 'labels.npy'), labels)
    np.save(os.path.join(args.target_path, 'conditions.npy'), data)
