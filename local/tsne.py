#!/usr/bin/env python3
import os

import numpy as np
from sklearn.manifold import TSNE
import argparse


def perform_tsne(args):
    data = np.load(args.data_path)
    label = np.load(args.label_path)
    data, label = preprocess(data, label, args)
    T = TSNE(n_components=args.n_components, init='pca', random_state=0, verbose=4, n_jobs=-1)
    print('performing TSNE')
    result = T.fit_transform(data)
    np.save(args.target_path, result)
    print(f'Successfully reduced dimention to {args.target_path}')


def preprocess(data, label, args):
    if args.sample_num == -1:
        print('Using all samples')
        return data, label
    idx = np.random.randint(len(label), size=(args.sample_num,))
    data_sampled, label_sampled = data[idx], label[idx]
    np.save(os.path.join(os.path.dirname(args.data_path), 'data_sampled.npy'), data_sampled)
    np.save(os.path.join(os.path.dirname(args.label_path), 'label_sampled.npy'), label_sampled)
    return data_sampled, label_sampled


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--label_path', type=str)
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--n_components', type=int, default=2, help="reduce to how many dimensions (2 or 3)")
    parser.add_argument('--sample_num', type=int, default=5000,
                        help="if set to -1, then use all data. Otherwise data_sampled.npy and label_sample.npy will be generated")
    args = parser.parse_args()
    perform_tsne(args)
