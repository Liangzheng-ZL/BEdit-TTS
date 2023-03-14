#!/usr/bin/env python3
import numpy as np
from sklearn.mixture import GaussianMixture
import argparse
import os


def perform_gmm(args):
    data = np.load(args.data_path)
    gm = GaussianMixture(n_components=args.n_components, random_state=0,
                         verbose=2,
                         covariance_type='full',).fit(data)
    pred = gm.predict(data)
    np.save(args.target_path, pred)


if __name__ == "__main__":
    # This script takes the extracted prosody embeddings and perform GMM clustering
    # Then outputs the predicted cluster label (n_samples, ) as args.target_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    # parser.add_argument('--label_path', type=str)
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--n_components', type=int, default=8)
    # parser.add_argument('--sample_num', type=int, default=5000)
    args = parser.parse_args()
    perform_gmm(args)
