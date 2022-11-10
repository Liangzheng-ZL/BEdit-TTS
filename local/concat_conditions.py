#!/usr/bin/env python3
import numpy as np
import os
import argparse
from tqdm import tqdm


def load_and_check(npy_path):
    array = np.load(npy_path)
    if len(array.shape)==1:
        print(f'Warning: {npy_path.split("/")[-1]} only has one word')
        array = array[np.newaxis, :]
    return array

    
def concat(args):
    conditions_dir = args.conditions_dir
    target = args.target_path
    if not os.path.exists(os.path.dirname(target)):
        print(f'Creating path {os.path.dirname(target)}')
        os.makedirs(os.path.dirname(target))
#    tmp = []
#    for f in tqdm(sorted(os.listdir(conditions_dir))):  # from small uttid to large (LJ010-0001 to LJ050-xxxx)
#        array = load_and_check(os.path.join(conditions_dir, f))
#        tmp.append(array)

    absolute_files = [os.path.join(conditions_dir, f) for f in sorted(os.listdir(conditions_dir))]
    tmp = map(load_and_check, absolute_files)
    print("Successful mapping array load")
    # data = np.concatenate(tqdm(tmp), axis=0)  # it's said that np.concatenate is slow
    data = np.vstack(tqdm(tmp))
    print(f'Concatenated npy shape is {data.shape}')
    np.save(target, data)


if __name__ == "__main__":
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--conditions_dir',
                        default="exp/phn_train_no_eval_pytorch_train_fastspeech/outputs_model.last1.avg.best_decode/phn_train_no_eval",
                        help="where every utt's prosody embedding is stored",
                        type=str,
                        required=True)
    Parser.add_argument('--target_path', help="where the concatenated npy is stored (dir+filename)",type=str, required=True)

    args = Parser.parse_args()
    concat(args)
    print('======> Done concatenating')

