#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--utt-list', type=str, help='utterance list')
parser.add_argument('--input', type=str, help='input file path')
parser.add_argument('--output', type=str, default=False, help='output file path')
args = parser.parse_args()

with open(args.input, 'r') as f:
    input_data = f.readlines()

fr_output = open(args.output, "w")


with open(args.utt_list, 'r') as f:
    utt_lists = f.readlines()

uttids = []
for line in utt_lists:
    uttid = line.split()[0]
    uttids.append(uttid)

input_uttids = []
for line in input_data:
    input_uttid = line.split()[0]
    input_uttids.append(input_uttid)

for uid in uttids:
    index = input_uttids.index(uid)
    fr_output.write(input_data[index])
    


