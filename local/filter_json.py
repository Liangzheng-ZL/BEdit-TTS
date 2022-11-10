#!/usr/bin/env python3

###################### cpd modified ######################

import json
import argparse
from espnet.utils.cli_utils import strtobool

parser = argparse.ArgumentParser()
parser.add_argument('--utt-list', type=str, help='utterance list')
parser.add_argument('--input', type=str, help='input json file path')
parser.add_argument('--reverse', type=strtobool, default=False, help='output noise utterances or clean utterances')
args = parser.parse_args()

with open(args.input, 'rb') as f:
    input_json = json.load(f)['utts']

keep_utterances = set()
with open(args.utt_list, 'r')as f:
    for line in f.readlines():
        keep_utterances.add(line.split(maxsplit=1)[0])

keep_output_json = {}
discard_output_json = {}
for uttid, info in input_json.items():
    if uttid in keep_utterances:
        keep_output_json[uttid] = info
    else:
        discard_output_json[uttid] = info

if not args.reverse:
    output_json = keep_output_json
else:
    output_json = discard_output_json

jsonstring = json.dumps({'utts': output_json}, indent=4, ensure_ascii=False,
                        sort_keys=True, separators=(',', ': '))
print(jsonstring)
