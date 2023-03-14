import argparse
import json
import random


def turn_to_add_json(args):
    json_file = open(args.json)
    json_data = json.load(json_file)
    json_file.close

    new_json = {}
    new_json["utts"] = {}
    for utt_item in json_data:

        # new_item = {}
        position = []
        position.append(utt_item["p1"])
        position.append(utt_item["p2"])

        new_json["utts"][utt_item["uid"]] = {}
        new_json["utts"][utt_item["uid"]]["position"] = position

    tmp = json.dumps(new_json, indent=4)
    with open(args.output, 'w') as f:
        f.write(tmp)
        f.close


def turn_to_add_json_phn(args):
    json_file = open(args.json)
    json_data = json.load(json_file)
    json_file.close

    new_json = {}
    new_json["utts"] = {}
    for utt_item in json_data:

        # new_item = {}
        position = []
        position.append(utt_item["phn_p1"])
        position.append(utt_item["phn_p2"])

        new_json["utts"][utt_item["uid"]] = {}
        new_json["utts"][utt_item["uid"]]["phn_position"] = position

    tmp = json.dumps(new_json, indent=4)
    with open(args.phn_output, 'w') as f:
        f.write(tmp)
        f.close




def main():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--json', type=str, default='/mnt/lustre/sjtu/home/zl128/tools/espnet/egs/hifitts/fs2_bert_ywg/hifi_all_kaldi_data/phn_eval_position.json', help='json file path')
    # output
    parser.add_argument('--output', type=str, default='add_position.json', help='output json')
    parser.add_argument('--phn_output', type=str, default='add_phn_position.json', help='output json')

    args = parser.parse_args()

    turn_to_add_json(args)
    turn_to_add_json_phn(args)



if __name__=="__main__":
    main()