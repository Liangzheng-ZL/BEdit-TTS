import json
import argparse


def json2utt2frames(json_file, file_out):
    fr_out = open(file_out, 'w')
    with open(json_file, 'r') as jfile:
        json_data = json.load(jfile)

    for item in json_data:
        fr_out.write(item["utt_id"]+' '+str(item["frames"])+'\n')

def main():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--input_json', 
                        type=str, 
                        default="dur_outs.json", 
                        help='input duration outs json')
    # output
    parser.add_argument('--output', type=str, default='utt2num_frames', help='output json')
    args = parser.parse_args()
    json2utt2frames(args.input_json, args.output)

if __name__=="__main__":
    main()