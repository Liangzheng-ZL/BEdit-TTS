import json
import argparse


def cat_json(json_files, out_json):
    json_datas = []

    for json_file in json_files:
        with open(json_file, 'r') as jfile:
            json_data = json.load(jfile)

        json_datas += json_data

    tmp = json.dumps(json_datas, indent=4)
    with open(out_json, 'w') as f:
        f.write(tmp)
        f.close

def main():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--json_files', 
                        type=list, 
                        default=["feats.1.dur.json","feats.2.dur.json","feats.3.dur.json","feats.4.dur.json","feats.5.dur.json","feats.6.dur.json","feats.7.dur.json","feats.8.dur.json","feats.9.dur.json","feats.10.dur.json",], 
                        help='text')
    # output
    parser.add_argument('--output_json', type=str, default='dur_outs.json', help='output json')
    args = parser.parse_args()

    json_files = args.json_files
    out_json = args.output_json

    cat_json(json_files, out_json)

if __name__=="__main__":
    main()