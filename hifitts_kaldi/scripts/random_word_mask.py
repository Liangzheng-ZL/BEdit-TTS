import argparse
import json
import random

def decide_words(words):
    eng_index = []
    for i in range(len(words)):
        if 'A' <= words[i][0] <= 'Z':
            eng_index.append(i)
    if len(eng_index)==0:
        return None, None
    else:
        return eng_index[0], eng_index[-1]-eng_index[0]+1
        



def mask_word(args):
    json_file = open(args.json)
    json_data = json.load(json_file)
    json_file.close

    position_json = []
    for utt_item in json_data:
        position_item = {}

        # words_num = len(utt_item["words"])
        # word_index = random.randint(0,words_num-1) 

        # mask_word_num = random.randint(1,1)
        # while (word_index+mask_word_num > words_num):
        #     mask_word_num -= 1
        word_index, mask_word_num = decide_words(utt_item["words"])

        if word_index is None:
            print(utt_item["uid"])
            continue

        phn_p1 = utt_item["ali_info"][word_index]["pronunciation"][0]["phn_index"]
        phn_p2 = utt_item["ali_info"][word_index+mask_word_num-1]["pronunciation"][-1]["phn_index"]+1


        # utt_len = utt_item["ali_info"][-1]["ef"]

        w_bf = utt_item["ali_info"][word_index]["bf"]
        w_ef = utt_item["ali_info"][word_index+mask_word_num-1]["ef"]

        p1 = w_bf  # 0 --- p1
        p2 = w_ef 
        # p2 = utt_len - w_ef # (utt_ef-p2) --- utt_ef

        mask_words = []
        for i in range(mask_word_num):
            mask_words.append(utt_item["words"][word_index + i])

        position_item["uid"] = utt_item["uid"]
        position_item["mask_word_index"] = word_index
        position_item["mask_word_num"] = mask_word_num
        position_item["mask_words"] = mask_words
        position_item["p1"] = p1
        position_item["p2"] = p2
        position_item["phn_p1"] = phn_p1
        position_item["phn_p2"] = phn_p2

        position_json.append(position_item)
    tmp = json.dumps(position_json, indent=4)
    with open(args.output, 'w') as f:
        f.write(tmp)
        f.close




def main():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--json', type=str, help='json file path')
    # output
    parser.add_argument('--output', type=str, help='output json')

    args = parser.parse_args()

    mask_word(args)



if __name__=="__main__":
    main()
