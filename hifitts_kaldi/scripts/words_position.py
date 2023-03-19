import argparse
import json
import random


def mask_word(args):
    json_file = open(args.json)
    json_data = json.load(json_file)
    json_file.close

    position_json = []
    for utt_item in json_data:
        position_item = {}

        words_num = len(utt_item["words"])
        word_index = random.randint(0,words_num-1) 

        mask_word_num = random.randint(1,1)
        while (word_index+mask_word_num > words_num):
            mask_word_num -= 1

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
    parser.add_argument('--json', type=str, default='phn_eval.json', help='json file')
    # output
    parser.add_argument('--phone_output', type=str, default='phone_positions.json', help='output phones position json')
    parser.add_argument('--frame_output', type=str, default='frame_positions.json', help='output frames position json')

    args = parser.parse_args()

    json_file = open(args.json)
    json_data = json.load(json_file)
    json_file.close

    phn_json = {}
    frame_json = {}
    phn_json["utts"] = {}
    frame_json["utts"] = {}


    for utt_item in json_data:
        utt_phn_position_item = {}
        utt_frame_position_item = {}

        phn_positions = []
        frame_positions = []
        for word_item in utt_item["ali_info"]:
            word_phn_position_item = []
            word_frame_position_item = []

            word_phn_position_item.append(word_item["pronunciation"][0]["phn_index"])
            word_phn_position_item.append(word_item["pronunciation"][-1]["phn_index"] + 1)

            word_frame_position_item.append(word_item["bf"])
            word_frame_position_item.append(word_item["ef"])

            phn_positions.append(word_phn_position_item)
            frame_positions.append(word_frame_position_item)

        utt_phn_position_item["phone_positions"] = phn_positions
        utt_frame_position_item["frame_positions"] = frame_positions

        uid = utt_item["uid"]
        phn_json["utts"][uid] = utt_phn_position_item
        frame_json["utts"][uid] = utt_frame_position_item
  
    tmp_phn = json.dumps(phn_json, indent=4)
    with open(args.phone_output, 'w') as f1:
        f1.write(tmp_phn)
        f1.close
    
    tmp_frame = json.dumps(frame_json, indent=4)
    with open(args.frame_output, 'w') as f2:
        f2.write(tmp_frame)
        f2.close


if __name__=="__main__":
    main()
