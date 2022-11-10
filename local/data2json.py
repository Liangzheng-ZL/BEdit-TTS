import argparse
import json
import time



parser = argparse.ArgumentParser()
# input
parser.add_argument('--text', type=str, default='/mnt/lustre/sjtu/home/zl128/tools/espnet/egs/hifitts/fs2_bert_ywg/hifi_all_kaldi_data/text', help='text')
parser.add_argument('--phn_text', type=str, default='tts_bert_1/hifi_all_kaldi_data_16k_source/phn_eval_phn_text', help='phoneme text')
parser.add_argument('--w2p_dict', type=str, default='/mnt/lustre/sjtu/home/zl128/tools/espnet/egs/hifitts/fs2_bert_ywg/hifi_all_kaldi_data/length_sort_dict.txt', help='word to phonemes')
parser.add_argument('--phn_duration', type=str, default='tts_bert_1/hifi_all_kaldi_data_16k_source/phn_eval_phn_duration', help='phone duration')
# output
# parser.add_argument('--unsuccess_utts', type=str, default='unsuccess_utts', help='unsuccess_utts')
# parser.add_argument('--dataset_id', type=int, default=2, help='train_id=0 dev_id=1 test_id 2')
parser.add_argument('--output_json', type=str, default='tts_bert_1/hifi_all_kaldi_data_16k_source/phn_eval.json', help='output json')
# parser.add_argument('--tol_output_json', type=str, default='debug_sub_train_xl_tol_result.json', help='output json')
args = parser.parse_args()

# input
ftext = args.text
fphn_text = args.phn_text
fphn_duration = args.phn_duration
fdict = args.w2p_dict
# output
output_json_file = args.output_json

# frame
fs = 16000
n_shift = 200 # 12.5ms
win_length = 800 # 50ms


# read_time_1=time.time()
# read text
with open(ftext, "r") as f:
    texts = f.readlines()

# read pronunciation text
with open(fphn_text, "r") as f:
    phn_texts = f.readlines()

# read word to phone dictionary
with open(fdict, 'r') as f:
    w2ps = f.readlines()

# read phoneme duration
with open(fphn_duration, 'r') as f:
    phn_duras = f.readlines()

# read_time_2=time.time()
# print("read files total time: %s" %(read_time_2-read_time_1) )

words_list = []
words_phn_list = []
for line in w2ps:
    word,word_phn=line.split(" ",1)
    words_list.append(word)
    words_phn_list.append(word_phn)

textids = []
for line in texts:
    uid = line.split()[0]
    textids.append(uid)

total_utt_num = len(phn_texts)
assert total_utt_num == len(phn_duras)

# phone_ali={}
json_data = []
# for each utt
for i in range(total_utt_num):
    utt_item = {}

    phones_id, phon_line = phn_texts[i].split(" ", 1)
    dura_id, dura_line = phn_duras[i].split(" ", 1)
    assert phones_id == dura_id
    phones = phon_line.split()
    duration = dura_line.split()

    print(phones_id)
    # utt begin and end time
    u_bt = 0.0
    u_et = 0.0
    # utt begin and end frame
    u_bf = 0
    u_ef = len(duration)

    # utt words sequence
    textindex = textids.index(phones_id)
    _, text_line = texts[textindex].split(' ', 1)
    words_source = text_line.split()

    # get length of phonmeme of each word
    # w2p_time_1=time.time()

    utt_words_pron = []
    for word in words_source:
        word_pron = {}
        cnt = 0
        tmp_length = 0
        if (word in words_list):
            if len(word_pron) != 0 and len(d_pron) >= tmp_length:
                continue
            word_index = words_list.index(word)
            d_pron = words_phn_list[word_index].split()
            word_pron["word"] = word
            word_pron["phoneme"] = d_pron 
            word_pron["phoneme_length"] = len(d_pron)
            tmp_length = len(d_pron)
        if (word not in words_list) and len(word_pron) == 0:
            word_pron["word"] = word
            word_pron["phoneme"] = ['SPN']
            word_pron["phoneme_length"] = 1

        utt_words_pron.append(word_pron)
    # w2p_time_2=time.time()
    # print("find word's phoneme: %s" %(w2p_time_2-w2p_time_1) )
    ##

    assert len(phones)==len(duration)
    utt_phones = []
    next_dt = u_bt
    next_df = u_bf
    # for each phoneme
   
    for j in range(len(phones)):
        pron_js = {}
        # remove silence
        if phones[j] in ['sil', 'sil0', 'sil1', 'sil2', 'sil3', 'SIL','SIL0', 'SIL1', 'SIL2', 'SIL3']:
            next_dt = next_dt + (float(duration[j]) * n_shift) / fs
            next_df = next_df + int(duration[j])
            continue
        pron_js["phoneme"] = phones[j]
        pron_js["bt"] = next_dt
        pron_js["bf"] = next_df
        next_dt = next_dt + (float(duration[j]) * n_shift ) / fs
        next_df = next_df + int(duration[j])
        pron_js["et"] = next_dt + float(win_length-n_shift) / fs
        pron_js["ef"] = next_df
        pron_js["phn_index"] = j
        utt_phones.append(pron_js)
    # phn_time_2=time.time()
    # print("process phoneme's alignment: %s" %(phn_time_2-phn_time_1) )

    words_ali = []
    word_b = 0
    word_bf = 0
    # word_time_1=time.time()
    for k in range(len(utt_words_pron)):
        word_e = word_b+utt_words_pron[k]["phoneme_length"]-1
        word_ali = {}
        word_ali["word"] = utt_words_pron[k]["word"]
        word_ali["bt"] = utt_phones[word_b]["bt"]
        word_ali["et"] = utt_phones[word_e]["et"]
        word_ali["bf"] = utt_phones[word_b]["bf"]
        word_ali["ef"] = utt_phones[word_e]["ef"]
        word_ali["pronunciation"] = utt_phones[word_b:word_b+utt_words_pron[k]["phoneme_length"]]
        word_b += utt_words_pron[k]["phoneme_length"]
        words_ali.append(word_ali)

    utt_item["uid"] = phones_id
    utt_item["text"] = text_line.split('\n')[0]
    utt_item["words"] = words_source
    utt_item["ali_info"] = words_ali
    json_data.append(utt_item)
    # json_data["audios"][dataset_id]["segments"][index]["ali_info"] = words_ali
    # phone_ali["segments"].append(json_data["audios"][dataset_id]["segments"][index])
tmp = json.dumps(json_data, indent=4)
with open(output_json_file, 'w') as f:
    f.write(tmp)
    f.close
    # word_time_2=time.time()
    # print("process word's alignment: %s" %(word_time_2-word_time_1) )

