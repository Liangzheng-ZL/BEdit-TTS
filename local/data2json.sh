#!/bin/bash

data_set=$1
text=$2
dict=$3

echo "sort phoneme text and phoneme duration"
sort ${data_set}/phn_text -o ${data_set}/phn_text
sort ${data_set}/phn_duration -o ${data_set}/phn_duration
# data2json
echo "aligment data to json data"
python local/data2json.py \
    --text=$text \
    --phn_text=${data_set}/text \
    --w2p_dict=$dict \
    --phn_duration=${data_set}/phn_duration \
    --output_json=${data_set}/data.json 
# randomly mask words
echo "randomly mask words in json data"
python local/random_word_mask.py \
    --json=${data_set}/data.json \
    --output=${data_set}/position.json
# normal json to addable json
echo "turn normal json data to espnet addable json"
python local/json2add_json.py \
    --json=${data_set}/position.json \
    --output=${data_set}/add_position.json \
    --phn_output=${data_set}/add_phn_position.json

