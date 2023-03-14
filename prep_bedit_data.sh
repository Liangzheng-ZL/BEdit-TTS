#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


# general configuration
stage=-1
stop_stage=100
nj=16        # numebr of parallel jobs
verbose=1    # verbose option (if set > 1, get more log)
seed=1       # random seed number

# feature extraction related
fs=16000       # sampling frequency
fmax=7600      # maximum frequency
fmin=80        # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024     # number of fft points
n_shift=200    # number of shift points
win_length=800 # window length

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

dumpdir=hifitts_kaldi
dict=lexicon.txt
train_set=test

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "stage 0: Data preparation"
    ali_dir=kaldi/egs/hifitts/s5_16k/exp
    local/phn_data_prep.sh ${ali_dir}/tri5a_ali ${dumpdir}/${train_set}
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Ali-information to json"

    for x in ${train_set}; 
    do
        sort ${dumpdir}/${x}/phn_text -o ${dumpdir}/${x}/phn_text
        sort ${dumpdir}/${x}/phn_duration -o ${dumpdir}/${x}/phn_duration
        python ${dumpdir}/scripts/data2json_cs.py \
                --text ${dumpdir}/${x}/text \
                --phn_text ${dumpdir}/${x}/phn_text \
                --w2p_dict ${dict} \
                --phn_duration ${dumpdir}/${x}/phn_duration \
                --output_json ${dumpdir}/${x}/${x}.json 
    done

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "stage 2: Mask words in json data"
    python ${dumpdir}/scripts/random_word_mask.py \
            --json=${x}.json \
            --output=${x}_position.json    
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.\
    echo "stage 3: Json to Espnet addable json"
    python ${dumpdir}/scripts/json2add_json.py \
        --json=${x}_position.json \
        --output=${x}_add_position.json \
        --phn_output=${x}_add_phn_position.json
fi
