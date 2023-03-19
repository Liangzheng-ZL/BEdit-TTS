#!/bin/bash

# Copyright 2019 Nagoya University (Takenori Yoshimura)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


# general configuration
backend=pytorch
stage=-1
stop_stage=100
ngpu=1       # number of gpu in training
nj=24        # numebr of parallel jobs
verbose=1    # verbose option (if set > 1, get more log)
seed=1       # random seed number
resume=""    # the snapshot path to resume (if set empty, no effect)

# feature extraction related
fs=16000       # sampling frequency
fmax=7600      # maximum frequency
fmin=80        # minimum frequency
n_mels=80     # number of mel basis
n_fft=1024     # number of fft points
n_shift=200    # number of shift points
win_length=800 # window length

# config files
train_config=conf/train_bedit.yaml
decode_config=conf/decode.yaml

# decoding related
model=model.loss.best
n_average=1 # if > 0, the model averaged with n_average ckpts will be used instead of model.loss.best
griffin_lim_iters=64  # the number of iterations of Griffin-Lim

# Set this to somewhere where you want to put your data, or where
# someone else has already put it. You'll want to change this
# if you're not on the CLSP grid.
datadir=dataset/tts

# base url for downloads.
data_url=www.openslr.org/resources/60

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#raw_set=all_16k
#complete_set=phn_train_16k
train_set=train
eval_set=test_seen_samples

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    echo "stage -1: Data preparation"
    ali_dir=kaldi/egs/hifitts
    local/phn_data_prep.sh ${ali_dir}/exp/tri5a_ali data/train

fi

dumpdir=dump # directory to dump full features
feat_tr_dir=${dumpdir}/${train_set}; mkdir -p ${feat_tr_dir}
feat_ev_dir=${dumpdir}/${eval_set}; mkdir -p ${feat_ev_dir}
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to design training and dev name by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Feature Generation"

    fbankdir=fbank
    variancedir=variance
    make_fbank.sh --cmd "${train_cmd}" --nj ${nj} \
            --fs ${fs} \
            --fmax "${fmax}" \
            --fmin "${fmin}" \
            --n_fft ${n_fft} \
            --n_shift ${n_shift} \
            --win_length "${win_length}" \
            --n_mels ${n_mels} \
            data/${eval_set} \
            exp/make_fbank/${eval_set} \
            ${fbankdir}
    local/make_variance.sh --nj ${nj} data/${eval_set} exp/make_variance/${eval_set} ${variancedir}

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "stage 1: Apply CMVN"
    # compute statistics for global mean-variance normalization
   compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
   compute-cmvn-stats scp:data/${train_set}/variance.scp data/${train_set}/cmvn_variance.ark

    # dump features for training
   dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
       data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${train_set} ${feat_tr_dir}/mel
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/${eval_set} ${feat_ev_dir}/mel
    # dump variance
   dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
       data/${train_set}/variance.scp data/${train_set}/cmvn_variance.ark exp/dump_variance/${train_set} ${feat_tr_dir}/variance
    dump.sh --cmd "$train_cmd" --nj ${nj} --do_delta false \
        data/${eval_set}/variance.scp data/${train_set}/cmvn_variance.ark exp/dump_variance/${eval_set} ${feat_ev_dir}/variance
fi

dict=data/lang_1phn/${train_set}_units.txt
echo "dictionary: ${dict}"
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
   mkdir -p data/lang_1phn/
   echo "<unk> 1" > ${dict}  # <unk> must be 1, 0 will be used for "blank" in CTC
   text2token.py -s 1 -n 1 --trans_type phn data/train/text | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' > ${dict}.tmp
   echo "SPN" >> ${dict}.tmp
   sort -u ${dict}.tmp | awk '{print $0 " " NR+1}' >> ${dict}
   rm ${dict}.tmp
   wc -l ${dict}

    # make json labels
   data2json.sh --feat ${feat_tr_dir}/mel/feats.scp,${feat_tr_dir}/variance/feats.scp --trans_type phn \
        data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_ev_dir}/mel/feats.scp,${feat_ev_dir}/variance/feats.scp --trans_type phn \
        data/${eval_set} ${dict} > ${feat_ev_dir}/data.json
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "stage 3: Update Json"
    # Update json
    kaldi_dataf=hifitts_kaldi
    tmpdir=json_tmp
    mkdir -p ${tmpdir}
    for name in ${eval_set} ; do
        awk '{for(i=2;i<=NF;i++){print $i, NR-1}}' data/${name}/spk2utt | scp2json.py --key speaker_id > ${tmpdir}/speaker_id.json
        cat data/${name}/phn_duration | scp2json.py --key duration > ${tmpdir}/duration.json
        addjson.py --verbose 0 ${dumpdir}/${name}/data.json ${tmpdir}/duration.json > ${tmpdir}/tmp1.json
        addjson.py --is-input False --verbose 0 ${tmpdir}/tmp1.json ${kaldi_dataf}/${name}/frame_positions.json > ${tmpdir}/tmp2.json
        addjson.py --is-input False --verbose 0 ${tmpdir}/tmp2.json ${kaldi_dataf}/${name}/phone_positions.json > ${tmpdir}/tmp3.json
        addjson.py --is-input False --verbose 0 ${tmpdir}/tmp3.json ${tmpdir}/speaker_id.json | \
            sed -e 's/"name": "input2"/"name": "variance"/g' \
                -e 's/"name": "input3"/"name": "duration"/g' \
                -e 's/"name": "target2"/"name": "frame_positions"/g' \
                -e 's/"name": "target3"/"name": "phone_positions"/g' \
                -e 's/"name": "target4"/"name": "speaker_id"/g' \
            > ${dumpdir}/${name}/updated_data.json
    done
    rm -rf ${tmpdir}
fi

if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Text-to-speech model training"
    tr_json=${feat_tr_dir}/updated_data.json
    dt_json=${feat_ev_dir}/updated_data.json
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/results/train.log \
        tts_train.py \
           --backend ${backend} \
           --ngpu ${ngpu} \
           --outdir ${expdir}/results \
           --tensorboard-dir tensorboard/${expname} \
           --verbose ${verbose} \
           --seed ${seed} \
           --resume ${resume} \
           --train-json ${tr_json} \
           --valid-json ${dt_json} \
           --config ${train_config}
fi

if [ ${n_average} -gt 0 ]; then
    model=model.last${n_average}.avg.best
fi
outdir=${expdir}/outputs_${model}_$(basename ${decode_config%.*})
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    if [ ${n_average} -gt 0 ]; then
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${model} \
                               --num ${n_average}
    fi
    pids=() # initialize pids
    for name in test_seen_replaced_tmp; do
    (
        [ ! -e ${outdir}/${name} ] && mkdir -p ${outdir}/${name}
        cp ${dumpdir}/${name}/updated_data.json ${outdir}/${name}/data.json
        splitjson.py --parts ${nj} ${outdir}/${name}/data.json
        # decode in parallel
        ${train_cmd} JOB=1:${nj} ${outdir}/${name}/log/decode.JOB.log \
            tts_decode.py \
                --backend ${backend} \
                --ngpu 0 \
                --verbose ${verbose} \
                --out $PWD/${outdir}/${name}/feats.JOB \
                --json ${outdir}/${name}/split${nj}utt/data.JOB.json \
                --model ${expdir}/results/${model} \
                --config ${decode_config}

        # concatenate scp files
        for n in $(seq ${nj}); do
            cat "${outdir}/${name}/feats.$n.scp" || exit 1;
        done > ${outdir}/${name}/feats.scp
    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
fi