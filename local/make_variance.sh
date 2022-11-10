#!/bin/bash

nj=4
cmd=run.pl

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

data=$1
logdir=$2
variance_dir=$3

mkdir -p $variance_dir || exit 1;
mkdir -p $logdir || exit 1;

name=$(basename ${data})
scp=${data}/wav.scp
split_scps=""
for n in $(seq $nj); do
  split_scps="$split_scps $logdir/wav.$n.scp"
done
utils/split_scp.pl ${scp} $split_scps || exit 1;

pitch_feats="ark:compute-kaldi-pitch-feats --verbose=2 --config=conf/pitch.conf scp,p:$logdir/wav.JOB.scp ark:- | process-kaldi-pitch-feats --add-raw-log-pitch=true ark:- ark:- |"
energy_feats="ark:compute-mfcc-feats --config=conf/mfcc.conf --use-energy=true scp,p:$logdir/wav.JOB.scp ark:- | select-feats 0 ark:- ark:- |"

$cmd JOB=1:$nj ${logdir}/make_variance.JOB.log \
  paste-feats --length-tolerance=2 "$pitch_feats" "$energy_feats" ark,scp:$variance_dir/variance_${name}.JOB.ark,$variance_dir/variance_${name}.JOB.scp \
   || exit 1;

# concatenate the .scp files together.
for n in $(seq $nj); do
  cat $variance_dir/variance_${name}.$n.scp || exit 1;
done | sort > $data/variance.scp
