#!/bin/bash

alidir=$1
outdir=$2

[ ! -e $outdir  ] && mkdir -p $outdir

# get ali.txt and phones.txt
ali-to-phones --write-lengths=true $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz|" ark,t:$outdir/ali.txt || exit 1
cp $alidir/phones.txt $outdir || exit 1

# get text and phn_duration
local/ali2phn_and_dur.py $outdir/ali.txt $outdir/phones.txt $outdir/text.tmp $outdir/phn_duration
sort -u $outdir/text.tmp > $outdir/phn_text
rm $outdir/text.tmp

