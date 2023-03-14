#!/bin/bash

alidir=$1
outdir=$2

[ ! -e $outdir  ] && mkdir -p $outdir

# get ali.txt and phones.txt
ali-to-phones --write-lengths=true $alidir/final.mdl "ark:gunzip -c $alidir/ali.*.gz|" ark,t:$outdir/ali.txt || exit 1
cp $alidir/phones.txt $outdir || exit 1

# get text and phn_duration
awk '{printf("%s", $1); for(i=2;i<=NF;i++){if((i-1)%3==1){printf(" %s", $i)}}; printf("\n")}' $outdir/ali.txt | ./utils/int2sym.pl -f 2- $alidir/phones.txt | sed 's/_[A-Za-z] / /g' > $outdir/text
awk '{printf("%s", $1); for(i=2;i<=NF;i++){if((i-1)%3==2){printf(" %s", $i)}}; printf("\n")}' $outdir/ali.txt | sort > $outdir/phn_duration
