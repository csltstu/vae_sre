#!/bin/bash

nj=40
cmd="run.pl"
stage=0

compress=true
cmn_window=300
left_context=5
right_context=5
subsample=10

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <in-data-dir> <out-data-dir> <feat-dir>"
  echo "e.g.: $0 data/train data/train_no_sil exp/make_xvector_features"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

data_in=$1
data_out=$2
dir=$3

name=`basename $data_out`

for f in $data_in/feats.scp $data_in/vad.scp ; do
  [ ! -f $f ] && echo "$0: No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
mkdir -p $data_out
featdir=$(utils/make_absolute.sh $dir)

cp $data_in/utt2spk $data_out/utt2spk
cp $data_in/spk2utt $data_out/spk2utt

sdata_in=$data_in/split$nj;
utils/split_data.sh $data_in $nj || exit 1;

$cmd JOB=1:$nj $dir/log/create_feats_${name}.JOB.log \
  apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=$cmn_window \
  scp:${sdata_in}/JOB/feats.scp ark:- \| \
  select-voiced-frames ark:- scp:${sdata_in}/JOB/vad.scp ark:- \| \
  splice-feats --left-context=$left_context --right-context=$right_context ark:- ark:- \| \
  subsample-feats --n=$subsample ark:- ark:- \| \
  copy-feats --compress=$compress ark:- \
  ark,scp:$featdir/feats_${name}.JOB.ark,$featdir/feats_${name}.JOB.scp || exit 1;

for n in $(seq $nj); do
  cat $featdir/feats_${name}.$n.scp || exit 1;
done > ${data_out}/feats.scp || exit 1

echo "$0: Succeeded creating features for $name"
