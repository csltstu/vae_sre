#!/bin/bash

# convert scp to npz
python -u ark2npz.py \
       --src-file example/feats.scp \
       --dest-file test/feats.npz \
       --utt2spk-file example/utt2spk

# convert npz to ark
python -u npz2ark.py \
       --src-file test/feats.npz \
       --dest-file test/feats.ark \

