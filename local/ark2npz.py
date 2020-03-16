import os
import numpy as np
import kaldi_io
import argparse


def create_utt2spk_map(utt2spk_file):
    '''build a hash map (utt->spk)'''
    print("Creating mapping dict utt2spk{}")
    items = np.loadtxt(utt2spk_file, dtype=str)

    spker_list = []
    for item in items:
        spker_list.append(item[1])

    spkers = np.unique(spker_list)
    idx = 0
    spk2id = {}
    for spk in spkers:
        spk2id[spk] = idx
        idx += 1

    utt2spk = {}
    for item in items:
        utt = item[0]
        spk = item[1]
        utt2spk[utt] = spk2id[spk]

    print("Created mapping dict utt2spk{}")
    return utt2spk


def kaldi2npz(scp_file, npz_file, utt2spk_file):
    '''load kaldi scp(ark) format and save as npz format'''
    print("Loading kaldi scp file...")
    utts = []
    mats = []
    for k, v in kaldi_io.read_mat_scp(scp_file):
        utts.append(k)
        mats.append(v)

    utt2spk = create_utt2spk_map(utt2spk_file)

    feats = []
    spker_label = []
    utt_label = []
    idx = 0
    for mat in mats:
        for i in mat:
            feats.append(i)
            spker_label.append(utt2spk[utts[idx]])
            utt_label.append(utts[idx])
        idx += 1
    
    if not os.path.exists(os.path.dirname(npz_file)):
        os.makedirs(os.path.dirname(npz_file))

    np.savez(npz_file, feats=feats, spker_label=spker_label, utt_label=utt_label)
    print("Convert {} to {} ".format(scp_file, npz_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file', default="feats.scp",
                        help='src file of feats.(scp)')
    parser.add_argument('--dest-file', default="feats.npz",
                        help='dest file of feats.(npz)')
    parser.add_argument('--utt2spk-file', default="utt2spk",
                        help='mapping file between utter and spker')
    args = parser.parse_args()

    kaldi2npz(args.src_file, args.dest_file, args.utt2spk_file)

    # load data and label
    feats = np.load(args.dest_file, allow_pickle=True)['feats']
    spker_label = np.load(args.dest_file, allow_pickle=True)['spker_label']
    utt_label = np.load(args.dest_file, allow_pickle=True)['utt_label']

    print("feats shape: ", np.shape(feats))

    print("spker label shape: ", np.shape(spker_label))
    print("num of spkers: ", np.shape(np.unique(spker_label)))
    print(spker_label[-1])

    print("utt label shape: ", np.shape(utt_label))
    print("num of utts: ", np.shape(np.unique(utt_label)))
    print(utt_label[-1])
