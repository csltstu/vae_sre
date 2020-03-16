import os
import numpy as np
import argparse
import kaldi_io
from tqdm import tqdm


def npz2ark(npz_file, ark_file):
    '''load npz format and save as kaldi ark format'''
    print("Loading npz file...")
    feats = np.load(npz_file)['feats']
    utt_label = np.load(npz_file)['utt_label']

    utters = np.unique(utt_label)
    utt_data = {}
    for i in utters:
        utt_data[i] = []
    for i in range(len(feats)):
        key = utt_label[i]
        utt_data[key].append(feats[i])

    if not os.path.exists(os.path.dirname(ark_file)):
        os.makedirs(os.path.dirname(ark_file))

    pbar = tqdm(total=len(utt_data))
    with open(ark_file,'wb') as f:
        for utt, data in utt_data.items():
            data = np.array(data)
            kaldi_io.write_mat(f, data, utt)
            pbar.update(1)
            pbar.set_description('generate utter {} of frames {}'.format(utt, data.shape[0]))
    pbar.close()
    print("Convert {} to {} ".format(npz_file, ark_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file', default="feats.npz",
                        help='src file of feats.(npz)')
    parser.add_argument('--dest-file', default="feats.ark",
                        help='dest file of feats.(ark)')
    args = parser.parse_args()

    npz2ark(args.src_file, args.dest_file)

