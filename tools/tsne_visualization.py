#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: tsne_vasualization.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Tue 21 Jan 2020 04:12:58 PM CST
# ************************************************************************/

import argparse
import os
import numpy as np



def save_label_tsv(save_path, x):
    y = x.flatten()
    n = len(y)
    with open(save_path, 'w') as f:
        for i in range(n):
            v = y[i]
            f.write('%s\n' % (str(v)))


def save_tensor_tsv(path, feats):
    with open(path, 'w') as f:
        num, dim = feats.shape
        for line in range(num):
            for d in range(dim):
                if d > 0:
                    f.write('\t')
                v = str(feats[line][d])
                f.write(v)
            f.write('\n')


def write_tsv_embeddings(prefix, feats, labels=None):
    '''
     Write a tensor (or meta) to a tsv file for the `Embedding Project` tool
    :param prefix: output file prefix
    :param feats: embedding tensor NxDim
    :param labels: meta data
    :return: None
    '''
    feat_path = prefix + '_data.tsv'
    save_tensor_tsv(feat_path, feats)
    if labels is None:
        return
    dims = len(labels.shape)
    label_path = prefix + '_meta.tsv'
    if dims == 1:
        save_label_tsv(label_path, labels)
    else:
        save_tensor_tsv(label_path, labels)


def sample(npz_path, prefix, sample_class=50, sample_num=500):
    print("loading data...")

    _data = np.load(npz_path, allow_pickle=True)['feats']
    _label = np.load(npz_path, allow_pickle=True)['spkers']

    print("start to sample...")
    sample_index = []
    while(len(sample_index)<sample_class):
        idx = np.random.randint(0, len(np.unique(_label))-1)
        if idx not in sample_index:
            sample_index.append(idx)

    sample_data = []
    sample_label = []

    print("label: ", len(_label))
    print("sample data...")
    print("sample class: {}, sample num: {}".format(sample_class, sample_num))

    for idx in sample_index:
        counter = 0
        print("sample idx", idx)
        for i in range(len(_label)):
            if(idx==_label[i]):
                if(counter<sample_num):
                    counter+=1
                    sample_label.append(idx)
                    sample_data.append(_data[idx])
                else:
                    break

    
    x = np.array(sample_data)

    y = []
    table = {}
    counter = 0
    for it in sample_label:
        if it not in table:
            table[it] = counter
            y.append(counter)
            counter+=1
        else:
            idx = table[it]
            y.append(idx)

    y = np.array(y)

    write_tsv_embeddings(prefix = "test", feats=x, labels=y)
    
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Flows for sre')
    parser.add_argument('--npz_path', type=str, default="../data/feats.npz", help='load the npz data')
    parser.add_argument('--pic_path', default='./pic.png',help='tsne pic path')
    args = parser.parse_args()

    sample(args.npz_path, args.pic_path)


