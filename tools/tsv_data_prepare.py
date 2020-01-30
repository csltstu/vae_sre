#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: tsv_data_prepare.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Tue 21 Jan 2020 04:12:58 PM CST
# ************************************************************************/

import os
import numpy as np



def save_label_tsv(path, x):
    y = x.flatten()
    n = len(y)
    with open(path, 'w') as f:
        for i in range(n):
            v = y[i]
            f.write('%s\n' % (str(v)))
        print("successfully saved in {}".format(path))


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
        print("successfully saved in {}".format(path))


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
    '''
    :param npz_path: npz file path which stores the datesets
    :param prefix: out_put tsv file name prefix
    :param sample_class: number of sample classes / number of spkers
    :param sample_num: sample number of each spkers
    :return: None
    '''
    print("loading data...")

    _data = np.load(npz_path)['feats']
    _label = np.load(npz_path)['spkers']

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

    write_tsv_embeddings(prefix=prefix, feats=x, labels=y)
    
   

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='a apple a day, keep doctor away')

    parser.add_argument('--npz_path', type=str, default="./data/feats.npz", help='load the npz data')
    parser.add_argument('--prefix', default='test',help='out_put tsv file name prefix')
    parser.add_argument('--class_num',type=int, default=30,help='class num / spker num')
    parser.add_argument('--sample_num',type=int, default=300,help='sample num of each spker')
    args = parser.parse_args()

    # sample, data preparetion and make tsv file
    sample(args.npz_path, args.prefix, args.class_num, args.sample_num)


