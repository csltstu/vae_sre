#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: data_loader.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Mon 20 Jan 2020 06:18:35 PM CST
# ************************************************************************/


import torch
import torch.utils.data as data
import os
import numpy as np

class feats_data_loader(data.Dataset):
    def __init__(self, npz_path="data/feats.npz", dataset_name="vox"):
        if(os.path.exists(npz_path)):
            print("hi")
        self.dataset_name=dataset_name

        _data = np.load(npz_path, allow_pickle=True)['feats']
        _label = np.load(npz_path, allow_pickle=True)['spkers']

        self._data = _data
        self._label = _label

        print("dataset: {}".format(dataset_name))
        print("feats shape: ", np.shape(_data))
        print("spker label shape: ", np.shape(_label))
        print("num of spker: ", np.shape(np.unique(_label)))

    def __len__(self):
        return len(self._label)

    def __getitem__(self, index):
        return self._data[index], self._label[index]

    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

    def sample(self):
        size = len(self.label)
        idx = np.random.randint(0, len(np.unique(_label))-1)
        return self.data[idx], label[idx]


if __name__ == "__main__":
    train_data = feats_data_loader(npz_path="./data/feats.npz", dataset_name="vox")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

