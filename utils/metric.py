#!/usr/bin/env python
# coding=utf-8

# *************************************************************************
#	> File Name: skew_kurt.py
#	> Author: Yang Zhang 
#	> Mail: zyziszy@foxmail.com
#	> Created Time: Sat 11 Jan 2020 08:29:59 AM CST
# ************************************************************************/

import math
import numpy as np
from scipy import stats

def get_skew_and_kurt(data):
    '''calculate skew and kurt'''
    data = np.array(data)
    data = data.transpose()

    print(data.shape)  # test

    skew = []
    kurt = []
    for i in data:
        # print(len(i)) # test
        skew.append(stats.skew(i))
        kurt.append(stats.kurtosis(i))

    skew_mean = sum(skew)/len(skew)  
    kurt_mean = sum(kurt)/len(kurt)

    # print('skew:', skew_mean)  # test
    # print('kurt:', kurt_mean)  # test
    return skew_mean, kurt_mean
