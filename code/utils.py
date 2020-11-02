#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
# @Time     : 2020/09/22 22:26
# @Author   : Yiwen Liao
# @File     : utils.py 
# @Software : PyCharm 
# @Contact  : yiwen.liao@iss.uni-stuttgart.de


import numpy as np
from tensorflow.keras.utils import to_categorical


def convert_labels(y=None):

    for idx, lb in enumerate(np.unique(y)):
        y[y == lb] = idx


def intra_class_split(x=None, y=None, model_stg_1=None, rho=0.1):

    assert y.min() == 0, 'labels should start with 0'

    y_score = model_stg_1.predict(x, batch_size=128)
    y_hat = np.argmax(y_score, axis=-1)

    y_score = np.max(y_score, axis=-1)
    y_score[y_hat != y] = 0
    # TODO: What if y_score < 0

    thr = np.percentile(y_score, rho)
    y_os = np.copy(y) + 1  # original labels start from 1; zero is reserved for open set class
    y_os[y_score < thr] = 0
    return y_os
