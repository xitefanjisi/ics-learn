#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
# @Time     : 2020/09/22 22:26
# @Author   : Yiwen Liao
# @File     : utils.py 
# @Software : PyCharm 
# @Contact  : yiwen.liao@iss.uni-stuttgart.de


import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10


def convert_labels(y=None):
    """convert labels so that the labels start from 0

    :param y: 1D-array, labels to convert
    :return: 1D-array, converted labers
    """
    for idx, lb in enumerate(np.unique(y)):
        y[y == lb] = idx
    return y


def intra_class_split(x=None, y=None, model_stg_1=None, rho=0.1, y_score=None):

    y_os = np.copy(y) + 1  # zero is reserved for open set class

    if y_score is not None:
        print('using predefined y_score to split data...')
    elif model_stg_1 is not None:
        print('use model_stg_1 to split data...')
        y_logits = model_stg_1.predict(x, batch_size=128)
        y_hat = np.argmax(y_logits, axis=-1)
        y_score = np.max(y_logits, axis=-1)
        y_score[y_hat != y] = y_score.min()  # reset misclassified scores using minimum
        print('# misclassified %d' % len(np.where(y_hat != y)[0]))
    else:
        raise ValueError('Please enter a Keras model...')

    score_order = np.argsort(y_score)
    thr_idx = int(rho * len(y))
    thr = y_score[score_order[thr_idx]]
    y_os[score_order[:thr_idx]] = 0  # atypical samples with labels of 0; typical samples with labels starting from 1

    print('min %.4f, max %.4f, thr %.4f' % (y_score.min(), y_score.max(), thr))
    print('y_os ratio', np.count_nonzero(y_os) / len(y_os))
    return y_os


# ----- data processing module: under construction -----
def update_labels(y, target_labels):
    idx_cs = [np.where(y == i)[0] for i in target_labels]
    idx_cs = np.concatenate(idx_cs, axis=-1)
    np.random.shuffle(idx_cs)  # mandatory, otherwise samples are ordered => bad for training
    idx_os = np.setdiff1d(np.arange(0, len(y)), idx_cs)
    return idx_cs, idx_os


class Datasets():
    def __init__(self, name=None):
        self.name = name

    def load_data(self, target_labels=None):
        if self.name == 'mnist':
            (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
            x_tr, x_te = x_tr / 255, x_te / 255
            x_tr = x_tr[..., tf.newaxis]
            x_te = x_te[..., tf.newaxis]
        elif self.name == 'cifar10':
            (x_tr, y_tr), (x_te, y_te) = cifar10.load_data()
            x_tr, x_te = x_tr / 255, x_te / 255
            y_tr, y_te = y_tr.reshape(-1, ), y_te.reshape(-1, )
        else:
            raise ValueError('Please enter a valid dataset...')

        if target_labels is not None:
            idx_cs, idx_os = update_labels(y_tr, target_labels)
            self.x_tr_cs, self.y_tr_cs = x_tr[idx_cs], y_tr[idx_cs]
            self.x_tr_os, self.y_tr_os = x_tr[idx_os], y_tr[idx_os]

            idx_cs, idx_os = update_labels(y_te, target_labels)
            self.x_te_cs, self.y_te_cs = x_te[idx_cs], y_te[idx_cs]
            self.x_te_os, self.y_te_os = x_te[idx_os], y_te[idx_os]
        else:
            raise ValueError('Please enter a valid target label list...')

        return self.x_tr_cs, self.y_tr_cs
