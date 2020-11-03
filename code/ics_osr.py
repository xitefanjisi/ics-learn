#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
# @Time     : 2020/09/27 15:23
# @Author   : Yiwen Liao
# @File     : ics_osr.py 
# @Software : PyCharm 
# @Contact  : yiwen.liao@iss.uni-stuttgart.de


import numpy as np
from backbone import CNN, OSRNet
from tensorflow.keras.optimizers import Adam
from utils import convert_labels, intra_class_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import Callback


def logit_loss(y_true, y_pred):
    y_pred = tf.nn.softmax(y_pred)
    return categorical_crossentropy(y_true, y_pred)


class OSRMonitor(Callback):
    def __init__(self, data=None, step=10):
        super(OSRMonitor, self).__init__()
        self.data = data
        self.step = step

    def on_epoch_begin(self, epoch, logs=None):
        if epoch % self.step == 0:
            res = np.argmax(self.model.predict(self.data.x_te_os)[1], axis=-1)
            os_rejection_rate = 1 - np.count_nonzero(res) / len(res)

            x_te_cs, y_te_cs = self.data.x_te_cs, self.data.y_te_cs
            y_te_cs = convert_labels(y_te_cs) + 1  # closed set labels start from 1
            res = np.argmax(self.model.predict(x_te_cs)[1], axis=-1)
            cs_accu = accuracy_score(y_te_cs, res)

            baccu = 0.5 * (os_rejection_rate + cs_accu)
            print('[Rejection rate %.4f] [CS Accu %.4f]\n[BACCU %.4f]' % (os_rejection_rate,
                                                                          cs_accu, baccu))


class OpenSetRecognizer():

    def __init__(self, num_classes=None, model_stg_1=None, model_stg_2=None):

        self.num_classes = num_classes
        self.num_os_classes = num_classes + 1
        self.hist_stg_1 = None
        self.hist_stg_2 = None

        if model_stg_1 is None:
            self.model_stg_1 = CNN(output_dim=self.num_classes)
        else:
            self.model_stg_1 = model_stg_1

        if model_stg_2 is None:
            self.model_stg_2 = OSRNet(output_dim=self.num_classes)
        else:
            self.model_stg_2 = model_stg_2

        self.model_stg_1.compile(optimizer=Adam(learning_rate=3e-4, decay=1e-8),
                                 loss=logit_loss, metrics=['accuracy'])

        self.model_stg_2.compile(optimizer=Adam(learning_rate=3e-4, decay=1e-8),
                                 loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, x, y, ics_train=True, rho=0.1, verbose=2, epochs=None, callback=None, **kwargs):

        assert y.max() == (len(np.unique(y)) - 1)

        if ics_train:
            print('training for intra-class splitting...')
            self.hist_stg_1 = self.model_stg_1.fit(x=x, y=to_categorical(y),
                                                   verbose=verbose, epochs=epochs[0], **kwargs)
        else:
            pass
        print('training for open set recognition...')
        y_os = intra_class_split(x, y, self.model_stg_1, rho=rho)
        self.hist_stg_2 = self.model_stg_2.fit(x=x,
                                               y=[to_categorical(y, num_classes=self.num_classes),
                                                  to_categorical(y_os, num_classes=self.num_os_classes)],
                                               verbose=verbose, epochs=epochs[1], callbacks=[callback],
                                               **kwargs)

    def predict(self, x):
        return np.argmax(self.model_stg_2.predict(x=x, batch_size=64), axis=-1)

    def score(self, x, y):
        y_hat = np.argmax(self.model_stg_2.predict(x=x, batch_size=64), axis=-1)
        return f1_score(y, y_hat, average='macro', pos_label=np.arange(1, self.num_os_classes))

    def decision_function(self, x):
        return np.max(self.model_stg_2.predict(x=x, batch_size=64), axis=-1)

    def ics_score(self):
        pass
