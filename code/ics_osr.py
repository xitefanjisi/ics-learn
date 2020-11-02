#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
# @Time     : 2020/09/27 15:23
# @Author   : Yiwen Liao
# @File     : ics_osr.py 
# @Software : PyCharm 
# @Contact  : yiwen.liao@iss.uni-stuttgart.de


import numpy as np
from backbone import CNN
from tensorflow.keras.optimizers import Adam
from utils import convert_labels, intra_class_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score


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
            self.model_stg_2 = CNN(output_dim=self.num_os_classes)
        else:
            self.model_stg_2 = model_stg_2

        self.model_stg_1.compile(optimizer=Adam(learning_rate=3e-4, decay=1e-8), loss='categorical_crossentropy')
        self.model_stg_2.compile(optimizer=Adam(learning_rate=3e-4, decay=1e-8), loss='categorical_crossentropy')

    def fit(self, x, y, ics_train=True, rho=0.1, verbose=2, **kwargs):

        assert y.max() == (len(np.unique(y))+1)

        if ics_train:
            print('training for intra-class splitting...')
            self.hist_stg_1 = self.model_stg_1.fit(x=x, y=to_categorical(y), verbose=verbose, **kwargs)
        else:
            print('training for open set recognition...')
            y_os = intra_class_split(x, y, self.model_stg_1, rho=rho)
            self.hist_stg_2 = self.model_stg_2.fit(x=x, y=to_categorical(y_os), verbose=verbose, **kwargs)

    def predict(self, x):
        return np.argmax(self.model_stg_2.predict(x=x, batch_size=64), axis=-1)

    def score(self, x, y):
        y_hat = np.argmax(self.model_stg_2.predict(x=x, batch_size=64), axis=-1)
        return f1_score(y, y_hat, average='macro', pos_label=np.arange(1, self.num_os_classes))

    def decision_function(self):
        return np.max(self.model_stg_2.predict(x=x, batch_size=64), axis=-1)

    def ics_score(self):
        pass
