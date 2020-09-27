#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
# @Time     : 2020/09/22 22:26
# @Author   : Yiwen Liao
# @File     : backbone.py 
# @Software : PyCharm 
# @Contact  : yiwen.liao@iss.uni-stuttgart.de


import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Softmax, MaxPool2D, Conv2D, BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, Reshape, Cropping2D, ZeroPadding2D, Concatenate


class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()


class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
