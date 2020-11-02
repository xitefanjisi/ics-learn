#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
# @Time     : 2020/09/22 22:26
# @Author   : Yiwen Liao
# @File     : backbone.py 
# @Software : PyCharm 
# @Contact  : yiwen.liao@iss.uni-stuttgart.de


import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Softmax, MaxPool2D, Conv2D, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, Reshape, Cropping2D, ZeroPadding2D, Concatenate, Flatten


class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()


class CNN(tf.keras.Model):
    def __init__(self, output_dim=None, alpha=0.2):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=None)
        self.bn1 = BatchNormalization()
        self.lrelu1 = LeakyReLU(alpha=alpha)
        self.conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=None)
        self.bn2 = BatchNormalization()
        self.lrelu2 = LeakyReLU(alpha=alpha)
        self.conv3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_regularizer=None)
        self.bn3 = BatchNormalization()
        self.lrelu3 = LeakyReLU(alpha=alpha)
        self.flt = Flatten()
        self.dense = Dense(units=output_dim, activation='linear')
        self.top = Softmax()

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        x = self.flt(x)
        x = self.dense(x)
        x = self.top(x)
        return x


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
