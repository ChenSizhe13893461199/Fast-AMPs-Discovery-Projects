#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Sizhe Chen
"""

import csv
import numpy as np
import keras.utils.np_utils as kutils
# from keras.optimizers import Adam, SGD
from keras.optimizers import adam_v2
from keras.layers import Conv1D,Conv2D, MaxPooling2D,MaxPooling1D,GlobalMaxPooling1D
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, AveragePooling1D,GlobalAveragePooling1D
#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
#from keras.layers import Input, merge, Flatten
from keras.models import Sequential, Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

# 建立网络的函数
def conv_factory(x, init_form, nb_filter, filter_size_block, dropout_rate, weight_decay):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout

    :param x: Input keras network
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras network with b_norm, relu and convolution2d added
    :rtype: keras network
    """
    #x = Activation('relu')(x)
    # 参数的名称有修改
    x = Conv1D(nb_filter, filter_size_block,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition(x, init_form, nb_filter, dropout_rate, weight_decay):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, 1,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = AveragePooling2D((2, 2),padding='same')(x)
    x = AveragePooling1D(pool_size=3, padding='same')(x)
    #x = AveragePooling2D((2,2), padding='same')(x)
    return x

def transitionh(x, init_form, nb_filter, dropout_rate, weight_decay):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D

    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool

    """
    #x = Activation('relu')(x)
    x = Conv1D(nb_filter, 1,
                      kernel_initializer=init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    #x = MaxPooling2D((2, 2),padding='same')(x)
    x = MaxPooling1D(pool_size=3, padding='same')(x)
    #x = MaxPooling2D((2,2), padding='same')(x)

    return x


def denseblock(x, init_form, nb_layers, nb_filter, growth_rate,filter_size_block,
               dropout_rate, weight_decay):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones

    :param x: keras model
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """

    list_feat = [x]
    concat_axis = -1

    for i in range(nb_layers):
        x = conv_factory(x, init_form, growth_rate, filter_size_block, dropout_rate, weight_decay)
        list_feat.append(x)
        x = concatenate(list_feat, axis=concat_axis)
        nb_filter += growth_rate
    return x


def Phos1(nb_classes, nb_layers,img_dim1,img_dim2,img_dim3,img_dim4, init_form, nb_dense_block,
             growth_rate,filter_size_block1,filter_size_block2,filter_size_block3,filter_size_block4,
             nb_filter, filter_size_ori,dense_number,dropout_rate,dropout_dense,weight_decay):
    """ Build the DenseNet model

    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :param nb_layers:int --numbers of layers in a dense block
    :param filter_size_ori: int -- filter size of first conv1d
    :param dropout_dense: float---drop out rate of dense

    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model

    """
    # first input of 33 seq #
    main_input = Input(shape=img_dim1)
    # Initial convolution
    x1 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(main_input)
    x11 = x1
    x1 = denseblock(x1, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)


    # second input of 21 seq #
    input2 = Input(shape=img_dim2)
    x2 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input2)
    x22 = x2
    
    x2 = denseblock(x2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)


    # The last denseblock does not have a transition

    #third input seq3 of 15 #
    input3 = Input(shape=img_dim3)
    x3 = Conv1D(nb_filter, filter_size_ori,
                kernel_initializer = init_form,
                activation='relu',
                padding='same',
                use_bias=False,
                kernel_regularizer=l2(weight_decay))(input3)
    x33 = x3
    
    x3 = denseblock(x3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    
    
    # first input of 33 seq #
    input4 = Input(shape=img_dim4)
    # Initial convolution
    x4 = Conv1D(nb_filter, filter_size_ori,
                      kernel_initializer = init_form,
                      activation='relu',
                      padding='same',
                      use_bias=False,
                      kernel_regularizer=l2(weight_decay))(input4)
    x44 = x4
    x4 = denseblock(x4, init_form, nb_layers, nb_filter, growth_rate,filter_size_block1,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)


    # The last denseblock does not have a transition
    # The last denseblock does not have a transition

    # contact 3 output features #
    x = concatenate([x1,x2,x3,x4], axis=-2, name='contact_multi_seq')

    #x = GlobalAveragePooling1D()(x)

    x = Flatten()(x)

    x = Dense(dense_number,
              name ='Dense_1',
              activation='relu',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    x = Dropout(dropout_dense)(x)
    #softmax
    x = Dense(nb_classes,
              name = 'Dense_softmax',
              activation='softmax',
              kernel_initializer = init_form,
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    phos_model = Model(inputs=[main_input,input2,input3,input4], outputs=[x], name="multi-DenseNet")
    #feauture_dense = Model(input=[main_input, input2, input3], output=[x], name="multi-DenseNet")

    return phos_model