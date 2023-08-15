#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on April 27 11:26:27 2022

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



# 将原始训练数据转成输入数据的函数
def getMatrixLabel(positive_position_file_name,sites, window_size=51, empty_aa = '*'):
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    prot = []  #
    pos = []  #
    rawseq = [] #
    all_label = [] #

    short_seqs = []
    half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:

                position = int(row[2])
                sseq = row[3]
                rawseq.append(row[3])
                center = sseq[position - 1]
                all_label.append(int(row[0]))
                prot.append(row[1])
                pos.append(row[2])
                # coding = one_hot_concat(shortseq)
                # all_codings.append(coding)
        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # “A binary matrix representation of the input”
        ONE_HOT_SIZE = 21
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19
        letterDict["*"] = 20

        #
        Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in rawseq:
            AANo = 0
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr

import gensim
from gensim.models import Word2Vec
import numpy as np
# 
def getMatrixLabelh(positive_position_file_name, window_size=51, empty_aa = '*'):
    # input format   label, proteinName, postion, shortsequence
    # label存储0/1值
    #positive_position_file_name='trainingAMP.csv'
    #window_size=50
    prot = []  #
    pos = []  #
    rawseq = [] #
    all_label = [] #
    length=[]
    short_seqs = []
    #half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:

                #position = int(row[2])
                a=window_size-len(row[1])
                sseq = row[1]#+a*' '
                rawseq.append(sseq)
                b=len(row[1])
                length.append(b)
                #center = sseq[position - 1]
            # 
                all_label.append(int(row[0]))
                #prot.append(row[1])
                #pos.append(row[2])

        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # “A binary matrix representation of the input”
        targetY = kutils.to_categorical(all_label)

        ONE_HOT_SIZE = 20
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0;
        letterDict["C"] = 1;
        letterDict["D"] = 2;
        letterDict["E"] = 3;
        letterDict["F"] = 4;
        letterDict["G"] = 5;
        letterDict["H"] = 6;
        letterDict["I"] = 7;
        letterDict["K"] = 8;
        letterDict["L"] = 9;
        letterDict["M"] = 10;
        letterDict["N"] = 11;
        letterDict["P"] = 12;
        letterDict["Q"] = 13;
        letterDict["R"] = 14;
        letterDict["S"] = 15;
        letterDict["T"] = 16;
        letterDict["V"] = 17;
        letterDict["W"] = 18;
        letterDict["Y"] = 19;
        #letterDict['Z'] = 23

        #
        #
        Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in rawseq:
            AANo = 0
            #print(seq)
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr, targetY,rawseq,length

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
    #model_input = Input(shape=img_dim)
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

    x11 = x1
    
    x11 = transitionh(x11, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x1 = transition(x1, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    
    x01=concatenate([x1,x11])
    channel = x01.shape[-1]
    x1 = layers.Dense(channel*0.5)(x1)
    x11 = layers.Dense(channel*0.5)(x11)
    x1 = Activation('relu',name='seq1')(x1)
    x11 = Activation('relu',name='seq11')(x11)
    x1 = layers.Dense(channel)(x1)
    x11 = layers.Dense(channel)(x11)
    xxx1=layers.Add()([x1, x11])
    #xxx1=x01
    xxx1 = tf.nn.sigmoid(xxx1)
    xxx1 = layers.Multiply()([x01, xxx1])
    xxx1 = denseblock(xxx1, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
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
    

    x22 = x2
    
    x22 = transitionh(x22, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    x2 = transition(x2, init_form, nb_filter, dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    
    x02=concatenate([x2,x22])
    channel = x02.shape[-1]
    x2 = layers.Dense(channel*0.25)(x2)
    x22 = layers.Dense(channel*0.25)(x22)
    x2 = Activation('relu',name='seq2')(x2)
    x22 = Activation('relu',name='seq22')(x22)
    x2 = layers.Dense(channel)(x2)
    x22 = layers.Dense(channel)(x22)
    xxx2=layers.Add()([x2, x22])
    #xxx2=x02
    xxx2 = tf.nn.sigmoid(xxx2)
    xxx2 = layers.Multiply()([x02, xxx2])
    xxx2 = denseblock(xxx2, init_form, nb_layers, nb_filter, growth_rate, filter_size_block2,
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
    

    x33 = x3
    
    x33 = transitionh(x33, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x3 = transition(x3, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    
    x03=concatenate([x3,x33])
    channel = x03.shape[-1]
    x3 = layers.Dense(channel*0.25)(x3)
    x33 = layers.Dense(channel*0.25)(x33)
    x3 = Activation('relu',name='seq3')(x3)
    x33 = Activation('relu',name='seq33')(x33)
    x3 = layers.Dense(channel)(x3)
    x33 = layers.Dense(channel)(x33)
    xxx3=layers.Add()([x3, x33])
    #xxx3=x03
    xxx3 = tf.nn.sigmoid(xxx3)
    xxx3 = layers.Multiply()([x03, xxx3])
    xxx3 = denseblock(xxx3, init_form, nb_layers, nb_filter, growth_rate, filter_size_block3,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)
    
    # first input of 33 seq #
    input4 = Input(shape=img_dim4)
    #model_input = Input(shape=img_dim)
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

    x44 = x4
    
    x44 = transitionh(x44, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    x4 = transition(x4, init_form, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)
    
    x04=concatenate([x4,x44])
    channel = x04.shape[-1]
    x4 = layers.Dense(channel*0.25)(x4)
    x44 = layers.Dense(channel*0.25)(x44)
    x4 = Activation('relu',name='seq4')(x4)
    x44 = Activation('relu',name='seq44')(x44)
    x4 = layers.Dense(channel)(x4)
    x44 = layers.Dense(channel)(x44)
    xxx4=layers.Add()([x4, x44])
    #xxx1=x01
    xxx4 = tf.nn.sigmoid(xxx4)
    xxx4 = layers.Multiply()([x04, xxx4])
    xxx4 = denseblock(xxx4, init_form, nb_layers, nb_filter, growth_rate, filter_size_block1,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay)

    # The last denseblock does not have a transition
    # The last denseblock does not have a transition

    # contact 3 output features #
    x = concatenate([xxx1,xxx2,xxx3,xxx4], axis=-2, name='contact_multi_seq')

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
# 
def model_net(X_train1, X_train2, X_train3, y_train,
              nb_epoch=60,weights=None):

    nb_classes = 2
    img_dim1 = X_train1.shape[1:]
    img_dim2 = X_train2.shape[1:]
    img_dim3 = X_train3.shape[1:]

    ##########parameters#########

    init_form = 'RandomUniform'
    learning_rate = 0.001
    nb_dense_block = 1
    nb_layers = 5
    nb_filter = 32
    growth_rate = 32
    # growth_rate = 24
    filter_size_block1 = 13
    filter_size_block2 = 7
    filter_size_block3 = 3
    filter_size_ori = 1
    dense_number = 32
    dropout_rate = 0.2
    dropout_dense = 0.3
    weight_decay = 0.0001
    nb_batch_size = 512



    ###################
    # Construct model #
    ###################
    # from phosnet import Phos
    model = Phos(nb_classes, nb_layers, img_dim1, img_dim2, img_dim3, init_form, nb_dense_block,
                             growth_rate, filter_size_block1, filter_size_block2, filter_size_block3,
                             nb_filter, filter_size_ori,
                             dense_number, dropout_rate, dropout_dense, weight_decay)
    # Model output

    # choose optimazation
    opt = adam_v2.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # model compile
    model.compile(loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    # load weights#
    if weights is not None:
        model.load_weights(weights)
        # model2 = copy.deepcopy(model)
        model2 = model
        model2.load_weights(weights)
        for num in range(len(model2.layers) - 1):
            model.layers[num].set_weights(model2.layers[num].get_weights())

    if nb_epoch > 0 :
      model.fit([X_train1, X_train2, X_train3], y_train, batch_size=nb_batch_size,
                         # validation_data=([X_val1, X_val2, X_val3, y_val),
                         # validation_split=0.1,
                         epochs= nb_epoch, shuffle=True, verbose=1)


    return model


# 处理测试数据
def getMatrixInput(positive_position_file_name,sites, window_size=51, empty_aa = '*'):
    number=1
    a=0
    # input format  proteinName, postion, shortsequence,
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    # all_label = []

    short_seqs = []
    half_len = (window_size - 1) / 2

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
                position = int(row[1])
                sseq = row[2]
                a=window_size-len(row[2])
                sseq = row[2]+a*' '
                rawseq.append(row[1])
                center = sseq[position - 1]
            # 
                #all_label.append(int(row[0]))
                prot.append(row[0])
                pos.append(row[1])

                #short seq
                if position - half_len > 0:
                    start = int(position - half_len)
                    left_seq = sseq[start - 1:position - 1]
                else:
                    left_seq = sseq[0:position - 1]

                end = len(sseq)
                if position + half_len < end:
                    end = int(position + half_len)
                right_seq = sseq[position:end]

                if len(left_seq) < half_len:
                    nb_lack = half_len - len(left_seq)
                    left_seq = ''.join([empty_aa for count in range(nb_lack)]) + left_seq

                if len(right_seq) < half_len:
                    nb_lack = half_len - len(right_seq)
                    right_seq = right_seq + ''.join([empty_aa for count in range(nb_lack)])
                shortseq = left_seq + center + right_seq
                short_seqs.append(shortseq)
                

        all_label = [0] *5 + [1]*(len(sseq)-5)
        targetY = kutils.to_categorical(all_label)

        ONE_HOT_SIZE = 21
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0
        letterDict["C"] = 1
        letterDict["D"] = 2
        letterDict["E"] = 3
        letterDict["F"] = 4
        letterDict["G"] = 5
        letterDict["H"] = 6
        letterDict["I"] = 7
        letterDict["K"] = 8
        letterDict["L"] = 9
        letterDict["M"] = 10
        letterDict["N"] = 11
        letterDict["P"] = 12
        letterDict["Q"] = 13
        letterDict["R"] = 14
        letterDict["S"] = 15
        letterDict["T"] = 16
        letterDict["V"] = 17
        letterDict["W"] = 18
        letterDict["Y"] = 19
        letterDict[" "] = 20

        # print len(short_seqs)
        if number==1:
         Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        else:
         a=a+1
        number=number+1
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            for AA in seq:
                try:
                    index = letterDict[AA]
                except:
                    index = 20
                # print index
                Matr[samplenumber][AANo][index] = 1
                # print samplenumber
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr, targetY, prot, pos

def getMatrixInputh(positive_position_file_name,sites, window_size=51, empty_aa = '*'):
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    #positive_position_file_name='AMPtest.csv'
    prot = []  # 
    pos = []  # 
    rawseq = [] # 
    all_label = [] # 
    window_size=200
    short_seqs = []

    with open(positive_position_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:

                #position = int(row[2])
                a=window_size-len(row[0])
                sseq = row[0]+a*' '
                rawseq.append(sseq)
                #center = sseq[position - 1]
            # 
                #all_label.append(int(row[0]))
                #prot.append(row[1])
                #pos.append(row[2])

        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # “A binary matrix representation of the input”
        #targetY = kutils.to_categorical(all_label)

        ONE_HOT_SIZE = 21
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        letterDict = {}
        letterDict["A"] = 0;
        letterDict["C"] = 1;
        letterDict["D"] = 2;
        letterDict["E"] = 3;
        letterDict["F"] = 4;
        letterDict["G"] = 5;
        letterDict["H"] = 6;
        letterDict["I"] = 7;
        letterDict["K"] = 8;
        letterDict["L"] = 9;
        letterDict["M"] = 10;
        letterDict["N"] = 11;
        letterDict["P"] = 12;
        letterDict["Q"] = 13;
        letterDict["R"] = 14;
        letterDict["S"] = 15;
        letterDict["T"] = 16;
        letterDict["V"] = 17;
        letterDict["W"] = 18;
        letterDict["Y"] = 19;
        letterDict[' '] = 20;
        #letterDict['Z'] = 23

        # 
        # 
        Matr = np.zeros((len(rawseq), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in rawseq:
            AANo = 0
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo+1
            samplenumber = samplenumber + 1

    return Matr

import csv
import numpy as np
import keras.utils.np_utils as kutils
# from keras.optimizers import Adam, SGD
from keras.optimizers import adam_v2
from keras.layers import Conv1D,Conv2D, MaxPooling2D
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
from keras.models import Sequential, Model
import numpy as np
import keras.utils.np_utils as kutils
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.PyPro import GetProDes
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.GetProteinFromUniprot import GetProteinSequence as gps
from propy import GetSubSeq
from propy.GetProteinFromUniprot import GetProteinSequence
from propy.AAComposition import CalculateAAComposition
from propy.AAComposition import CalculateAADipeptideComposition
from propy.AAComposition import GetSpectrumDict
from propy.AAComposition import Getkmers

def getMatrixLabelFingerprint(positive_position_file_name, window_size=51, empty_aa = '*'):
       # 
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    prot = []  # 
    pos = []  # 
    rawseq11 = [] # 
    all_label = [] # 
    window_size=91
    short_seqs = []
    #half_len = (window_size - 1) / 2
    ONE_HOT_SIZE = 17
    a=0
    b=0
    o=0
    number=1
    index=0
    #positive_position_file_name='111antivirual56297.csv'#'Antivirual.csv'#'测试集.csv'#'Non-AMPsfilter.csv'#'TrainingAMP.csv'#'需要预测的序列6.csv'#'需要预测的序列.csv'#'TrainingAMP.csv'#'测试集.csv'#'细菌素补充数据集.csv'#'TrainingAMP.csv'#'Extracellular space.csv'#'mitochondria.csv'#'Vesicle.csv'#'NonAMPtraining.csv'#'Negative50.csv'#'1.csv'#'Non-AMPsfilter.csv'#'Non-AMPsfilter.csv'#'Training1.csv'#'Non-AMPs.csv'#'验证数据集.csv'#'德国小蠊蛋白组.csv'#'美洲大蠊宏基因组AMPs未预测.csv'#'AMPs.csv'#'Training1T.csv'#'Non-AMPs.csv'#'AMPs.csv'#'Training1.csv'
    with open(positive_position_file_name, 'r') as rf:
      reader = csv.reader(rf)
      for row in reader:
        #position = int(row[2])
        #sseq = row[1]
        rawseq11.append(row[1])
        #center = sseq[position - 1]
            # 
            #if center in sites:
        #.append(int(row[0]))
        index=index+1
        #print(index-1)
        #prot.append(row[1])
        #pos.append(row[2])
                # coding = one_hot_concat(shortseq)
                # all_codings.append(coding)
        
        # Keras的utilities，
        # 输出是“A binary matrix representation of the input”
        #targetY = kutils.to_categorical(all_label)
    index=0
    a=0
    b=0
    yyy=np.zeros(shape=(1547,1))
    Matr = np.zeros((len(rawseq11),window_size, ONE_HOT_SIZE))

    for index in range(0,len(rawseq11)):
      result = GetProDes(rawseq11[index]).GetALL()
      print(index)
      for p in range(0,1547):
          yyy[p][0]=list(result.values())[p]
      for i in range(0,91):
          for j in range(0,17):
            Matr[index][i][j]=yyy[j+a][0]
          if a<1530:
            a=a+17
          else:
            a=a
      a=0
#np.save(file="X2.npy",arr=aaa) X2=np.load(file="X2.npy")
#np.save(file="X2s.npy",arr=X_train2) X_train2=np.load(file="X2s.npy")
#np.save(file="Y.npy",arr=targetY)
    return Matr

def getMatrixLabelFingerprintp(positive_position_file_name, window_size=51, empty_aa = '*'):
       # 
    # input format   label, proteinName, postion, shortsequence
    # label 0/1
    prot = []  # 
    pos = []  # 
    rawseq11 = [] # 
    all_label = [] # 
    window_size=91
    short_seqs = []
    #half_len = (window_size - 1) / 2
    ONE_HOT_SIZE = 17
    a=0
    b=0
    o=0
    number=1
    index=0
    positive_position_file_name='Predicted.csv'
    with open(positive_position_file_name, 'r') as rf:
      reader = csv.reader(rf)
      for row in reader:
        #position = int(row[2])
        #sseq = row[1]
        rawseq11.append(row[1])
        #center = sseq[position - 1]
            # 
            #if center in sites:
        #.append(int(row[0]))
        index=index+1
        #print(index-1)
        #prot.append(row[1])
        #pos.append(row[2])
                # coding = one_hot_concat(shortseq)
                # all_codings.append(coding)
        
        # Keras的utilities，用于“Converts a class vector (integers) to binary class matrix.”
        # 输出是“A binary matrix representation of the input”
        #targetY = kutils.to_categorical(all_label)
    index=0
    a=0
    b=0
    yyy=np.zeros(shape=(1547,1))
    Matr = np.zeros((len(rawseq11),window_size, ONE_HOT_SIZE))

    for index in range(0,len(rawseq11)):
      result = GetProDes(rawseq11[index]).GetALL()
      print(index)
      for p in range(0,1547):
          yyy[p][0]=list(result.values())[p]
      for i in range(0,91):
          for j in range(0,17):
            Matr[index][i][j]=yyy[j+a][0]
          if a<1530:
            a=a+17
          else:
            a=a
      a=0
#np.save(file="X2.npy",arr=aaa) X2=np.load(file="X2.npy")
#np.save(file="X2s.npy",arr=X_train2) X_train2=np.load(file="X2s.npy")
#np.save(file="Y.npy",arr=targetY)
    return Matr


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def plot_ROC(labels,preds,savepath):
    """
    Args:
        labels : ground truth
        preds : model prediction
        savepath : save path 
    """
    #labels=y_train1[:,1] preds=predictions_p[:,1] savepath='D://'
    fpr1, tpr1, threshold1 = metrics.roc_curve(labels, preds)  ###
    precision,recall,threshold1 = metrics.precision_recall_curve(labels, preds)
    roc_auc1 = metrics.auc(fpr1,tpr1)  ###计算auc的值，auc
    roc_auc1 = metrics.auc(recall,precision)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr1, tpr1, color='darkorange',
            lw=lw, label='AUC = %0.2f' % roc_auc1)  ###
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    # plt.title('ROCs for Densenet')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(savepath)
 
    return x


