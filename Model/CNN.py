# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 19:45:56 2022

@author:Sizhe Chen
"""
import os
import string
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, AveragePooling1D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalAveragePooling1D

#from keras.layers import Input, merge, Flatten
from keras.layers import Input
from keras.layers.reshaping import Flatten
from keras.layers import concatenate, add
from keras.layers.normalization import batch_normalization
from keras.layers import BatchNormalization


from keras.regularizers import l2
import keras.backend as K
from keras.layers import Conv1D, Conv2D, MaxPooling2D


import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from utils import getMatrixLabel, Phos1, getMatrixInput, getMatrixInputh, getMatrixLabelFingerprint, getMatrixLabelh, plot_ROC
from keras.optimizers import adam_v2
#from utils import channel_attenstion

import matplotlib.pyplot as plt

from keras.utils.vis_utils import plot_model
#
import csv
import numpy as np
import keras.utils.np_utils as kutils
from keras.optimizers import adam_v2
from keras.layers import Conv1D, Conv2D, MaxPooling2D
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
from sklearn.preprocessing import StandardScaler

img_dim1 = aaa.shape[1:]

img_dim2 = aaa.shape[1:]

img_dim3 = bbb.shape[1:] #img_dim3 = aaa.shape[1:]

img_dim4 = bbb.shape[1:] #img_dim4 = aaa.shape[1:]


init_form = 'RandomUniform'
learning_rate = 0.0005#0.001
nb_dense_block = 9
nb_layers = 9
nb_filter = 36
growth_rate = 36
filter_size_block1 = 11
filter_size_block2 = 7
filter_size_block3 = 11
filter_size_block4 = 7
filter_size_ori = 1
dense_number = 36
dropout_rate = 0.2
dropout_dense = 0.2
weight_decay = 0.000001
nb_batch_size = 512
nb_classes = 2
nb_epoch = 15

model1 = Phos1(nb_classes, nb_layers, img_dim1, img_dim2, img_dim3,img_dim4, init_form, nb_dense_block,
              growth_rate, filter_size_block1, filter_size_block2, filter_size_block3,filter_size_block4,
              nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay)

# 模型可视化
print(model1.summary())
plot_model(model1, to_file='DTLDephos.png',
           show_shapes=True, show_layer_names=True)

opt = adam_v2.Adam(learning_rate=learning_rate,
                   beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model1.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

history = model1.fit([aaa[:],aaa[:],bbb[:],bbb[:]], ddd[:], batch_size=nb_batch_size,validation_data=([aaa[:],aaa[:],bbb[:],bbb[:]], ddd[:]),epochs=nb_epoch, shuffle=True, verbose=1)
predictions_p = model1.predict([X1tt,X1tt,X2tt,X2tt])#Evaluating the effects on Test dataset
print(np.sum(predictions_p[990:,0]>0.5))#Print AMPs predictions
print(np.sum(predictions_p[:990,1]>0.5))#Print Non-AMPs predictions




