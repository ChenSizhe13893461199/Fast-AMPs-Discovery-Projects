# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 19:45:56 2022

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

from utils import getMatrixLabel, Phos,Phos1, getMatrixInput, getMatrixInputh, getMatrixLabelFingerprint, getMatrixLabelh, plot_ROC, getwordvector
from keras.optimizers import adam_v2
from utils import channel_attenstion
from utils import Phosh

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
# *************************************************************
train_file_name = 'TrainingAMP.csv'  # 训练集
win1 = 50

X1, T, rawseq, length = getMatrixLabelh(train_file_name, win1)
train_file_name = 'Non-AMPsfilter.csv'  # 测试集

win1 = 50
X1tt, y_train1, rawseq1, length = getMatrixLabelh(train_file_name, win1)

train_file_name = '测试集.csv'
win1 = 50
X_val, y_train111, rawseq116, length = getMatrixLabelh(train_file_name, win1)

X2 = np.load(file="Training_vector.npy")
X2tt = np.load(file="Test_vector.npy")
X2_val = np.load(file="5810_vector.npy")

###################################################################
aaa = np.zeros((43404+5810, 50, 20))
bbb = np.zeros((43404+5810, 91, 17))
aaa[:43404] = X1[:]
aaa[43404:] = X_val[:]
bbb[:43404] = X2[:]
bbb[43404:] = X2_val[:]

ddd = np.zeros(shape=(43404+5810, 2))
ddd[:43404] = T[:]
ddd[43404:] = y_train111[:]
###################################################################
# 模型训练区
#a1=aaa
#b1=bbb
#e1=eee
img_dim1 = aaa.shape[1:]

img_dim2 = aaa.shape[1:]

img_dim3 = bbb.shape[1:] #img_dim3 = aaa.shape[1:]

img_dim4 = bbb.shape[1:] #img_dim4 = aaa.shape[1:]


init_form = 'RandomUniform'
learning_rate = 0.0015#0.001
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
nb_epoch = 11

model1 = Phos1(nb_classes, nb_layers, img_dim1, img_dim2, img_dim3,img_dim4, init_form, nb_dense_block,
              growth_rate, filter_size_block1, filter_size_block2, filter_size_block3,filter_size_block4,
              nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay)
#model1 = Phosh(nb_classes, nb_layers, img_dim1, img_dim2, img_dim3, init_form, nb_dense_block,
              #growth_rate, filter_size_block1, filter_size_block2, filter_size_block3,filter_size_block4,filter_size_block5,filter_size_block6,
              #nb_filter, filter_size_ori,dense_number, dropout_rate, dropout_dense, weight_decay)
# model = Phos(nb_classes, nb_layers, img_dim1,img_dim2, init_form, nb_dense_block,
#growth_rate, filter_size_block1, filter_size_block2,
#nb_filter, filter_size_ori,
# dense_number, dropout_rate, dropout_dense, weight_decay)

# 模型可视化
print(model1.summary())
plot_model(model1, to_file='DTLDephos.png',
           show_shapes=True, show_layer_names=True)

opt = adam_v2.Adam(learning_rate=learning_rate,
                   beta_1=0.9, beta_2=0.999, epsilon=1e-08)

model1.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

# history = model.fit([X1,Y1,Z1], T, batch_size=nb_batch_size,
# validation_split=0.2,
# epochs= nb_epoch, shuffle=True, verbose=1)#validation_split=0.2,validation_data=([X_val1, X_val2, X_val3], y_train111)

# history = model.fit([g1[:50485],g2[:50485],g3[:50485]], y_train11[:50485], batch_size=nb_batch_size,
# validation_split=0.0,
# epochs= nb_epoch, shuffle=True, verbose=1)
#history = model.fit([aaa[:43214],bbb[:43214],eee[:43214]], ddd[:43214], batch_size=nb_batch_size,validation_data=([aaa[43214:49214],bbb[43214:49214],eee[43214:49214]], ddd[43214:49214]),epochs=nb_epoch, shuffle=True, verbose=1)
#history = model1.fit([aaa[:43404+2975],aaa[:43404+2975],bbb[:43404+2975],bbb[:43404+2975]], ddd[:43404+2975], batch_size=nb_batch_size,validation_data=([aaa[43404+2975:43404+4975],aaa[43404+2975:43404+4975],bbb[43404+2975:43404+4975],bbb[43404+2975:43404+4975]], ddd[43404+2975:43404+4975]),epochs=nb_epoch, shuffle=True, verbose=1)

history = model1.fit([aaa[:43404+885],aaa[:43404+885],bbb[:43404+885],bbb[:43404+885]], ddd[:43404+885], batch_size=nb_batch_size,validation_data=([aaa[43404+885:49214],aaa[43404+885:49214],bbb[43404+885:49214],bbb[43404+885:49214]], ddd[43404+885:49214]),epochs=nb_epoch, shuffle=True, verbose=1)
#history = model1.fit([X1[:],X1[:],X2[:],X2[:]], T[:], batch_size=nb_batch_size,validation_split=0.0,epochs=nb_epoch, shuffle=True, verbose=1)
#validation_data=([a[43338:44209],b[43338:44209],c[43338:44209]], d[43338:44209]),
# history = model.fit([X[2500:],Y[2500:]], T[2500:], batch_size=nb_batch_size,
#validation_data=([X[:2500],Y[:2500]], T[:2500]),
# epochs= nb_epoch, shuffle=True, verbose=1)
# model1.save_weights('AMP_Prediction2.h5', overwrite=True)#882,38118
model1.save_weights('AMP_Prediction1.h5',overwrite=True)#869,38150
#model1.save_weights('AMP_Prediction1.h5',overwrite=True)#909,38058
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#model1 =create_model1()
#model1.load_weights('AMP_Prediction.h5')9
#model1.load_weights('AMP_Prediction1.h5')9
#model1.load_weights('AMP_Prediction111.h5')9
predictions_p = model1.predict([X1tt,X1tt,X2tt,X2tt])
predictions_p = model1.predict([Xpq,Xpq,Matr,Matr])
#predictions_p1 = model1.predict([X111,X111,Matr11,Matr11])

print(np.sum(predictions_p[990:,0]>0.5))
print(np.sum(predictions_p[:990,1]>0.5))











