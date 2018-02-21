# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 22:19:22 2018

@author: agarwal.270
"""
import numpy as np
import tensorflow as tf
#tf.reset_default_graph()
from tensorflow import set_random_seed
#set_random_seed(1)
import keras as kr
from keras.models import Model # Neural-Network model
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import MaxPooling2D, Input, concatenate
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
import keras.regularizers as regularizers
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
#from sklearn.model_selection import train_test_split
from keras.models import load_model
import pandas as pd
import timeit
import os
import scipy.io
import peakutils as pk
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#config = tf.ConfigProto(
#        device_count = {'GPU': 0}
#    )
#sess = tf.Session(config=config)
# In[1]

#close('all')

def data_setup_predict(arr,w):
    arr0=np.array([arr[i:i+w,0] for i in range(len(arr)-w+1)])
    arr1=np.array([arr[i:i+w,1] for i in range(len(arr)-w+1)])
    arr2=np.array([arr[i:i+w,2] for i in range(len(arr)-w+1)])
    X=np.stack((arr0.reshape(arr0.shape+(1,)),arr1.reshape(arr1.shape+(1,)),arr2.reshape(arr2.shape+(1,))),axis=2)
    X_test_Acc=X.reshape(X.shape[0],1,X.shape[1],3)
    del arr0,arr1,arr2,X
    arr0=np.array([arr[i:i+w,3] for i in range(len(arr)-w+1)])
    arr1=np.array([arr[i:i+w,4] for i in range(len(arr)-w+1)])
    arr2=np.array([arr[i:i+w,5] for i in range(len(arr)-w+1)])
    X=np.stack((arr0.reshape(arr0.shape+(1,)),arr1.reshape(arr1.shape+(1,)),arr2.reshape(arr2.shape+(1,))),axis=2)
    X_test_LED=X.reshape(X.shape[0],1,X.shape[1],3)
    return X_test_Acc,X_test_LED

def CNN_Model(input_1):
    initizer_TN=kr.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
    initizer_0=kr.initializers.zeros()

    C1=Conv2D(64,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(input_1)
    A1=Activation('relu')(C1)
    C2=Conv2D(64,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A1)
    A2=Activation('relu')(C2)

    C3=Conv2D(128,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A2)
    A3=Activation('relu')(C3)
    C4=Conv2D(128,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A3)
    A4=Activation('relu')(C4)

    C5=Conv2D(256,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A4)
    A5=Activation('relu')(C5)
    C6=Conv2D(256,(1, 3), padding='valid',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A5)
    A6=Activation('relu')(C6)

    C7=Conv2D(512,(1, 3), padding='valid',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A6)
    A7=Activation('relu')(C7)
    M1=MaxPooling2D(pool_size=(1,2))(A7)

    F1=Flatten()(M1)
    return F1
#
# def CNN_Model(input_1):
#     initizer_TN=kr.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
#     initizer_0=kr.initializers.zeros()
#
#     C1=Conv2D(64,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(input_1)
#     A1=Activation('relu')(C1)
#     C2=Conv2D(64,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A1)
#     A2=Activation('relu')(C2)
#
#     C3=Conv2D(128,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A2)
#     A3=Activation('relu')(C3)
#     C4=Conv2D(128,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A3)
#     A4=Activation('relu')(C4)
#
#     C5=Conv2D(256,(1, 3), padding='same',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A4)
#     A5=Activation('relu')(C5)
#     C6=Conv2D(256,(1, 3), padding='valid',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A5)
#     A6=Activation('relu')(C6)
#
#     C7=Conv2D(512,(1, 3), padding='valid',kernel_initializer=initizer_TN,kernel_regularizer=regularizers.l2(0.00001))(A6)
#     A7=Activation('relu')(C7)
#     M1=MaxPooling2D(pool_size=(1,2))(A7)
#
#     F1=Flatten()(M1)
#     return F1

def data_setup_predict(arr,w):
    arr0=np.array([arr[i:i+w,0] for i in range(len(arr)-w+1)])
    arr1=np.array([arr[i:i+w,1] for i in range(len(arr)-w+1)])
    arr2=np.array([arr[i:i+w,2] for i in range(len(arr)-w+1)])
    X=np.stack((arr0.reshape(arr0.shape+(1,)),arr1.reshape(arr1.shape+(1,)),arr2.reshape(arr2.shape+(1,))),axis=2)
    X_test_Acc=X.reshape(X.shape[0],1,X.shape[1],3)
    del arr0,arr1,arr2,X
    arr0=np.array([arr[i:i+w,3] for i in range(len(arr)-w+1)])
    arr1=np.array([arr[i:i+w,4] for i in range(len(arr)-w+1)])
    arr2=np.array([arr[i:i+w,5] for i in range(len(arr)-w+1)])
    X=np.stack((arr0.reshape(arr0.shape+(1,)),arr1.reshape(arr1.shape+(1,)),arr2.reshape(arr2.shape+(1,))),axis=2)
    X_test_LED=X.reshape(X.shape[0],1,X.shape[1],3)
    return X_test_Acc,X_test_LED

def HR_predict(arr,w):
    Fs=25
    arr0=np.array([list(arr[4*Fs*i:(4*Fs*i)+w]) for i in np.arange(np.round((len(arr)-w+1)/(4*Fs))).astype(int)])
    HR=arr0.sum(axis=1)/(w*(1/25)/60) # HR in BPM
    return HR

def predict_likelihood(final_model,val,time_input):
    batches=1024
    w = 11
    time_test = time_input[:-(w-1)]
    ts = list(map(lambda x:x.timestamp()*1000,list(time_test)))
    ts = np.array(ts)
    data_test = val
    # search for max length array in list_test_RA
    X_t,Y_test=data_test[:,[0,1,2,6,7,8]],data_test[:,-1]
    X_test_Acc,X_test_LED=data_setup_predict(X_t,w)
    # Reshaping for 3 branched model
    Acc_RMS=np.mean(((X_test_Acc[:,:,:,0])**2+(X_test_Acc[:,:,:,1])**2+(X_test_Acc[:,:,:,2])**2)**0.5)
    X_test_LED=X_test_LED.reshape((-1,1,11,1,3))
    Y_test=Y_test[:-(w-1)]  # make Y_test same length as predictions
    d=4
    ans1=final_model.predict([X_test_LED[:,:,:,:,0],X_test_LED[:,:,:,:,1],X_test_LED[:,:,:,:,2],X_test_Acc],batch_size=batches)
    #make flat predictions pointy
    ans1_copy=ans1*1
    idx_1=np.where(ans1>0.99)[0]
    dif_idx=np.diff(idx_1)
    tempr=np.array([]).astype(int)
    for i in range(len(dif_idx)):
        if dif_idx[i]==1:
            tempr=np.append(tempr,i)
        else:
            tempr=np.append(tempr,i)
            if len(tempr)==1:
                continue
            pk_idx=int((max(tempr)+min(tempr))/2)
            tempr=tempr[tempr!=pk_idx] #remove the middle index
            ans1_copy[idx_1[tempr]]=0.99;del tempr  # assign all indices a lowerr prob than 1
            tempr=np.array([]).astype(int)
    # Scale predictions
    Fs_pred=25;BPM=80;
    ans3=0.95*ans1_copy
    BPS=BPM/60
    pred =(1/4)*((1-(BPS/Fs_pred))/(BPS/Fs_pred))*(ans3/(1-ans3)) # 1/4 mulltiplied for sake similarity with ju's plot
    return ts,pred,Acc_RMS

