#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 17:39:32 2020

@author: mmezyk
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
import tensorflow as tf
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def plot_precomputed_fm(igathers,data,labels,ax_names):
    fig, axs = plt.subplots(data.shape[0],len(igathers),figsize=(30,7.5),sharex=True)
    for i,ii in enumerate(igathers):
        axs[0, i].imshow(data[0,ii])
        axs[0, i].set_ylabel(ax_names[0])
        axs[0, i].set_title(str(np.int(labels[i])))
        axs[1, i].imshow(data[1,ii])
        axs[1, i].set_ylabel(ax_names[1])
        axs[2, i].imshow(data[2,ii])
        axs[2, i].set_ylabel(ax_names[2])
        axs[0, i].set_xticks([])
        axs[0, i].set_yticks([])
        axs[1, i].set_xticks([])
        axs[1, i].set_yticks([])
        axs[2, i].set_xticks([])
        axs[2, i].set_yticks([])
    fig.tight_layout()
    
def define_model():
    input1 = keras.Input(shape=(32, 32, 3), name="original_img1")
    input2 = keras.Input(shape=(32, 32, 3), name="original_img2")
    input3 = keras.Input(shape=(32, 32, 3), name="original_img3")
    x1 = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(input1)
    x2 = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(input2)
    x3 = layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(input3)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x3 = layers.MaxPooling2D((2, 2))(x3)
    x1 = layers.Dropout(0.2)(x1)
    x2 = layers.Dropout(0.2)(x2)
    x3 = layers.Dropout(0.2)(x3)
    x1 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(x1)
    x2 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(x2)
    x3 = layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(x3)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x3 = layers.MaxPooling2D((2, 2))(x3)
    x1 = layers.Dropout(0.2)(x1)
    x2 = layers.Dropout(0.2)(x2)
    x3 = layers.Dropout(0.2)(x3)
    x1 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(x1)
    x2 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(x2)
    x3 = layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform', padding='same')(x3)
    x1 = layers.MaxPooling2D((2, 2))(x1)
    x2 = layers.MaxPooling2D((2, 2))(x2)
    x3 = layers.MaxPooling2D((2, 2))(x3)
    x1 = layers.Dropout(0.2)(x1)
    x2 = layers.Dropout(0.2)(x2)
    x3 = layers.Dropout(0.2)(x3)
    x1 = layers.Flatten()(x1)
    x2 = layers.Flatten()(x2)
    x3 = layers.Flatten()(x3)
    x1 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x1)
    x2 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x2)
    x3 = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x3)
    x = layers.concatenate([x1,x2,x3])
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(2, activation='softmax')(x)
    model = tf.keras.models.Model(inputs = [input1,input2,input3], outputs = output)
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_amp,data_fft,labels,test_size=0.25,random_state=28):
    X_train, X_test, y_train, y_test = train_test_split(np.arange(0,data_amp.shape[1]), 
                                                        np.arange(0,data_amp.shape[1]), 
                                                        test_size=test_size, 
                                                        random_state=random_state)
    print('Training of model_amp started.')
    model_amp = define_model()
    model_amp_history = model_amp.fit([data_amp[0,list(X_train)],data_amp[1,list(X_train)],data_amp[1,list(X_train)]],
                                                labels[list(y_train)],
                                                epochs=200, 
                                                batch_size=16, 
                                                verbose=0)
    print('Training of model_fft started.')
    model_fft = define_model()
    model_fft_history = model_fft.fit([data_fft[0,list(X_train)],data_fft[1,list(X_train)],data_fft[1,list(X_train)]],
                                                labels[list(y_train)],
                                                epochs=200, 
                                                batch_size=16, 
                                                verbose=0)
    print('Running prediction on test set.')
    preds_amp=model_amp.predict([data_amp[0,list(X_test)],
                                 data_amp[1,list(X_test)],
                                 data_amp[2,list(X_test)]])
    preds_fft=model_fft.predict([data_fft[0,list(X_test)],
                                 data_fft[1,list(X_test)],
                                 data_fft[2,list(X_test)]])
    preds_avg=np.average((preds_amp[:,1],preds_fft[:,1]),axis=0)
    return model_amp,model_fft,model_amp_history,model_fft_history,labels[list(y_test),1],preds_avg