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
import tensorflow as tf
from scipy import fftpack
from scipy.signal import hilbert
from keras import models
from PIL import Image
import io

def plot_gather(panel,amin=-1,amax=1,scale=True):
    if scale == False:
        plt.figure(figsize=(10,10))
        amin=np.amin(panel)/10000
        amax=np.amax(panel)/10000
        plt.imshow(panel,vmin=amin,vmax=amax)
    elif scale == True:
        plt.figure(figsize=(10,10))
        plt.imshow(rms_scale(panel),vmin=amin,vmax=amax)
    
def plot_images_for_activations(data,mode,dim1=1,dim_scalar=1):
    if mode == 0:
        fig, axs = plt.subplots(1,6,figsize=(15,15),sharex=True)
        axs[0].imshow(data[0],vmin=0,vmax=40,cmap='nipy_spectral')
        axs[0].set_ylabel('AFFT_10_20Hz')
        #axs[0].set_title(str(np.int(labels[ii])))
        axs[1].imshow(data[1],vmin=0,vmax=40,cmap='nipy_spectral')
        axs[1].set_ylabel('AFFT_20_30Hz')
        axs[2].imshow(data[2],vmin=0,vmax=40,cmap='nipy_spectral')
        axs[2].set_ylabel('AFFT_30_40Hz')
        axs[3].imshow(data[3],vmin=0,vmax=30,cmap='nipy_spectral')
        axs[3].set_ylabel('AMP_MAX')
        axs[4].imshow(data[4],vmin=0,vmax=1250,cmap='nipy_spectral')
        axs[4].set_ylabel('ENV_0_33')
        axs[5].imshow(data[5],vmin=0,vmax=1250,cmap='nipy_spectral')
        axs[5].set_ylabel('ENV_33_66')
        fig.tight_layout()
    elif mode == 1:
        fig, axs = plt.subplots(data.shape[0],dim1,figsize=(dim1*dim_scalar,6*dim_scalar),sharex=True,sharey=True)
        for i in np.arange(0,dim1):
            axs[0,i].imshow(data[0,:,:,i],aspect='auto')
            axs[1,i].imshow(data[1,:,:,i],aspect='auto')
            axs[2,i].imshow(data[2,:,:,i],aspect='auto')
            axs[3,i].imshow(data[3,:,:,i],aspect='auto')
            axs[4,i].imshow(data[4,:,:,i],aspect='auto')
            axs[5,i].imshow(data[5,:,:,i],aspect='auto')
            for j in np.arange(0,6):
                axs[j,i].set_xticks([])
                axs[j,i].set_yticks([])
        plt.subplots_adjust(wspace = 0, hspace = 0)
        
def generate_fmap(freqs,fsize,panel,fs=0.004,n=2):
    fmap=np.zeros((len(freqs)+1,panel.shape[1]))
    for i in np.arange(0,panel.shape[1]):
        amp=np.abs(fftpack.fft(panel[:,i]))
        amp=amp[0:amp.size//2]
        amp=20*np.log10(amp+1)
        #amp=minmax_scale(amp)
        freq = fftpack.fftfreq(panel.shape[0], fs)
        freq = freq[0:freq.size//2]
        if np.logical_or(np.sum(amp)==0,np.isinf(np.sum(amp))):
            fmap[:,i]=0
        else:
            for j in np.arange(0,len(freqs)):
                frange=np.where(np.logical_and(freq>=freqs[j],freq<freqs[j]+fsize))[0]
                fmap[j,i]=np.sum(amp[list(frange)])/len(freq[list(frange)])
            fmap[-1,i]=np.std(amp)
    fmap=np.repeat(np.reshape(fmap,(len(freqs)+1,19,38)),repeats=2,axis=1)
    return fmap

def rms_scale(panel,trc_mode=False):
    nsamp=panel.shape[0]
    divider=np.sqrt(np.sum(np.square(panel),axis=0)/nsamp)
    if trc_mode==False:
        tmp=np.where(divider==0)[0]
        divider[list(tmp)]=1
    g=1/divider
    return panel*g
    
def generate_img(panel,freqs=np.arange(10,60,10),fsize=10,amps=[0,0.25,0.7,100]):
    panel_env=np.abs(hilbert(rms_scale(panel),axis=0))
    panel_scaled=np.abs(rms_scale(panel))
    fmaps=generate_fmap(freqs,fsize,panel)
    fmaps_out=np.zeros((freqs.shape[0]+1,32,32,3))
    for j in np.arange(0,fmaps.shape[0]):
        fig=plt.figure(figsize=(38,38))
        if j==fmaps.shape[0]-1:
            plt.imshow(fmaps[j,:,:],vmin=0,vmax=40,cmap='nipy_spectral')
        else:
            plt.imshow(fmaps[j,:,:],vmin=1,vmax=15,cmap='nipy_spectral')
        fig.tight_layout()
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=4,bbox_inches='tight')
            fmaps_out[j]=np.asarray(Image.open(out).convert("RGB").resize((32,32),Image.ANTIALIAS),dtype=np.float32)/255
        plt.close(fig)
    astats=np.zeros((3,38,38))
    astats[0]=np.repeat(np.reshape(np.average(panel_scaled,axis=0),(19,38)),repeats=2,axis=0)
    astats[1]=np.repeat(np.reshape(np.amax(panel_scaled,axis=0),(19,38)),repeats=2,axis=0)
    astats[2]=np.repeat(np.reshape(np.std(panel_scaled,axis=0),(19,38)),repeats=2,axis=0)
    astats_out=np.zeros((3,32,32,3))
    for j in np.arange(0,len(astats)):
        fig=plt.figure(figsize=(38,38))
        if j==0:
            plt.imshow(astats[j,:,:],vmin=0.2,vmax=0.8,cmap='nipy_spectral')
        if j==1:
            plt.imshow(astats[j,:,:],vmin=0,vmax=30,cmap='nipy_spectral')
        elif j==2:
            plt.imshow(astats[j,:,:],vmin=0.6,vmax=1,cmap='nipy_spectral')
        fig.tight_layout()
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=4,bbox_inches='tight')
            astats_out[j]=np.asarray(Image.open(out).convert("RGB").resize((32,32),Image.ANTIALIAS),dtype=np.float32)/255
        plt.close(fig)
    ahist=np.zeros((len(amps)-1,722))
    for itrc in np.arange(0,panel_env.shape[1]):
        ahist[:,itrc]=np.histogram(panel_env[:,itrc],bins=amps)[0]
    ahist=np.repeat(np.reshape(ahist,(ahist.shape[0],19,38)),repeats=2,axis=1)
    ahist_out=np.zeros((3,32,32,3))
    for j in np.arange(0,len(ahist)):
        fig=plt.figure(figsize=(38,38))
        if j==0:
            plt.imshow(ahist[j,:,:],vmin=0,vmax=1250,cmap='nipy_spectral')
        elif j==1:
            plt.imshow(ahist[j,:,:],vmin=0,vmax=1250,cmap='nipy_spectral')
        elif j==2:
            plt.imshow(ahist[j,:,:],vmin=0,vmax=1250,cmap='nipy_spectral')
        fig.tight_layout()
        with io.BytesIO() as out:
            fig.savefig(out, format="png", dpi=4,bbox_inches='tight')
            ahist_out[j]=np.asarray(Image.open(out).convert("RGB").resize((32,32),Image.ANTIALIAS),dtype=np.float32)/255
        plt.close(fig)
    return np.concatenate((fmaps_out,astats_out,ahist_out),axis=0)

def predict(panel,receivers,model_fft,model_amp):
    imgs=generate_img(panel)
    imgs_fft=[np.reshape(imgs[0],(1,32,32,3)),
              np.reshape(imgs[1],(1,32,32,3)),
              np.reshape(imgs[2],(1,32,32,3))]
    imgs_amp=[np.reshape(imgs[7],(1,32,32,3)),
              np.reshape(imgs[9],(1,32,32,3)),
              np.reshape(imgs[10],(1,32,32,3))]
    p_fft=model_fft.predict(imgs_fft)[0]
    p_amp=model_amp.predict(imgs_amp)[0]
    #p_avg0=(p_fft[0]+p_amp[0])/2
    p_avg1=(p_fft[1]+p_amp[1])/2
    #print('p_noise = ',p_avg0)
    print('p_event = ',p_avg1)
    return p_avg1
    
def get_activations(panel,inds,model_fft,model_amp,layer):
    imgs=generate_img(panel)
    imgs_fft=[np.reshape(imgs[0],(1,32,32,3)),
              np.reshape(imgs[1],(1,32,32,3)),
              np.reshape(imgs[2],(1,32,32,3))]
    imgs_amp=[np.reshape(imgs[7],(1,32,32,3)),
              np.reshape(imgs[9],(1,32,32,3)),
              np.reshape(imgs[10],(1,32,32,3))]
    sh=list(model_amp.layers[layer*3].output_shape)[1:]
    sh=np.array(np.insert(sh,0,3),dtype=np.int)
    if layer == 0:
        sh=(3,32,32,3)
    a_fft=np.zeros(sh)
    a_amp=np.zeros(sh)
    for l,ll in enumerate(np.arange(layer*3,(layer+1)*3,dtype=np.int)):
        layer_outputs_fft = model_fft.layers[ll].output
        activation_model_fft = models.Model(inputs = model_fft.input, outputs = layer_outputs_fft)
        layer_outputs_amp = model_amp.layers[ll].output
        activation_model_amp = models.Model(inputs = model_amp.input, outputs = layer_outputs_amp) 
        a_fft[l,:]=activation_model_fft.predict(imgs_fft)[0]
        a_amp[l,:]=activation_model_amp.predict(imgs_amp)[0]
    return a_fft,a_amp