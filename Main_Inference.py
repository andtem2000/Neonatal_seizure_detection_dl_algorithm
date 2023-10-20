# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:25:11 2022

@author: Aengus.Daly

"""

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import os
import scipy.io as sio
import keras.backend as K
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # To be used for GPU use
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from Utils.Post_processing_AUC_calc import calc_roc
from keras.utils import np_utils
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Flatten, Input, Add, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, DepthwiseConv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
init = initializers.glorot_uniform(seed=717)
from keras.optimizers import Adam # RAdam used in training put inference here so Adam optimizer has no impact
import time
import config_1 as cfg
from Utils.ConvNet import build_model
from Utils.Inference_probs import mean_maf_probability

start_time = time.time()

def getdata(Baby,eeg_file_path,eeg_channels, input_sampling_rate,):

    trainX = []
    trainY = []

    X = sio.loadmat(eeg_file_path+str('eeg')+str(Baby) + str('_SIGNAL.mat'))['EEG'] # X is 32 Hz EEG signal and 18 channels
    try:
        Y = sio.loadmat(eeg_file_path + str('annotations_2017.mat'))['annotat_new'][0][Baby-1]  # Y is the label, 1 per second, choose baby with index Baby-1
        Y = np.sum(Y, axis =0) # For consensus anns
        Y = np.where(Y == 3,1,0) # For consensus anns
    except:
        Y = []
    if int(np.shape(X)[0]/input_sampling_rate) != len(Y):
        print('Anns length different to EEG length', len(X), int(np.shape(X)[0]/input_sampling_rate))

    trainY.extend(Y.repeat(input_sampling_rate))
    trainX.extend(X.reshape(len(X),eeg_channels,1))

    return trainX, trainY


def res_net(kernel, filters, eeg_channels,input_length):

    input_layer = Input((input_length, eeg_channels , 1))
    output_layer = build_model(input_layer, filters, init, kernel=kernel, eeg_channels=eeg_channels)

    model = Model(input_layer, output_layer)
    opt = Adam(lr=0.001, decay=1e-6) # This is not training only inference so these parameters need not be changed
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return(model)


def inference(hski_file = cfg.hski_baby, eeg_file_path = cfg.path_test_files, weights_path=cfg.path_weights, results_path = cfg.path_results,
              name=cfg.results_name, kernel=cfg.kernel, filters = cfg.filters, runs = cfg.runs, eeg_channels = cfg.eeg_channels,
              input_sampling_rate=cfg.input_sampling_rate, epoch_length=cfg.epoch_length):
    probs_full = []
    downsampled_y_full = []
    input_length = epoch_length * input_sampling_rate  # here it is 512


    for baby in range(hski_file,hski_file+1): # total of 79 EEG_files/babies, only doing inference on 1 here

        print('Test baby....', baby)
        print("--- %.0f seconds ---" % (time.time() - start_time))

        testX, testY = getdata(baby, eeg_file_path,eeg_channels, input_sampling_rate)

        model = res_net(kernel, filters, eeg_channels,input_length)

        print(model.summary())

        probs = mean_maf_probability(model, testX, testY,weights_path, runs, input_length, epoch_length)

        probs_full = np.append(probs_full, probs)

        downsampled_y = testY[::input_sampling_rate][:-epoch_length]
        downsampled_y_full = np.append(downsampled_y_full, downsampled_y)

    AUC = calc_roc(probs_full, downsampled_y_full, epoch_length=epoch_length)
    print('AUC %f, AUC90 %f' % (AUC))
    print('runs', runs)
    print("--- %.0f seconds ---" % (time.time() - start_time))
    np.save(results_path + name + '.npy', AUC)
    print("--- %.0f seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    # LOGGER.info("Script started...")

    # run_model = True
    # save_results = False
    # check_options(run_models, save_results)

    inference()