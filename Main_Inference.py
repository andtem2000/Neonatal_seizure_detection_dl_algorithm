# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:25:11 2022

@author: Aengus.Daly

"""

import os
import scipy.io as sio
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # To be used for GPU use
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras import initializers
init = initializers.glorot_uniform(seed=717)
from keras.optimizers import Adam # RAdam used in training put inference here so Adam optimizer has no impact
import time
import config_1 as cfg
from Utils.ConvNet import build_model
from Utils.Inference_probs import mean_maf_probability

start_time = time.time()

def getdata(Baby,eeg_file_path, input_sampling_rate,):

    trainX = []
    trainY = []

    X = sio.loadmat(eeg_file_path+str(Baby))['EEG'] # X is 32 Hz EEG signal and 18 channels
    no_eeg_channels = np.shape(X)[1]
    trainX.extend(X.reshape(len(X),no_eeg_channels,1))

    return trainX, no_eeg_channels


def res_net(eeg_channels,input_length):

    input_layer = Input((input_length, eeg_channels , 1))
    output_layer = build_model(input_layer, init, eeg_channels=eeg_channels)

    model = Model(input_layer, output_layer)
    opt = Adam(lr=0.001, decay=1e-6) # This is not training only inference so these parameters need not be changed
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return(model)


def inference(eeg_file_path = cfg.path_test_files, weights_path=cfg.path_weights, results_path = cfg.path_results,
            runs = cfg.runs,
              input_sampling_rate=cfg.input_sampling_rate, epoch_length=cfg.epoch_length):
    probs_full = []
    input_length = epoch_length * input_sampling_rate  # here it is 512

    file_list = cfg.file_list
    for baby in (file_list):

        print('EEG file started inference....', baby)
        print("--- %.0f seconds ---" % (time.time() - start_time))

        testX, no_eeg_channels = getdata(baby, eeg_file_path, input_sampling_rate)

        model = res_net(no_eeg_channels,input_length)

        print(model.summary())

        probs = mean_maf_probability(model, testX,weights_path, runs, input_length, epoch_length)
        probs_full = probs
        # probs_full = np.append(probs_full, probs) # to be used for concatenating probs

        print('For no. of model runs...', runs)
        results_file_name = results_path + 'probs_'+ baby[:-4] + '.npy'
        np.save(results_file_name, probs_full)
        print('Probabilities created in folder....',results_file_name)
        print("--- %.0f seconds ---" % (time.time() - start_time))
        print("--- %.0f seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    # LOGGER.info("Script started...")

    # run_model = True
    # save_results = False
    # check_options(run_models, save_results)

    inference()