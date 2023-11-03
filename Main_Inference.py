# -*- coding: utf-8 -*-
"""
Created on Mon Sept 11 11:25:11 2023

@author: Aengus.Daly

"""

import os
import os.path
import scipy.io as sio
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # To be used for GPU use
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from keras.models import Model, load_model
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import np_utils
import time
from Utils.ConvNet import res_net


start_time = time.time()
weights_path = './Benchmark_weights/'  # Folder with the model weights
eeg_file_path = 'EEG_files/'  # Folder with EEG signal data
results_path = './Results/' # Folder for storing results output
model_path = 'Utils/ConvNet_model.keras'  # Path for keras model
file_list = ["eeg1_SIGNAL.mat", "eeg4_SIGNAL.mat"]  # List of EEG signal files which are located in eeg_file_path
epoch_length = 16  # Length of epoch/window input of EEG signal in seconds
epoch_shift = 1 # Epoch/window shift in seconds
input_sampling_rate = 32  # 32 Hz is the input signal sampling rate
sample_rate = 1 # Sampling rate per input_sample_rate used to make data epoch/window slices in TSG; unlikely to be adjusted
runs = 3  # No. of model weights used; weights were generated via different random initializations during training runs.
maf_window_size = 69 - epoch_length  # 53 for 16 sec window, used in Moving Average Filter
weights_list = {"1": ['best_weights_run0_hski_trained.hdf5'],
                "2": ['best_weights_run0_hski_trained.hdf5','best_weights_run1_hski_trained.hdf5'],
                "3": ['best_weights_run0_hski_trained.hdf5','best_weights_run1_hski_trained.hdf5','best_weights_run2_hski_trained.hdf5']}
                # Dictionary -  runs: List of model weights' file names


def getdata(baby, file_path):
    """
    Function to process the EEG signal data
    :param baby: file name of the EEG signal data
    :param file_path: file_path for the above file name
    :return: data after reshaping for TSG, no. of eeg channels
    """

    test_x = []

    X = sio.loadmat(file_path+str(baby))['EEG']  # X is 32 Hz EEG signal and 18 channels
    no_eeg_channels = np.shape(X)[1]
    test_x.extend(X.reshape(len(X), no_eeg_channels,1))

    return test_x, no_eeg_channels


def movingaverage(data, epoch_len):
    """
    Moving average filter used on outputted probabilities
    :param data: the vector which the MAF will be applied to
    :param epoch_len: used for calc of the size of the moving average window
    :return: data after the MAF has been applied
    """
    data = data
    window = 69 - epoch_len
    window = np.ones(int(window)) / float(window)
    return np.convolve(data, window, "same")


def inference():
    """ Primary function, run below via __main__"""

    for baby in file_list:

        print('EEG file started inference....', baby)
        print("--- %.0f seconds ---" % (time.time() - start_time))

        testx, no_eeg_channels = getdata(baby, eeg_file_path)
        if os.path.isfile(model_path) == False:  # compile and save model file if it does not exist
            model = res_net(no_eeg_channels,input_length=epoch_length*input_sampling_rate)
            model.save(model_path)
        model = load_model(model_path)
        print(model.summary())

        # Generate data slices of (length= epoch_length*input_sampling_rate), shifted by (stride = input_sampling rate*epoch_shift), and sampling rate = sample rate
        testy = np.ones((len(testx)))  # this is needed for TimeSeriesGenerator (TSG)
        data_gen = TimeseriesGenerator(np.asarray(testx), np_utils.to_categorical(np.asarray(testy)),
                                       length=epoch_length*input_sampling_rate, sampling_rate=sample_rate, 
                                       stride=input_sampling_rate*epoch_shift, batch_size=300, shuffle=False)
        probs_full = []

        for weights_str in weights_list[str(runs)]:
            model.load_weights(weights_path + weights_str)

            probs = model.predict(data_gen)[:, 1]
            probs = movingaverage(probs, epoch_length)  # Applying moving average filter to probs
            probs_full.append(probs)  # Appending probs so that they can be averaged

        probs_full = np.asarray(probs_full)
        probs_full = np.mean(probs_full, 0)
        # probs_full = np.append(probs_full, probs) # To be used for concatenating probs

        print('For no. of model runs...', runs)
        results_file_name = results_path + 'probs_' + baby[:-4] + '.npy'
        np.save(results_file_name, probs_full)
        print('Probabilities created in folder/file....',results_file_name)
        print("--- %.0f seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    
    inference()