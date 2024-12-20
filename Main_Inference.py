# -*- coding: utf-8 -*-
"""
Created on Mon Sept 11 11:25:11 2023

@author: Aengus.Daly

"""

import scipy.io as sio
import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import np_utils
import time
from ConvNet import res_net
from pathlib import Path

start_time = time.time()
epoch_length = 16  # Length of epoch/window input of EEG signal in seconds; needs to be greater or equal to 16
epoch_shift = 1 # Epoch/window shift in seconds
maf_window_parameter = 69 # In seconds used in Moving Average Filter
file_list = ["./EEG_files/eeg1_SIGNAL.mat", "./EEG_files/eeg4_SIGNAL.mat"]  # List of EEG signal files
weights_list = ['./Benchmark_weights/best_weights_run0_hski_trained.hdf5',
                './Benchmark_weights/best_weights_run1_hski_trained.hdf5',
                './Benchmark_weights/best_weights_run2_hski_trained.hdf5']  # List of file names for model weights
results_path = './Results/'  # Folder for storing results/probability output on a per EEG signal file basis
# If you want to use the model weights that include the pseudo labelled data use the following as the weights_list
# weights_list = ['./Benchmark_weights/best_weights_run0_hski_plus_pslabel_HIEInfant.hdf5',
#                './Benchmark_weights/best_weights_run0_hski_plus_pslabel_HIEInfant.hdf5',
#                './Benchmark_weights/best_weights_run0_hski_plus_pslabel_HIEInfant.hdf5']
# List of file names for model weights after training with pseudo labels

input_sampling_rate = 32  # 32 Hz sampling rate for the EEG signal, not to be changed


def getdata(baby):
    """
    Function to generate windows of data from the EEG signal data
    :param baby: file name of the EEG signal data
    :return: series of data windows, no. of eeg channels
    """

    test_x = []

    X = sio.loadmat(str(baby))['EEG']  # X is 32 Hz EEG signal and 18 channels
    no_eeg_channels = np.shape(X)[1]
    test_x.extend(X.reshape(len(X), no_eeg_channels,1))

    # Generate data slices of (length= epoch_length*input_sampling_rate), shifted by (stride = input_sampling rate*epoch_shift)
    testy = np.ones((len(test_x)))  # this is needed for TimeSeriesGenerator (TSG)

    data_gen = TimeseriesGenerator(np.asarray(test_x), np_utils.to_categorical(np.asarray(testy)),
                                   length=epoch_length * input_sampling_rate, sampling_rate=1,
                                   stride=input_sampling_rate * epoch_shift, batch_size=300, shuffle=False)

    return data_gen, no_eeg_channels


def moving_average_filter(data):
    """
    Moving average filter function that is applied to outputted probabilities
    :param data: the vector to which the MAF will be applied
    :return: data after the MAF has been applied
    """
    data = data
    window = (maf_window_parameter - epoch_length)/epoch_shift # default case is (69 - 16)/ 1 = 53 seconds
    window = np.ones(int(window)) / float(window)
    return np.convolve(data, window, "same")


def inference():
    """ Primary function, run below via __main__
    Outputs to file the probability trace for each individual EEG signal file inputted
    """

    first = True  # first item of loop indicator for printing one model summary etc

    for signal_file in file_list:

        print('Started inference for EEG folder/file....', signal_file)
        data_windowed, no_eeg_channels = getdata(signal_file)

        model = res_net(no_eeg_channels, input_length=epoch_length*input_sampling_rate)

        if first:
            print(model.summary())
            print('No. of model weights used...', len(weights_list))
            first = False

        probs_full = []

        for weights_str in weights_list:
            model.load_weights(weights_str)

            probs = model.predict(data_windowed)[:, 1]
            probs_full.append(probs)  # Appending probs so that they can be averaged

        probs_full = np.asarray(probs_full)
        probs_full = np.mean(probs_full, 0)
        probs_full = moving_average_filter(probs_full) # Applying moving average filter to probs
        # probs_full = np.append(probs_full, probs) # To be used if concatenating probs for many EEG signal files together for outputting to one file

        file_name = Path(signal_file).name # Used for naming output file by using input file name
        results_file_name = results_path + 'probs_' + file_name[:-4] + '.npy'
        np.save(results_file_name, probs_full)
        print('Completed inference for EEG folder/file....', signal_file)
        print('Probability results created in folder/file....',results_file_name)
        print("--- %.0f seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    inference()
