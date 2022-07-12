from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import scipy.io as sio
import pdb
from keras.utils import to_categorical
from numpy import insert
import os

# slice_length = 256 # For the X's, at 32 Hz so 8 seconds = 8*32 = 256 (AD 8th June 2019)
slice_length = 512 # For the X's, at 32 Hz so 8 seconds = 16*32 = 512 (AD 8th July 2020)



def getdatagen(file_anns, file_Matlab, dataseries = 'anser1',path_2 = '../../../../media/ExtHDD02/Aengus/'):

    trainX = []
    trainY = []

    for name in file_anns:

        # file_mat_name = name[:-4] + str('_bp_SIGNAL')
        file_mat_name = name[:-15] + str('SIGNAL')

        # X = sio.loadmat(path_2 +dataseries +str('/Matlab_EEG_files_1/') + file_mat_name +'.mat' )#['EEG'] # X is 32 Hz EEG signal and 8 channels
        X = sio.loadmat(path_2 + dataseries + str(
            '/Matlab_EEG_files/0') + file_mat_name + '.mat')['EEG'] # X is 32 Hz EEG signal and 8 channels
        try:
            # Y = np.load(path_2 +dataseries + str('/Anns_per_file_rename_files/') + name) # Y is the label, 1 per second
            Y = np.load(
                path_2 + dataseries + str('/Anns_per_file/') + name)  # Y is the label, 1 per second
        # X = sio.loadmat(path_2 + dataseries
        # str('/Anns_per_file/') + name)['EEG'][:-7 * 32]  # X is 32 Hz EEG signal and 8 channels
        # try:
        #     Y = sio.loadmat('./Data/Annotations/' + name[:10] + '_map_8_1.mat')[
        #         'epoch_map']  # Y is the label, 1 per second
        except:
            Y = []
        if int(np.shape(X)[0]/32) != len(Y):
            print('Anns length different to EEG length', len(X), int(np.shape(X)[0]/32))

        trainY.extend(Y.repeat(32))
        trainX.extend(X.reshape(len(X),8,1))

    # AD I think there is no need for insert zeros for inference/prediction 12th Oct 2020
    # seconds = int(slice_length / 32)  # (So 256 = 8 seconds and 512 = 16 seconds)
    #
    # trainY = insert(trainY, 0, np.zeros((seconds * 32)))  # AD inserts zeros at start
    #
    # trainX_zeros = np.zeros((seconds * 32, 8, 1))
    #
    # trainX = np.append(trainX, trainX_zeros, axis=0)  # AD inserts zeros at end

    # AD the rationale for inserting zeros at start for labels and at end for signal is to adjust for TSG that
    # takes the first slice and for signal and first slice +1 for label in training
    #  Now it takes the first slice for signal and the first label

    trainX = np.asarray(trainX)

    # train_data_gen = TimeseriesGenerator(trainX, to_categorical(np.asarray(trainY),2), # previously just trainY
    #                                      length=slice_length, sampling_rate=1, stride=32, batch_size=300,
    #                                      shuffle=True)

    # return(trainX, trainY, train_data_gen)
    return trainX, trainY
