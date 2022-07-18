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
from Neonatal_Seizure_Resnext_algorithm.score_tool_DNN_resp_v2 import calc_roc
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

start_time = time.time()
epoch_length = 16 # Lenght of input EEG signal in seconds
window_size = 69 - epoch_length # 53 for 16 sec window, used in Moving Average Filter
eeg_channels = 18 # 18 for Helsinki files
path_1 = '../Benchmark_weights/'
path_2 = '../Helsinki files/'
label = 'run_hski_1'
hski_baby = 4
runs = 3 # no. of sets of weights used.  This corresponds to the no. of training runs.
# Cannot change the following parameters in test
filters = 32
kernel = 5


def getdata(Baby,path_2 = '../Helsinki files/'):

    trainX = []
    trainY = []

    X = sio.loadmat(path_2+str('eeg')+str(Baby) + str('_SIGNAL.mat'))['EEG'] # X is 32 Hz EEG signal and 18 channels
    try:
        Y = sio.loadmat(path_2 + str('annotations_2017.mat'))['annotat_new'][0][Baby-1]  # Y is the label, 1 per second, choose baby with index Baby-1
        Y = np.sum(Y, axis =0) # For consensus anns
        Y = np.where(Y == 3,1,0) # For consensus anns
    except:
        Y = []
    if int(np.shape(X)[0]/32) != len(Y):
        print('Anns length different to EEG length', len(X), int(np.shape(X)[0]/32))

    trainY.extend(Y.repeat(32))
    trainX.extend(X.reshape(len(X),18,1))

    return trainX, trainY


def movingaverage(data, window):
    '''

    :param data: the vector which the MAF will be applied to
    :param window: the size of the moving average window
    :return: data after the MAF has been applied
    '''
    data = data
    window = np.ones(int(window)) / float(window_size)
    return np.convolve(data, window, "same")


def fe_block(block_input, filters, init):

    pad = 'valid'

    conv1 = Conv2D(filters*1, (3,1), activation='relu', padding=pad, kernel_initializer=init)(block_input)
    conv1 = Conv2D(filters*1, (3,1), activation='relu', padding=pad, kernel_initializer=init)(conv1)
    conv1 = Conv2D(filters * 1, (3, 1), activation=None, padding=pad, kernel_initializer=init)(conv1)

    skip_conv = Cropping2D(cropping=((3, 3), (0, 0)))(block_input)

    res1 = Add()([conv1, skip_conv])

    atv1 = Activation('relu')(res1)
    norm1 = BatchNormalization()(atv1)

    return norm1


def fe_block_dep(block_input, filters, init, dm = 1, kernel=3):

    pad = 'valid'
    adj = int((kernel-3)/2)
    conv1 = DepthwiseConv2D(kernel_size = (kernel,1), depth_multiplier=dm, activation='relu', padding=pad, kernel_initializer=init)(block_input)

    conv1 = Conv2D(filters*2, (3,1), activation='relu', padding=pad, kernel_initializer=init)(conv1)
    conv1 = Conv2D(filters * 1, (1, 1), activation=None, padding=pad, kernel_initializer=init)(conv1)
    skip_conv = Cropping2D(cropping=((2+adj, 2+adj), (0, 0)))(block_input)

    res1 = Add()([conv1, skip_conv])

    atv1 = Activation('relu')(res1)
    norm1 = BatchNormalization()(atv1)
    return norm1


def build_model(input_layer, filters, init, kernel):

    x = fe_block(input_layer, filters, init)
    x = AveragePooling2D(pool_size=(4, 1), strides=(3, 1))(x)

    x = fe_block_dep(x, filters, init, dm = 1, kernel=kernel)
    x = AveragePooling2D(pool_size=(4, 1), strides=(3, 1))(x)

    x = fe_block_dep(x, filters, init, dm=2, kernel=kernel)
    x = AveragePooling2D(pool_size=(4, 1), strides=(3, 1))(x)

    x = fe_block_dep(x, filters, init, dm=2, kernel=kernel)

    x = Conv2D(filters=2, kernel_size=(2, 1), strides=(1, 1), activation="relu", padding='valid', kernel_initializer=init)(x)
    x = (AveragePooling2D(pool_size=(K.int_shape(x)[-3], 1), strides=(1, 1)))(x)

    pool5 = MaxPooling2D(pool_size=(1, eeg_channels), strides=(1, 1))(x)

    pool5 = Activation(("softmax"))(pool5)

    output_layer = Flatten()(pool5)

    return output_layer


# Model

def res_net(kernel = 5, filters = filters):

    input_layer = Input((512, eeg_channels , 1))
    output_layer = build_model(input_layer, filters, init, kernel=kernel)

    model = Model(input_layer, output_layer)
    opt = Adam(lr=0.001, decay=1e-6) # This is not training only inference so these parameters need not be changed
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return(model)


def crossval_mean_probability(model, testX, testY, path = path_1):
    """
    Getting the mean of three model training routine runs before the AUC is calculated.
    A moving average filter is applied.
    The weights of 3 runs
    :param model: model architecture
    :param testX: the test EEG
    :param testY: the test labels
    :param path: path to saved weights
    :return: the mean of 3 probabilities
    """

    data_gen = TimeseriesGenerator(np.asarray(testX), np_utils.to_categorical(np.asarray(testY)),
                                   length=512, sampling_rate=1, stride=32, batch_size=300, shuffle=False)
    probs = []
    for loop in range(runs):

        if loop == 0:
            saved_weights_str = path + 'best_weights_run0_hski_mixup.hdf5'
        if loop == 1:
            saved_weights_str = path + 'best_weights_run1_hski_mixup.hdf5'
        if loop == 2:
            saved_weights_str = path + 'best_weights_run2_hski_mixup.hdf5'

        model.load_weights(saved_weights_str)

        p = model.predict(data_gen)[:, 1]
        p = movingaverage(p, window_size)

        probs.append(p)

    probs = np.asarray(probs)
    mean_probability = np.mean(probs, 0)

    return mean_probability


model = res_net(kernel = 5, filters=filters)

print(model.summary())

probs_full = []
downsampled_y_full = []

for baby in range(hski_baby,hski_baby+1): # total of 79 Helsinki files/babies, only doing infrence on 1 here

    print('Test baby....', baby)
    print("--- %.0f seconds ---" % (time.time() - start_time))

    testX, testY = getdata(baby, path_2)

    probs = crossval_mean_probability(model, testX, testY, path = path_1)

    probs_full = np.append(probs_full, probs)

    downsampled_y = testY[::32][:-epoch_length]
    downsampled_y_full = np.append(downsampled_y_full, downsampled_y)

AUC = calc_roc(probs_full, downsampled_y_full, epoch_length=epoch_length) # Removed MAF
print('AUC %f, AUC90 %f' % (AUC))
print('runs', runs)
print("--- %.0f seconds ---" % (time.time() - start_time))
np.save('../Results/hski_rxt_ + '+ str(label) + '.npy', AUC)
