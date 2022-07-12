# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:25:11 2020

@author: Aengus.Daly
This file is copied from dnet
This file is for inference on Anser1 - training 18 + HIE
Now 3 runs
resnext with kernel = 5
LOO_res512_radam_aug_w_t2_gap_resnext_k5_mixup.py
Anser1 with 23 seiz patients
train form Ltrain_res512_resnxt_mixup_hski_iz_val_50.py
dir dna2/Seiz_25
Ltrain_hski_a1full2_resnxt_nz_k5_mixup_val50_Anser23.py
"""

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
import os
import keras.backend as K
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import numpy as np
from score_tool_DNN_resp_v import calc_roc
import GetData_perfile_512_1_nz
from keras.utils import np_utils
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import model_from_json
from keras.layers import Flatten, Input, BatchNormalization, Activation, Add, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, DepthwiseConv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import initializers
init = initializers.glorot_uniform(seed=717)
from keras.optimizers import Adam
filters = 32
import time


start_time = time.time()
epoch_length = 16
window_size = 53 # AD originally 61 but , 8 + (61-1) = 68, so 16 + (x-1) =68, so x = 53 for 16 sec window
dataseries = 'Anser2 files'
path_2 = '../'
kernel = 5
label = 'hski_mixupe_t50'
runs = 3
results = 'resnxt_hski_val'


def get_files(baby, dataseries='Anser1 files'):
    path = '../' + dataseries
    # path_anns = path +str('/Anns_per_file_rename_files')
    path_anns = path + str('/Anns_per_file')
    path_Matlab = path + str('/Matlab_EEG_files')
    file_anns = []
    file_Matlab = []

    for i in os.listdir(path_anns):
        if os.path.isfile(os.path.join(path_anns, i)) and baby in i:
            file_anns.append(i)
            file_mat_name = i[:-15]+str('SIGNAL')
            for q in os.listdir(path_Matlab):
                if os.path.isfile(os.path.join(path_Matlab, q)) and file_mat_name in q:
                    file_Matlab.append(q)

    if len(file_anns) != len(file_Matlab):
        print(' Anns not equal to Matlab no.',len(file_anns),len(file_Matlab), baby)

    return file_anns, file_Matlab


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

    pool5 = MaxPooling2D(pool_size=(1, 8), strides=(1, 1))(x)

    pool5 = Activation(("softmax"))(pool5)

    output_layer = Flatten()(pool5)

    return output_layer


# Model

def res_net(kernel = 5, filters = filter):

    input_layer = Input((512, 8 , 1))
    output_layer = build_model(input_layer, filters, init, kernel=kernel)

    model = Model(input_layer, output_layer)
    opt = Adam(lr=0.001, decay=1e-6)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return(model)


def crossval_mean_probability(baby, model, testX, testY):
    '''
    Getting the mean of three model training routines before the AUC is calculated. The weights of 3 loops
    (using different babies for validation) are loaded
    :param baby: name of test baby
    :param model: model architecture
    :param testX: the test EEG
    :param testY: the test labels
    :return: the mean of 3 probabilities
    '''

    data_gen = TimeseriesGenerator(np.asarray(testX), np_utils.to_categorical(np.asarray(testY)),
                                   length=512, sampling_rate=1, stride=32, batch_size=300, shuffle=False)
    probs = []
    for loop in range(runs):

        path = '../Benchmark_weights/'

        if loop == 0:
            saved_weights_str = '/best_weights_balance_CV_r1_round0' +str(label) + '_epoch44.hdf5'
        if loop == 1:
            saved_weights_str = '/best_weights_balance_CV_r1_round1' + str(label) + '_epoch49.hdf5'
        if loop == 2:
            saved_weights_str = '/best_weights_balance_CV_r1_round2' + str(label) + '_epoch44.hdf5'

        model.load_weights(path + saved_weights_str)

        # p = model.predict_generator(data_gen)[:, 1]
        p = model.predict(data_gen)[:, 1]
        p = movingaverage(p, window_size)

        probs.append(p)

    probs = np.asarray(probs)
    mean_probability = np.mean(probs, 0)

    return mean_probability


model = res_net(kernel = 5, filters=filters)

print(model.summary())
# AnSEr 1 full list - 23 seizures
file_list = ['024_AB', '025_AB', '027_AB', '029_A', '030_A', '031_A', '032_A',
       '033_AB', '037_AB', '042_A',  '047_A', '048_A', '051_AB',
       '053_AB', '054_AB', '057_A', '061_A', '063_A', '064_A', '085_A',
       '095_A', '096_AB', '102_A', '107_A', '116_A', '128_A', '142_AB',
       '144_A', '153_AB', '180_A', '181_A', '225_AB', '226_AB', '231_AB',
       '232_A', '233_A', '240_A',  '299_A', '307_A', '314_AB',
       '326_A', '408_A', '411_AB', '418_A', '434_AB', '446_A', '452_A',
       '460_AB', '465_A', '466_AB', '473_AB', '477_A', '506_AB', '529_A',
       '532_AB', '546_AB', '554_A', '570_A', '572_A', '587_A',
       '590_A', '601_A', '620_A', '621_A', '622_A', '628_A', '630_A',
       '639_A', '647_AB', '657_A', '658_A', '710_A', '713_A', '722_A',
       '730_AB', '734_A', '750_A', '751_A']

# file_list = ['029_A']

file_list = ['029_A']
# Anser 2 file
file_list = ['41']

probs_full = []
downsampled_y_full = []

for baby in file_list:

    file_anns, file_Matlab = get_files(baby, dataseries)

    Test = baby

    print('Test baby....', Test)
    print("--- %.0f seconds ---" % (time.time() - start_time))

    testX, testY = GetData_perfile_512_1_nz.getdatagen(file_anns, file_Matlab, dataseries, path_2)

    probs = crossval_mean_probability(baby, model, testX, testY)

    probs_full = np.append(probs_full, probs)

    downsampled_y = testY[::32][:-16]
    downsampled_y_full = np.append(downsampled_y_full, downsampled_y)

AUC = calc_roc(probs_full, downsampled_y_full, epoch_length=epoch_length) # Removed MAF
print('AUC %f, AUC90 %f' % (AUC))
print('runs', runs)
print("--- %.0f seconds ---" % (time.time() - start_time))
np.save('Results/Anser1_hski_rxt_Anser2_' +str(kernel) + str(label) + 'AUC.npy', AUC)
