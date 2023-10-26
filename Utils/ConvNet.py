"""File  to generate the ConvNet_model.keras file"""

import keras.backend as k
from keras.layers import Flatten, Input, Add, Cropping2D
from keras.layers.core import Activation
from keras.layers.convolutional import Conv2D, DepthwiseConv2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras import initializers
from keras.optimizers import Adam  # RAdam used in training put inference here so Adam optimizer has no impact
filters_size = 32  # Cannot change the following parameters in test
kernel_size = 5
init = initializers.glorot_uniform(seed=717)


def fe_block(block_input, filters, initial):

    pad = 'valid'

    conv1 = Conv2D(filters * 1, (3, 1), activation='relu', padding=pad, kernel_initializer=initial)(block_input)
    conv1 = Conv2D(filters * 1, (3, 1), activation='relu', padding=pad, kernel_initializer=initial)(conv1)
    conv1 = Conv2D(filters * 1, (3, 1), activation=None, padding=pad, kernel_initializer=initial)(conv1)

    skip_conv = Cropping2D(cropping=((3, 3), (0, 0)))(block_input)

    res1 = Add()([conv1, skip_conv])

    atv1 = Activation('relu')(res1)
    norm1 = BatchNormalization()(atv1)

    return norm1


def fe_block_dep(block_input, filters, initial, dm=1, kernel=kernel_size):

    pad = 'valid'
    adj = int((kernel-3)/2)
    conv1 = DepthwiseConv2D(kernel_size = (kernel,1), depth_multiplier=dm, activation='relu', padding=pad, kernel_initializer=initial)(block_input)

    conv1 = Conv2D(filters * 2, (3,1), activation='relu', padding=pad, kernel_initializer=initial)(conv1)
    conv1 = Conv2D(filters * 1, (1, 1), activation=None, padding=pad, kernel_initializer=initial)(conv1)
    skip_conv = Cropping2D(cropping=((2+adj, 2+adj), (0, 0)))(block_input)

    res1 = Add()([conv1, skip_conv])

    atv1 = Activation('relu')(res1)
    norm1 = BatchNormalization()(atv1)
    return norm1


def build_model(input_layer, initial, eeg_channels, filters=filters_size, kernel=kernel_size):

    x = fe_block(input_layer, filters, initial)
    x = AveragePooling2D(pool_size=(4, 1), strides=(3, 1))(x)

    x = fe_block_dep(x, filters, initial, dm = 1, kernel=kernel)
    x = AveragePooling2D(pool_size=(4, 1), strides=(3, 1))(x)

    x = fe_block_dep(x, filters, initial, dm=2, kernel=kernel)
    x = AveragePooling2D(pool_size=(4, 1), strides=(3, 1))(x)

    x = fe_block_dep(x, filters, initial, dm=2, kernel=kernel)

    x = Conv2D(filters=2, kernel_size=(2, 1), strides=(1, 1), activation="relu", padding='valid', kernel_initializer=initial)(x)
    x = (AveragePooling2D(pool_size=(k.int_shape(x)[-3], 1), strides=(1, 1)))(x)

    pool5 = MaxPooling2D(pool_size=(1, eeg_channels), strides=(1, 1))(x)

    pool5 = Activation("softmax")(pool5)

    output_layer = Flatten()(pool5)

    return output_layer


def res_net(eeg_channels,input_length):

    input_layer = Input((input_length, eeg_channels , 1))
    output_layer = build_model(input_layer, init, eeg_channels=eeg_channels)

    model = Model(input_layer, output_layer)
    opt = Adam(lr=0.001, decay=1e-6)  # This is not training only inference so these parameters need not be changed
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model._name = "ConvNet"
    return model
