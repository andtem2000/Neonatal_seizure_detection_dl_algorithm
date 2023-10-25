
import numpy as np
from keras.utils import np_utils
from keras.preprocessing.sequence import TimeseriesGenerator


def movingaverage(data, epoch_length):
    '''

    :param data: the vector which the MAF will be applied to
    :param window: the size of the moving average window
    :return: data after the MAF has been applied
    '''
    data = data
    window = 69 - epoch_length
    window = np.ones(int(window)) / float(window)
    return np.convolve(data, window, "same")


def mean_maf_probability(model, testX, path, runs, input_length, epoch_length):
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
    testY = np.ones((len(testX)))
    data_gen = TimeseriesGenerator(np.asarray(testX), np_utils.to_categorical(np.asarray(testY)),
                                   length=input_length, sampling_rate=1, stride=32, batch_size=300, shuffle=False)
    probs = []
    for loop in range(runs):

        if loop == 0:
            saved_weights_str = path + 'best_weights_run0_hski_trained.hdf5'
        if loop == 1:
            saved_weights_str = path + 'best_weights_run1_hski_trained.hdf5'
        if loop == 2:
            saved_weights_str = path + 'best_weights_run2_hski_trained.hdf5'

        model.load_weights(saved_weights_str)

        p = model.predict(data_gen)[:, 1]
        p = movingaverage(p, epoch_length) # Apply moving average filter

        probs.append(p)

    probs = np.asarray(probs)
    mean_probability = np.mean(probs, 0)

    return mean_probability
