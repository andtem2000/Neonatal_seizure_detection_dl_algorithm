import numpy as np
import scipy.io as sio


def getdatagen(Baby,path_2 = '../Helsinki files/'):

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
