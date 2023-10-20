"""
Main config file for Inference_hski_resnxt_train_hski_cf2

"""

file_list = ["eeg1_SIGNAL.mat", "eeg4_SIGNAL.mat"]
epoch_length = 16  # Length of input EEG signal in seconds
epoch_shift = 1
input_sampling_rate = 32  # 32 Hz is the input signal sampling rate after preprocessing
runs = 3  # no. of sets of weights used.  This corresponds to the no. of training runs.
window_size = 69 - epoch_length  # 53 for 16 sec window, used in Moving Average Filter
# input_length = epoch_length * input_sampling_rate  # here it is 512
model_weights_files = []
# runs = 3  # no. of sets of weights used.  This corresponds to the no. of training runs.
window_size = 69 - epoch_length  # 53 for 16 sec window, used in Moving Average Filter

path_weights = './Benchmark_weights/'  # folder with weights
path_test_files = 'EEG_files/'  # folder with EEG signal data and labels for test
path_results = './Results/'

