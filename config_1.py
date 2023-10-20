"""
Main config file for Inference_hski_resnxt_train_hski_cf2

"""

epoch_length = 16  # Lenght of input EEG signal in seconds
input_sampling_rate = 32  # 32 Hz is the input signal sampling rate after preprocessing
# input_length = epoch_length * input_sampling_rate  # here it is 512
eeg_channels = 18  # 18 for EEG_files
hski_baby = 4 # the Helsinki file/baby no. for test
hski_baby_num = 1 # the number of consecutive EEG_files to be used in test
runs = 3  # no. of sets of weights used.  This corresponds to the no. of training runs.
window_size = 69 - epoch_length  # 53 for 16 sec window, used in Moving Average Filter
filters = 32 # Cannot change the following parameters in test
kernel = 5

path_weights = './Benchmark_weights/'  # folder with weights
path_test_files = 'EEG_files/'  # folder with EEG signal data and labels for test
path_results = './Results/'

results_name = 'run_hski_1' # name for saving results file
