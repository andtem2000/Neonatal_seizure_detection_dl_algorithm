"""
Main config file

"""

# import log_configuration
# LOGGER = log_configuration.logger
from Neonatal_Seizure_Resnext_algorithm.Inference_hski_resnxt_train_hski import inference

if __name__ == '__main__':
    # LOGGER.info("Script started...")

    # run_model = True
    # save_results = False
    # check_options(run_models, save_results)

    epoch_length = 16  # Lenght of input EEG signal in seconds
    input_sampling_rate = 32  # 32 Hz is the input signal sampling rate after preprocessing
    # input_length = epoch_length * input_sampling_rate  # here it is 512
    eeg_channels = 18  # 18 for Helsinki files
    path_1 = 'Benchmark_weights/'  # folder with weights
    path_2 = './Helsinki files/'  # folder with EEG signal data and labels for test
    name = 'run_hski_1' # name for saving results file
    hski_baby = 4 # the Helsinki file/baby no. for test
    runs = 2  # no. of sets of weights used.  This corresponds to the no. of training runs.
    window_size = 69 - epoch_length  # 53 for 16 sec window, used in Moving Average Filter
    # Cannot change the following parameters in test
    filters = 32
    kernel = 5
    models = inference(hski_baby, path_2,path_1,kernel,filters,runs=runs,name=name,eeg_channels=eeg_channels,input_sampling_rate=input_sampling_rate,epoch_length=epoch_length)
    # models = Inference_run()