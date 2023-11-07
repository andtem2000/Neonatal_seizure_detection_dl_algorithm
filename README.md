<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">  
  <br><br><strong>NeoNateNet</strong>
  <br><br><strong>Neonatal Seizure Detection Algorithm</strong>
  
---  
  ## Table of contents
1. [Introduction](#introduction)  
2. [Software requirements](#software-requirements)  
3. [Software build](#software-build)  
4. [File and Folder details](#File-descriptions)
5. [Instructions for Use](#InstructionsforUse)
6. [License](#License)
7. [Authors](#Authors)
8. [References](#References)
9. [Contact](#Contact)

---  
## 1. Introduction

This repository contains instructions for use and code for running a neonatal seizure detection deep learning algorithm using EEG signals as input.

<br /> It is based on the published paper [1] -link.
 
---  
   
## 2. Software/Hardware requirements
Python 3.8
<br />Tensorflow 2.3.0, Keras 2.4.3
<br /> GPU is not necessary.  
___  
## 3. Software build
Step 1: Get sources from GitHub 
```shell   
$ git clone https://github.com/CiallC/Neonatal_seizure_resnext.git
 
```  
___

## 4. File and Folder details
  

| Files                                      | Details                                          |    
|--------------------------------------------|--------------------------------------------------|        
| [Main_Inference.py](Main_Inference.py)     | The file for running seizure detection algorithm |
| [ConvNet.py](ConvNet.py)                   | Code for generating the deep learning model      |


| Folders                                  | Details                                                                                       |    
|------------------------------------------|-----------------------------------------------------------------------------------------------|        
| [Benchmark_weights](./Benchmark_weights) | Contains 3 model weights files; generated using 3 different seeds in training.                |
| [EEG files](./EEG_files)                 | Folder containing example EEG signal files from the publicly available Helskinki dataset [2]. |
| [Results](./Results)                     | Folder for results, i.e probability trace outputted for each EEG signal file inputted.        | 

___

## 6. Instructions for Use

The file to run the algorithm is [Main_Inference.py](Main_Inference.py).  
<br />  The probabilities of a seizure per second of inputted EEG signal are outputted by the algorithm in .npy format to the [Results](./Results) folder.
<br />  You can run this main file using the EEG files given with this repository which are from the Helsinki publicly available dataset [2]
and have been preprocessed as detailed below and as described in the paper  [1].
### EEG signal input file specifications
The input EEG files need to be in .mat format, a matrix of N by M, where N is the EEG signal data and M is the number of EEG channels in a bipolar montage.
<br /> The bipolar montage used, including order, in training and inference are given in [1] and [2], other bipolar configurations can be tested. 
<br /> EEG signal data should be at 32Hz sampling rate and during training was preprocessing by a DC notch filter and 0.5-12.8 bandwidth anti-aliasing filter.

### Adjustable parameters in [Main_Inference.py](Main_Inference.py)
These are the main parameters that can be adjusted by the user and are situated at the top of [Main_Inference.py](Main_Inference.py).  The default values, used in training and inference, are also given here.

| Parameter            | Description                                                                                                                                                                     |    
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| epoch_length         | Epoch/window length of the EEG input signal, in seconds.                                                                                                                        |
|                      | Default is 16                                                                                                                                                                   |
| epoch_shift          | Epoch/window shift of EEG input signal, in seconds.                                                                                                                             
|                      | Default is 1                                                                                                                                                                    |
| maf_window_parameter | Length in seconds of the moving average filter (maf) window parameter used in the maf.                                                                                          |
|                      | Default is 69                                                                                                                                                                   |
| file_list            | List of folder/file names of EEG signal files to be processed.                                                                                                                  |
|                      | e.g. ["./EEG_files/eeg1_SIGNAL.mat", "./EEG_files/eeg4_SIGNAL.mat"]                                                                                                             |
| weights_list         | List of folder/file names of model weight files; 3 different files exist from 3 different training seed-runs                                                                    |
|                      | ['./Benchmark_weights/best_weights_run0_hski_trained.hdf5',                                                                                                                     |
|                      | ,'./Benchmark_weights/best_weights_run1_hski_trained.hdf5','./Benchmark_weights/best_weights_run2_hski_trained.hdf5','./Benchmark_weights/best_weights_run2_hski_trained.hdf5'] | 
| results_path         | Folder to store the results, i.e. probabilities outputted per individual file                                                                                                   |
|                      | './Results/'                                                                                                                                                                    |

Further details can be found in the paper [1]
___

## 7. License
___
## 8. Authors
Aengus Daly, Gordon Lightbody, Andriy Temko
___
## 9. References
[1]  Main file link
[2]  Nathan Stevenson, Karoliina Tapani, Leena Lauronenand Sampsa Vanhatalo, “A dataset of neonatal EEG recordings with seizures annotations”. Zenodo, Jun. 05, 2018. doi: 10.5281/zenodo.2547147.
___
## 10. Contact

Aengus Daly 
<br /> Munster Technological University,
<br /> Cork City,
<br /> Ireland.

<br /> email aengus dot daly 'at' mtu.ie

___
