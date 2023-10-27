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
<br />Tensorflow 2.3.0
<br /> GPU is not necessary.  
___  
## 3. Software build
Step 1: Get sources from GitHub 
```shell   
$ git clone https://github.com/CiallC/Neonatal_seizure_resnext.git
 
```  
___

## 4. File and Folder details
  

| Files                                      | Details                                                                                  |    
|--------------------------------------------|------------------------------------------------------------------------------------------|        
| [Main_Inference.py](Main_Inference.py)     | Main file for running seizure detection algorithm                                        |
| [ConvNet.py](ConvNet.py)                   | Code for generating the keras model file, both are stored in the folder [Utils](./Utils) |
| [ConvNet_model.keras](ConvNet_model.keras) | Keras model file                                                                         |

| Folder                                   | Details                                                                                        |    
|------------------------------------------|------------------------------------------------------------------------------------------------|        
| [Benchmark_weights](./Benchmark_weights) | Contains 3 model weights files; generated using 3 different seeds in training.                 |
| [EEG files](./EEG_files)                 | EEG signal files; example files are included from the publicly available Helskinki dataset [2] |
| [Results](./Results)                     | Folder for results output                                                                      |
| [Utils](./Utils)                         | Contains the ConvNet.py file and the keras model file                                          

___

## 6. Instructions for Use

The main file to run the algorithm is [Main_Inference.py](Main_Inference.py).  
<br />  The probabilities of a seizure per second of inputted EEG signal are outputted by the algorithm in .npy format to the [Results](./Results) folder.
<br />  You can run this main file using the EEG files given with this repository which are from the Helsinki publicly available dataset [2]
and are preprocessed as detailed below and described in the paper [1].
### EEG signal input file specifications
The input EEG files need to be in .mat format, a matrix of N by M, where N is the EEG signal data and M is the number of EEG channels in a bipolar montage.
<br /> The bipolar montage used, including order, in training and inference are given in [1] and [2], other bipolar configurations can be tested. 
<br /> EEG signal data, used in training and inference, was 32Hz sampling rate following preprocessing by a DC notch filter and 0.5-12.8 bandwidth anti-aliasing filter.

### Adjustable parameters in [Main_Inference.py](Main_Inference.py)
These are the main parameters that can be adjusted by the user and are situated at the top of [Main_Inference.py](Main_Inference.py).  The default values, used in training and inference, are also given here.

| Parameter           | Description                                                                                              |    
|---------------------|----------------------------------------------------------------------------------------------------------|        
| file_list           | List of EEG signal files names to be processed; these files should be located in [EEG files](./EEG_files).   
|                     | e.g. ["eeg1_SIGNAL.mat", "eeg4_SIGNAL.mat"]                                                              |
| epoch_length        | Epoch/window length of the EEG input signal, in seconds.                                                 |
|                     | Default is 16                                                                                            |
| epoch_shift         | Epoch/window shift of EEG input signal, in seconds.                                                      
|                     | Default is 1                                                                                             |
| input_sampling_rate | EEG input signal sampling rate in Hz.                                                                    |
|                     | Default is 32                                                                                            |
| runs                | No. of model weights files used; weights were generated via 3 different random initialization training runs. 
|                     | Default is 3                                                                                             
| maf_window_size     | Used in the moving average filter applied to the probabilities before output.                            |
|                     | Default is  69 - epoch_length, i.e. 53 for 16 sec window                                                 |

Further details can be found in the paper [1]
___

## 7. License
___
## 8. Authors
Aengus Daly, Gordon Lightbody, Andriy Temko
___
## 9. References
[2]  Nathan Stevenson, Karoliina Tapani, Leena Lauronenand Sampsa Vanhatalo, “A dataset of neonatal EEG recordings with seizures annotations”. Zenodo, Jun. 05, 2018. doi: 10.5281/zenodo.2547147.
___
## 10. Contact

Aengus Daly 
<br /> Munster Technological University,
<br /> Cork City,
<br /> Ireland.

<br /> email aengus dot daly 'at' mtu.ie

___
