<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">  
  <br><br><strong>EegNeoNet or NeoNateNet</strong>
  <br><br><strong>Neonatal Seizure Detection Algorithm</strong>
  
---  
  ## Table of contents
1. [Introduction](#1-introduction) 
2. [Instructions for Use](#2-Instructions-for-Use)
3. [File and Folder details](#3-File-and-Folder-details)
4. [Software/Hardware requirements](#4-SoftwareHardware-Requirements)  
5. [Software build](#5-software-build)
6. [License](#6-License)
7. [Authors](#7-Authors)
8. [References](#8-References)
9. [Contact](#9-Contact)
10. [Citation](#10-Citation)

---  
## 1. Introduction

This repository contains code and instructions for running a neonatal seizure detection deep learning algorithm using EEG signals as input and a probability of seizure trace as output.

<br /> It is based on the published papers [1] and [[2]](https://ieeexplore.ieee.org/document/10290015), and references are [below](#8-References).

___

## 2. Instructions for Use

The file to run the algorithm is [Main_Inference.py](Main_Inference.py).  
<br />  The probabilities of a seizure per second are output, per second of inputted EEG signal, in .npy format to the [Results](./Results) folder.
<br />  You can run this main file using as input the EEG files given with this repository which are from the Helsinki publicly available dataset [3]
and have been preprocessed as detailed below and as described in the paper  [1].
### EEG signal input file specifications
The input EEG files need to be in .mat format, a matrix of N by M, where N is the EEG signal data and M is the number of EEG channels in a bipolar montage.
The bipolar montage used, including order, in training and inference are given in [1] and [2], other bipolar configurations can be tested. 
<br /> EEG signal data should be at 32Hz sampling rate and during training was preprocessing by a DC notch filter and 0.5-12.8 bandwidth anti-aliasing filter.
### 2 Model variations
There are two different model variations (weights) given.  One that was trained on the publicly available Helsinki dataset [3] and a second that was trained on 
the Helsinki dataset plus the publicly available unannotated dataset [4] which was pseudo labelled by the authors using a novel technique fully described in [2].
For each of these training variations 3 runs were completed using three different random seeds.  Further details are given in [1].

### Adjustable parameters in [Main_Inference.py](Main_Inference.py)
These are the parameters that can be adjusted by the user and are situated at the top of [Main_Inference.py](Main_Inference.py) file.  The default values, used in training and inference, are also given here.

| Parameter            | Description                                                                                                                                                                                                                                                                                                                                                                                     |    
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| epoch_length         | Epoch/window length of the EEG input signal, in seconds.                                                                                                                                                                                                                                                                                                                                        |
|                      | Default is 16                                                                                                                                                                                                                                                                                                                                                                                   |
| epoch_shift          | Epoch/window shift of EEG input signal, in seconds.                                                                                                                                                                                                                                                                                                                                             
|                      | Default is 1                                                                                                                                                                                                                                                                                                                                                                                    |
| maf_window_parameter | Length in seconds of the moving average filter (maf) window parameter.                                                                                                                                                                                                                                                                                                          |
|                      | Default is 69                                                                                                                                                                                                                                                                                                                                                                                   |
| file_list            | List of folder/file names of EEG signal files to be processed.                                                                                                                                                                                                                                                                                                                                  |
|                      | e.g. ["./EEG_files/eeg1_SIGNAL.mat", "./EEG_files/eeg4_SIGNAL.mat"]                                                                                                                                                                                                                                                                                                                             |
| weights_list         | List of folder/file names of model weight files; 2 sets of model weights are given<br/>,one trained on Helsinki data only( ..best_weights_run2_hski_trained) and a second trained on the Helsinki data plus pseudo label data from another publicly available dataset(...hski_plus_pslabel_HIEInfant);<br/> for each set of weights 3 different files exist from 3 different training seed runs |                                                                                                                    |
|                      | ['./Benchmark_weights/best_weights_run1_hski_trained.hdf5','./Benchmark_weights/best_weights_run2_hski_trained.hdf5','./Benchmark_weights/best_weights_run2_hski_trained.hdf5']                                                                                                                                                                                                                 | 
| results_path         | Folder to store the results, i.e. probabilities outputted per individual file                                                                                                                                                                                                                                                                                                                   |
|                      | './Results/'                                                                                                                                                                                                                                                                                                                                                                                    |

Further details can be found in the paper [1]

---  

## 3. File and Folder details
  

| Files                                      | Details                                              |    
|--------------------------------------------|------------------------------------------------------|        
| [Main_Inference.py](Main_Inference.py)     | The file for running the seizure detection algorithm |
| [ConvNet.py](ConvNet.py)                   | Code for generating the deep learning model          |


| Folders                                  | Details                                                                                       |    
|------------------------------------------|-----------------------------------------------------------------------------------------------|        
| [Benchmark_weights](./Benchmark_weights) | Contains the model weights files; generated using 3 different seeds in training.              |
| [EEG files](./EEG_files)                 | Folder containing example EEG signal files from the publicly available Helskinki dataset [2]. |
| [Results](./Results)                     | Folder for results, i.e probability trace outputted for each EEG signal file inputted.        | 

___
   
## 4. Software/Hardware requirements
Python 3.9.12, Tensorflow 2.10.0, Keras 2.10.0
<br /> It will work with older versions also.
<br /> GPU use will reduce run time.  
___  
## 5. Software build
Step 1: Get sources from GitHub 
```shell   
$ git clone https://github.com/CiallC/Neonatal_seizure_resnext.git
 
```  
___

## 6. License

The model code and weights are made available for use under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). You may obtain a copy of the License at: https://creativecommons.org/licenses/by-nc-sa/4.0/.
___
## 7. Authors
Aengus Daly, Gordon Lightbody, Andriy Temko
___
## 8. References

[1] Main link: <br /> 
[2] A. Daly, G. Lightbody and A. Temko, "Bridging the Source-Target Mismatch with Pseudo Labeling for Neonatal Seizure Detection," 2023 31st European Signal Processing Conference (EUSIPCO), Helsinki, Finland, 2023, pp. 1100-1104, doi: 10.23919/EUSIPCO58844.2023.10290015. [paper](https://ieeexplore.ieee.org/document/10290015) <br /> 
[3] N. J. Stevenson, K. Tapani, L. Lauronen and S. Vanhatalo, "A dataset of neonatal EEG recordings with seizure annotations", Scientific Data, vol. 6, no. 1, Mar. 2019, Available: https://zenodo.org/records/4940267 <br /> 
[4] J. M. O'Toole, "Neonatal EEG graded for severity of background abnormalities in hypoxic-ischaemic encephalopathy", arXiv, Jun. 2022, [online] Available: http://arxiv.org/abs/2206.04420
___
## 9. Contact

Aengus Daly 
<br /> Munster Technological University,
<br /> Cork City,
<br /> Ireland.

<br /> email aengus dot daly 'at' mtu.ie

## 10. Citation

If you use this work, consider citing our (update to new paper) [paper](https://ieeexplore.ieee.org/document/10290015):

```latex
@INPROCEEDINGS{10290015,

  author={Daly, Aengus and Lightbody, Gordon and Temko, Andriy},
  booktitle={2023 31st European Signal Processing Conference (EUSIPCO)}, 
  title={Bridging the Source-Target Mismatch with Pseudo Labeling for Neonatal Seizure Detection}, 
  year={2023},
  volume={},
  number={},
  pages={1100-1104},
  doi={10.23919/EUSIPCO58844.2023.10290015}}

```

___
