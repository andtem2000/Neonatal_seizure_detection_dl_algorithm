<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">  
  <br><br><strong>NeoNateNet</strong>
  <br><br><strong>Neonatal Seizure Detection Algorithm</strong>
  
---  
  ## Table of contents
1. [Introduction](#introduction)  
2. [Software requirements](#software-requirements)  
3. [Software build](#software-build)  
4. [File description](#File-descriptions)
5. [EEG files](#EEG-files)
6. [Instructions for Use](#InstructionsforUse)
8. [License](#License)
9. [Authors](#Authors)
10. [References](#References)
11. [Contact](#Contact)

---  
## 1. Introduction
  
This repository contains code for a deep learning neonatal seizure detection algorithm using EEG signals as input.  It is based on the published paper -link.
Details of how the algorithm can be run are given below using the publicly available Helsinki EEG dataset.
  
---  
  
  
## 2. Software/Hardware requirements
Python 3.8
<br />Tensorflow 2.3.0
<br /> GPU is not necessary.  
  
## 3. Software build
Step 1: Get sources from GitHub 
```shell   
$ git clone https://github.com/CiallC/Neonatal_seizure_resnext.git
 
```  
  
## 4. File descriptions
  

| Files                                  | Description                                       |    
|----------------------------------------|---------------------------------------------------|        
| [Main_Inference.py](Main_Inference.py) | Main file for running seizure detection algorithm |
| [config_1.py](Config_1.py)             | Configuration file                                |

| Folder                                   | Description                                       |    
|------------------------------------------|---------------------------------------------------|        
| [Benchmark_weights](./Benchmark_weights) | Contains 3 sets of weights for the model          |
| [EEG files](./EEG_files)                 | EEG signal files and annotations                  |
| [Utils](./Utils)                         | Model code, inference calcs, post processing etc. |



## 5. EEG files

The database of neonatal EEG used to develop the algorithm is available at DOI: 10.5281/zenodo.2547147 or https://zenodo.org/record/2547147 [1]

## 6. Instructions for Use

## 7. License

## 8. Authors
Aengus Daly, Gordon Lightbody, Andriy Temko

## 9. References
[1]  Nathan Stevenson, Karoliina Tapani, Leena Lauronenand Sampsa Vanhatalo, “A dataset of neonatal EEG recordings with seizures annotations”. Zenodo, Jun. 05, 2018. doi: 10.5281/zenodo.2547147.

## 10. Contact

Aengus Daly 
<br /> Munster Technological University,
<br /> Cork City,
<br /> Ireland.

<br /> email aengus dot daly 'at' mtu.ie