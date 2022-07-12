<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">  
  <br><br><strong>Neonatal Seizure Detection Resnext Algorithm</strong>  
  
---  
  ## Table of contents
1. [Introduction](#introduction)  
2. [Software requirements](#software-requirements)  
3. [Software build](#software-build)  
4. [File description](#File-descriptions)
5. [EEG files](#EEG-files)
6. [License](#License)
7. [Authors](#Authors)
8. [References](#References)
9. [Contact](#Contact)

---  
## 1. Introduction
  
This is the python implementation code for running a neonatal seizure detection algorithm using deep learning.
EEG signal and annotation files are given from the Helsinki publicly available dataset.
  
---  
  
  
## 2. Software/Hardware requirements
Python 3.8
<br />Tensorflow 2.3.0
<br /> GPU is not required but code should run faster with one.  
  
## 3. Software build
Step 1: Get sources from GitHub 
```shell   
$ git clone https://github.com/CiallC/Neonatal_seizure_resnext.git
 
```  
  
## 4. File descriptions
  

| File                                                                                | Description |    
|-------------------------------------------------------------------------------------|---|        
| [Inference_hski_resnxt_train_hski_50e.py](./Neonatal_Seizure_Resnext_algorithm/Inference_hski_resnxt_train_hski_50e.py)             | Main file for seizure detection|
| [GetData_perfile_512_1_hski.py](./Neonatal_Seizure_Resnext_algorithm/GetData_perfile_512_1_hski.py) |File for importing and loading EEG and annotation data|
| [score_tool_DNN_resp_v2.py](./Neonatal_Seizure_Resnext_algorithm/score_tool_DNN_resp_v2.py) |File for calculating AUC, AUC90 with post processing|

| Folder                                                                                | Description |    
|-------------------------------------------------------------------------------------|---|        
| [Benchmark_weights](./Benchmark_weights)             | Contains 3 sets of weights for the resnext model|
| [Helsinki files](./Helsinki_files) |Contains EEG signal files and annotations|



## 5. EEG files

The database of neonatal EEG used to develop the algorithms is available at DOI: 10.5281/zenodo.2547147 or https://zenodo.org/record/2547147 [1]


## 6. License

## 7. Authors
Aengus Daly, Gordon Lightbody, Andriy Temko

## 8. References
[1]  Nathan Stevenson, Karoliina Tapani, Leena Lauronenand Sampsa Vanhatalo, “A dataset of neonatal EEG recordings with seizures annotations”. Zenodo, Jun. 05, 2018. doi: 10.5281/zenodo.2547147.

## 9. Contact

Aengus Daly 
<br /> Munster Technological University,
<br /> Cork City,
<br /> Ireland.

<br /> email aengus dot daly 'at' mtu.ie
