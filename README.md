# 
<h1 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">  
  <br><br><strong>Neonatal seizure detection resnext algorithm</strong>  
  
</h1>  
<h1 align="center">  
Aengus Daly et al.
</h1>  
  
---  
  ## Table of contents
1. [Introduction](#introduction)  
2. [Software requirements](#software-requirements)  
3. [Software build](#software-build)  
4. [Project files description](#project-files-description)  

---  
## 1. Introduction
  
This is the python implementation code for running a neonatal seizure detection algorithm using deep learning.
EEG signal and annotation files are given from the Helsinki publicly available dataset.
  
---  
  
  
## 2. Software requirements
Python 3.8
Tensorflow 2.3.0
  
  
## 3. Software build
Step 1: Get sources from GitHub 
```shell   
$ git clone https://github.com/
 
```  
  
## 4. Project files description
  
* Feature Engineering and EDA

| File                                                                                | Description |    
|-------------------------------------------------------------------------------------|---|        
| [psykose_feature_engineering.py](./code/psykose_feature_engineering.py)             | Reads the raw files of control and patients and generate the features |
| [psykose_eda.py](./code/psykose_eda.py) <br/> [psykose_machine_learning_plots.py](./code/psykose_machine_learning_plots.py) ||
  
* Machine Learning 

| File                                                                            | Description                                                                   |    
|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------|                 
| [psykose_machine_learning_models.py](./code/psykose_machine_learning_models.py) | Contains the ML models for 10-Fold Cross-Validation and LOO using One Day Out |
| [psykose_machine_learning_loo.py](./code/psykose_machine_learning_loo.py)       | ML models for LOPO - leave one person out                                     |
| [hyperparameter_tuning.py](./code/hyperparameter_tuning.py)                                 | Perform the models hyper parameters tuning                                    |

- Deep Learning  

| File | Description |
|---|---|
| [psyche_2_loo_time_cat_cnn2d_6_5_noval_6t.py](./code/psyche_2_loo_time_cat_cnn2d_6_5_noval_6t.py) | Contains the Deep Learning model |


- Utils 

| File | Description                                |
|---|--------------------------------------------|
|[my_metrics.py](./code/my_metrics.py) | Class with the metrics used in the project |
