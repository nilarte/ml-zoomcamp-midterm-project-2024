# ml-zoomcamp-midterm-project-2024
## Description
This is DataTalksClub ML zoomcamp midterm project repo. 

It uses dataset with person details like cholestrol level, BP, ECG status etc to predict heartdisease.

Dataset credit: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Here are step by step details about how we build optimal model to predict heartdisease probability. 

## 1. Data preparation, cleanup and EDA
[notebook.ipynb](./notebook.ipynb) 
Parse downloaded dataset [employee_data.csv](./employee_data.csv) via `pandas`.

Note: We are using local dataset copy here but we can download data from kaggle in notebook as well.
```python
kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
```
Look for NAN values in data (There are none). 

Feature importance of rest features with our target variable: heartdisease:
We find mutual info for categorical features and correlation for numerical features.

## 2. Training a model
### 2.1 One-hot encoding
Turn categorical data into binary vector
### 2.2 Simple Logistic regression
Train a simple logistic regression model.
Check AUC score for validation data.
### 2.3 



