#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

patient_id = 'xyz-123'
patient = {
    "age": 40,
    "sex": "M",
    "chestpaintype": "ATA",
    "restingbp": 140,
    "cholesterol": 289,
    "fastingbs": 0,
    "restingecg": "Normal",
    "maxhr": 172,
    "exerciseangina": "N",
    "oldpeak": 0.0,
    "st_slope": "Up",
}


response = requests.post(url, json=patient).json()
print(response)

if response['heartdisease'] == True:
    print('High probability of heartdisk  %s' % patient_id)
else:
    print('No high probability of heartdisk  %s' % patient_id)