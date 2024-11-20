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
# patient = {
#     "age": 49,
#     "sex": "F",
#     "chestpaintype": "NAP",
#     "restingbp": 160,
#     "cholesterol": 180,
#     "fastingbs": 0,
#     "restingecg": "Normal",
#     "maxhr": 156,
#     "exerciseangina": "N",
#     "oldpeak": 1.0,
#     "st_slope": "Flat",
# }


response = requests.post(url, json=patient).json()
print(response)

if response['heartdisease'] == True:
    print('High probability of heartdisease for patient %s' % patient_id)
else:
    print('No high probability of heartdisease for patient %s' % patient_id)
