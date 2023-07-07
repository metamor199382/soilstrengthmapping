# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 16:53:49 2023

@author: ruizhen_wang
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt



os.chdir('E:/HWSD/classfied/')
dataset = pd.read_excel('texture__rough_gSSURGO_data_20230213.xlsx')
feat_labels = dataset.columns[1:11]

dataset['LL'] = 0.773*dataset['claytotal_']+0.373*dataset['cec7_r']+10.921
dataset['PI'] = 0.558*dataset['claytotal_']+0.149*dataset['cec7_r']-1.884
output = []


def USCS_Classification(input_series): 
    if input_series['sandtotal_'] > 50:
        if input_series['silttotal_'] + input_series['claytotal_'] < 5:
            return 'SP'
        elif input_series['silttotal_'] + input_series['claytotal_']  >=5 and input_series['silttotal_'] + input_series['claytotal_'] <=12:
            return 'SP-SM'            
        elif input_series['silttotal_'] + input_series['claytotal_'] > 12:
            if input_series['PI'] < 4 or input_series['PI'] < 0.73*(input_series['LL']-20):
                return 'SM'
            elif input_series['PI'] > 7 and input_series['PI'] > 0.73*(input_series['LL']-20):
                return 'SC'
            else:
                return 'SC-SM'
    if input_series['sandtotal_'] <= 50: 
        if input_series['LL'] < 50:
            if input_series['PI'] < 4 or input_series['PI'] < 0.73*(input_series['LL']-20):
                return 'ML'
            elif input_series['PI'] > 7 and input_series['PI'] > 0.73*(input_series['LL']-20):
                return 'CL'
            else:
                return 'CL-ML'
        if input_series['LL'] >= 50:
            if input_series['PI'] < 0.73*(input_series['LL']-20):
                return 'MH'
            elif input_series['PI'] >= 0.73*(input_series['LL']-20):
                return 'CH'    

for index, series in dataset.iterrows():
    result = USCS_Classification(series)
    output.append(result)

dataset['critera'] = pd.Series(output)
dataset = dataset.values
X = dataset[:,-1]
Y = dataset[:,0] 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, shuffle=True, random_state=42)
target_labels = ['CH','CL','CL-ML','MH','ML','SC','SC-SM','SM','SP', 'SP-SM']
print(classification_report(Y, X, target_names = target_labels, digits = 4))
print(cohen_kappa_score(y_test, X_test))