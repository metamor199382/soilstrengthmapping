# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 11:23:12 2023

@author: ruizhen_wang
"""
import os
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from xgboost.sklearn import XGBRegressor
from lce import LCERegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold




os.chdir('C:/Users/ruizhen_wang/Desktop/一改')
path = 'C:/Users/ruizhen_wang/Desktop/一改/SM_dataset_all_depth_2023216_FULLY.xlsx'
dataset = pd.read_excel(path)

depth = '_0_5cm'
soil_depth = '05cm'


model1 = ['NTR', 'NDVI', 'Slope', 'DEM', 'aspect', 'SM_'+soil_depth]

model2 = ['VSDI', 'NSDSI', 'Slope', 'DEM', 'aspect','SM_'+soil_depth]

model3 = ['NTR', 'NDVI', 'VSDI', 'NSDSI', 'Slope', 'DEM', 'aspect','SM_'+soil_depth]

model4 = ['NTR', 'NDVI', 'Slope', 'DEM', 'aspect', 'clay'+depth, 'sand'+depth, 'silt'+depth, 
          'OC'+depth, 'bulkdensity'+depth, 'porosity'+depth,'SM_'+soil_depth]

model5 = ['VSDI', 'NSDSI', 'Slope', 'DEM', 'aspect','clay'+depth, 'sand'+depth, 'silt'+depth, 
          'OC'+depth, 'bulkdensity'+depth, 'porosity'+depth,'SM_'+soil_depth]

model6 = ['NTR', 'NDVI', 'VSDI', 'NSDSI', 'Slope', 'DEM', 'aspect','clay'+depth, 'sand'+depth,
          'silt'+depth, 'OC'+depth, 'bulkdensity'+depth, 'porosity'+depth,'SM_'+soil_depth]

model7 = ['NTR', 'NDVI', 'Slope', 'DEM', 'aspect', 'clay'+depth, 'sand'+depth, 'silt'+depth, 
          'OC'+depth, 'bulkdensity'+depth, 'porosity'+depth, 'residual water content'+depth, 
          'saturated water content'+depth, 'saturated hydraulic conductivity'+depth,'SM_'+soil_depth]

model8 = ['VSDI', 'NSDSI', 'Slope', 'DEM', 'aspect','clay'+depth, 'sand'+depth, 'silt'+depth, 
          'OC'+depth, 'bulkdensity'+depth, 'porosity'+depth, 'residual water content'+depth, 
          'saturated water content'+depth, 'saturated hydraulic conductivity'+depth,'SM_'+soil_depth]

model9 = ['NTR', 'NDVI', 'VSDI', 'NSDSI', 'Slope', 'DEM', 'aspect','clay'+depth, 'sand'+depth,
          'silt'+depth, 'OC'+depth, 'bulkdensity'+depth, 'porosity'+depth, 'residual water content'+depth, 
          'saturated water content'+depth, 'saturated hydraulic conductivity'+depth,'SM_'+soil_depth]




dataset['SM_'+soil_depth] = dataset['SM_'+soil_depth]*1.0614+0.0134
dataset = dataset[model9]
dataset = dataset.values
X_train = dataset[:,:-1].astype(float)
y_train = dataset[:,-1].astype(float)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

kfold = KFold(n_splits=10, shuffle=True, random_state=42)


# best_score = 0
# i =1
# for n_estimators in [100,200]:
#     for min_samples_leaf in [2,4,6,8]:
#         for min_samples_split in [2,5,8]:
#             for learning_rate in [0.03,0.05,0.08,0.10,0.12,0.14]:
#                 for max_depth in [2,4,6,10]:
#                     print(i)
#                     i = i+1
#                     random_forest_model = GradientBoostingRegressor(n_estimators = n_estimators,  max_depth =max_depth, learning_rate = learning_rate,
#                                                                         min_samples_leaf =min_samples_leaf, min_samples_split =min_samples_split, random_state=42)
#                     scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'r2')
#                     score = scores.mean()
#                     if score > best_score:
#                         best_score = score
#                         best_parameters = {"n_estimators":n_estimators,"min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split, 
#                                                 "learning_rate":learning_rate, "max_depth":max_depth}
# random_forest_model = GradientBoostingRegressor(**best_parameters)                           
# scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'r2')
# print("Best parameters:{}".format(best_parameters))
# print('r2: ' + str(scores.mean()))
# scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'neg_mean_squared_error')
# RMSE = abs(scores)**0.5
# print('RMSE: ' + str(RMSE.mean()))





# best_score = 0
# i =1
# for n_estimators in [100,200]:
#     for subsample in [1]:
#         for colsample_bytree in [0.8,1]:
#             for min_child_weight in [1,3,5,7]:
#                 for learning_rate in [0.03,0.05,0.08,0.10,0.12,0.14,0.16]:
#                     for max_depth in [2,3,5,6,10]:
#                         print(i)
#                         i = i+1
#                         random_forest_model =XGBRegressor(n_estimators=n_estimators, max_depth = max_depth, random_state = 42, 
#                                                           learning_rate = learning_rate, subsample =subsample,
#                                                           colsample_bytree = colsample_bytree, min_child_weight = min_child_weight)
#                         scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'r2')
#                         score = scores.mean()
#                         if score > best_score:
#                             best_score = score
#                             best_parameters = {"n_estimators":n_estimators,"subsample":subsample, "colsample_bytree":colsample_bytree, "min_child_weight":min_child_weight, 
#                                                "learning_rate":learning_rate, "max_depth":max_depth}
# random_forest_model = XGBRegressor(**best_parameters)                           
# scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'r2')
# print("Best parameters:{}".format(best_parameters))
# print('r2: ' + str(scores.mean()))
# scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'neg_mean_squared_error')
# RMSE = abs(scores)**0.5
# print('RMSE: ' + str(RMSE.mean()))


# best_score = 0
# i =0
# for max_depth in [2,3]:
#     for n_estimators in [100,200]:
#         i = i+1
#         print(i)
#         random_forest_model = LCERegressor(n_jobs = -1, random_state=42, max_depth =max_depth, n_estimators = n_estimators)
#         scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'r2')
#         score = scores.mean()
#         if score > best_score:
#             best_score = score
#             best_parameters = {"n_estimators":n_estimators,"max_depth":max_depth}
# random_forest_model = LCERegressor(**best_parameters)                           
# scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'r2')




best_score = 0
i =1
for n_estimators in [100,200]:
    for min_samples_split in [2,4,5,7,10]:
        for min_samples_leaf in [1,2,3,4]:
            print(i)
            i = i+1
            random_forest_model=RandomForestRegressor(n_estimators=n_estimators,  min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf, n_jobs = -1, random_state=42,
                                                    bootstrap=True, oob_score=False)
            scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'r2')
            score = scores.mean()
            if score > best_score:
                best_score = score
                best_parameters = {"n_estimators":n_estimators, "min_samples_split":min_samples_split, "min_samples_leaf":min_samples_leaf}
random_forest_model = RandomForestRegressor(**best_parameters)                           
scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'r2')
print("Best parameters:{}".format(best_parameters))
print('r2: ' + str(scores.mean()))
scores = cross_val_score(random_forest_model, X_train, y_train, cv=kfold, scoring = 'neg_mean_squared_error')
RMSE = abs(scores)**0.5
print('RMSE: ' + str(RMSE.mean()))

#保存模型
joblib.dump(random_forest_model,'E:/SDR-SMN_extraction/SDR_RF_model_r2=0.54.pkl')

