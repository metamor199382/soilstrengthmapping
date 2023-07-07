# -*- coding: utf-8 -*-
"""
Created on Fri Nov 4 13:43:09 2022

@author: lenovo
"""
import os
import numpy as np
import pandas as pd
import time
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lce import LCEClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import shap

#data reading
os.chdir('E:/HWSD/classfied/')
dataset = pd.read_excel('texture__rough_gSSURGO_data_20230213.xlsx')
feat_labels = dataset.columns[1:8]
#data labeling
dataset = dataset.values
X = dataset[:,1:].astype(float) 
Y = dataset[:,0]

features = ['Sand','Silt',"Clay","OC","BD","CEC"]

#sampling_strategy=strategy
#onehotencoder
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, encoded_Y, test_size = 0.3, shuffle=True, random_state=42)

target_labels = ['CH','CL','CL-ML','MH','ML','SC','SC-SM','SM','SP', 'SP-SM']

# classifier = GradientBoostingClassifier(n_estimators = 160, random_state=42,
#                                         learning_rate= 0.3, max_depth = 30)

# classifier = XGBClassifier(objective='multi:softmax', num_class=10, n_estimators = 160, random_state=42, 
#                             eta = 0.3, max_depth = 18, min_child_weight = 1, gamma = 0, subsample = 1, 
#                             colsample_bytree=0.9, alpha = 0, reg_lambda=1)

#classifier = RandomForestClassifier(n_estimators = 160, random_state=42, max_depth=25)

classifier = LCEClassifier(n_estimators = 160, n_jobs = -1, random_state=42)

result = classifier.fit(X_train, y_train.ravel()).predict(X_test)


# explainer = shap.TreeExplainer(classifier,feature_names=features)
# shap_values = explainer.shap_values(X_test)
# #shap.force_plot(explainer.expected_value[3], shap_values[3][0,:], X_test[0,:], matplotlib=True)
# shap.dependence_plot(2, np.array(shap_values)[1], X_test, feature_names=features, interaction_index= 'CEC')


# shap.summary_plot(shap_values, X_test,feature_names= features, class_names = target_labels, plot_size=(12,8),show=False)
# fig = plt.gcf()
# plt.xticks(fontsize=20, fontname='Times New Roman')
# plt.yticks(fontsize=20, fontname='Times New Roman')
# plt.legend(fontsize=20)
# plt.xlabel('mean(|SHAP value|)(average impact on model output magnitude)', fontsize=20, fontname='Times New Roman')
# ax = plt.gca()
# ax.tick_params(axis='both', direction='in', width=1, right=True, top=True)
# ax.spines['bottom'].set_visible(True)
# ax.spines['top'].set_visible(True)
# ax.spines['left'].set_visible(True)
# ax.spines['right'].set_visible(True)
# plt.xlim(0, 25)
# plt.show()
# #fig.savefig('C:/Users/ruizhen_wang/Desktop/二改/shap_output_test.png', dpi=600)


#confusion_matrix
matrix = confusion_matrix(y_test, result)
norm_matrix = matrix.astype('float')/matrix.sum(axis = 1)[:, np.newaxis]
plt.figure(figsize=(12, 8), dpi=600)

ind_array = np.arange(len(target_labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    value = norm_matrix[y_val][x_val]
    plt.text(x_val, y_val, "%0.2f" % (value), ha = 'center', va = 'center', fontsize = 15, family='Times New Roman')



plt.imshow(norm_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix',fontsize = 20,family='Times New Roman',fontweight='bold')
cb = plt.colorbar() 
for l in cb.ax.yaxis.get_ticklabels(): 
    l.set_family('Times New Roman')
#cb.ax.tick_params(labelsize=15) 
xlocations = np.array(range(len(target_labels)))
plt.xticks(xlocations, target_labels, fontsize = 15, family='Times New Roman')
plt.yticks(xlocations, target_labels, fontsize = 15, family='Times New Roman')
plt.xlabel('Predicted Label', fontsize = 18,family='Times New Roman')
plt.ylabel('True Label', fontsize = 18,family='Times New Roman')
np.set_printoptions(precision=2)

tick_marks = np.array(range(len(target_labels))) + 0.5
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plt.show()

#accuracy
print("训练集：",classifier.score(X_train,y_train))
print("测试集：",classifier.score(X_test,y_test))

#precision, recall
print(classification_report(y_test, result, target_names = target_labels, digits = 4))
report = classification_report(y_test, result, target_names = target_labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()


#Kappa score, MCC
print(matthews_corrcoef(y_test, result))
print(cohen_kappa_score(y_test, result))

#feature weight
importance = classifier.feature_importances_
indices = np.argsort(importance)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importance[indices[f]]))


# pre_dataset = pd.read_excel('C:/Users/ruizhen_wang/Desktop/一改/soilgrid/15-30 cm.xlsx')
# column = ['T_SAND', 'T_SILT', 'T_CLAY', 'OC', 'T_BULK_DEN', 'T_CEC_SOIL']
# data_cn = pre_dataset[column].values
# prediction = classifier.predict(data_cn)
# prediction1 = encoder.inverse_transform(prediction)
# pred_USCS_code = pd.Series(prediction)
# pred_USCS_texture = pd.Series(prediction1)
# pre_dataset['pred_USCS_texture'] = pred_USCS_texture
# pre_dataset['pred_USCS_code'] = pred_USCS_code

# # time = time.strftime('%Y%m%d',time.localtime(time.time()))
# # filename = 'Soilgrid_Pred_USCS' + str(time) + '.xlsx'
# pre_dataset.to_excel('Soilgrid_Pred_USCS_2023_15-30.xlsx', index=False)