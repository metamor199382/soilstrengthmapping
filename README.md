# This is the code used for study 'A Novel Finer Soil Strength Mapping Framework Based on Machine Learning and Remote Sensing Images'
Data extracted_gSSURGO_data.xlsx include the basic properties obtained from GSSURGO database to predict USCS soil classification, including sand, clay, silt, CEC, organic carbon, bulk density
Data extracted_gSSURGO_data_with LL and PI.xlsx include measured liquid limit and plastic index for comparing with the criterion-based classification with estimated LL and PI by linear model
Data SM_dataset_all_depth_FULLY.xlsx include the pixel information from Sentinel-2, SoilGrids, DEM model and estimated hydrodynamic properties for the soil mositure observation stations for model training that can be directly used to train the machine learning regressors.
SoilGrids_Pred_USCS_15-30cm.xlsx is part of the prediction of the example study site based on SoilGrids pixels.


USCS_classfiers.py include 4 tree-based models (RF,LCE,XGBOOST,GBDT) for the prediction of USCS soil classification by using the data extracted_gSSURGO_data.xlsx
evaluation metrics of accuracy, precision, recall, f1-score, kappa, confusion matrix are all included. SHAP interpretation method is also included in this file.


