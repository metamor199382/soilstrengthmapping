# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:15:49 2022

@author: ruizhen_wang
"""
import joblib
import pandas as pd
import numpy as np

input_data = pd.read_csv('E:/mosaicked_SDR/10m 20cm/10m 20cm/Data_for_Predict_20191005_3_model5.csv')
input_data = np.array(input_data, dtype = 'float16')

random_forest_model = joblib.load('E:/SDR-SMN_extraction/SDR_RF_model5_20cm_lr11.pkl')


result = random_forest_model.predict(input_data)
out_result = pd.DataFrame(result)
out_result = out_result.to_csv('E:/mosaicked_SDR/10m 20cm/Predicted_20191005_3_model5_20cm_lr11.csv', index=0)

# if os.path.exists('E:/mosaicked_SDR/SM_20191005.tif'):
#     os.remove('E:/mosaicked_SDR/SM_20191005.tif')

# output_path = 'E:/mosaicked_SDR/SM_20191005.tif'
# driver = gdal.GetDriverByName("Gtiff")
# outdataset = driver.Create(output_path, shape[1], shape[0], 1, gdal.GDT_Float32)
# outdataset.SetGeoTransform(geotransform)
# outdataset.SetProjection(geoprojection)
# outband1 = outdataset.GetRasterBand(1)
# outband1.WriteArray(result)
# outdataset.FlushCache()
