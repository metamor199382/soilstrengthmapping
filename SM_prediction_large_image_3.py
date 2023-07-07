# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 18:19:59 2022

@author: ruizhen_wang
"""

import os
import pandas as pd
import numpy as np
from osgeo import gdal

input_data1 = pd.read_csv('E:/mosaicked_SDR/10m 20cm/Predicted_20191005_1_model5_20cm_lr11.csv')
input_data2 = pd.read_csv('E:/mosaicked_SDR/10m 20cm/Predicted_20191005_2_model5_20cm_lr11.csv')
input_data3 = pd.read_csv('E:/mosaicked_SDR/10m 20cm/Predicted_20191005_3_model5_20cm_lr11.csv')

data1 = np.array(input_data1).ravel()
data2 = np.array(input_data2).ravel()
data3 = np.array(input_data3).ravel()

SIPI = gdal.Open('E:/mosaicked_SDR/mosaic_20181109.data/SIPI.img')
geotransform = (374799.0728357713, 10.0, -0.0, 4700043.111945425, -0.0, -10.0)
geoprojection = SIPI.GetProjection()
SIPI = SIPI.ReadAsArray()
SIPI = SIPI[109::]
shape = SIPI.shape

result = np.concatenate([data1,data2,data3]).reshape(shape[0],shape[1])

#if os.path.exists('E:/mosaicked_SDR/SM_20190905.tif'):
#    os.remove('E:/mosaicked_SDR/SM_20190905.tif')

output_path = 'E:/mosaicked_SDR/SM_20191005_20cm_model5_lr11.tif'
driver = gdal.GetDriverByName("Gtiff")
outdataset = driver.Create(output_path, shape[1], shape[0], 1, gdal.GDT_Float32)
outdataset.SetGeoTransform(geotransform)
outdataset.SetProjection(geoprojection)
outband1 = outdataset.GetRasterBand(1)
outband1.WriteArray(result)
outdataset.FlushCache()
