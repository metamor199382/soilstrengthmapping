# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 14:41:17 2022

@author: ruizhen_wang

需要修改下方路径存在判定的文件名！！！
"""


import os
from osgeo import gdal
import numpy as np
import pandas as pd

os.chdir('E:/mosaicked_SDR/')
#SM_path = 'E:/mosaicked_SDR/SDR_SM/tif_sm/20191005new.tif'
SM_path = 'E:/mosaicked_SDR/SM_20191005_20cm_model5_lr11.tif'
USCS_path = 'C:/Users/ruizhen_wang/Desktop/一改/soilgrid/USCS_masked_10m.tif'
output_path = 'E:/mosaicked_SDR/SDR_and_RCI/RCI_20191005.tif'
BD_path = 'E:/mosaicked_SDR/10m 20cm/10m 20cm/Soilgrid_bd.tif'
data_SM = gdal.Open(SM_path)
data_USCS = gdal.Open(USCS_path)
data_bd = gdal.Open(BD_path)

print(data_SM.GetProjection())
print(data_SM.GetGeoTransform())
geotransform = data_SM.GetGeoTransform()
geoprojection = data_SM.GetProjection()

soil = data_USCS.GetRasterBand(1).ReadAsArray()
soil = soil[109:]
soil_moisture = data_SM.GetRasterBand(1).ReadAsArray()
soil_bd = data_bd.GetRasterBand(1).ReadAsArray()/100
soil_bd = soil_bd[109:]

cols = soil_moisture.shape[1]   #XSize
rows = soil_moisture.shape[0]   #YSize

soil = pd.DataFrame(soil)
soil_moisture = pd.DataFrame(soil_moisture)

soil_moisture[soil_moisture<0] = np.nan
soil_moisture[soil_moisture==0] = np.nan

RCI_soil_a = soil.copy()
RCI_soil_b = soil.copy()


#classification = '1:CL, 2:CL-ML, 4:ML, 5:SC, 6:SC-SM, 7: SM 

RCI_soil_a.replace([1,2,4,5,6,7], [15.506, 14.236, 11.936, 12.542, 12.542, 12.542], inplace=True)
RCI_soil_b.replace([1,2,4,5,6,7], [-3.530, -3.137, -2.407, -2.955, -2.955, -2.955], inplace=True)

RCI_soil_a = np.array(RCI_soil_a).ravel()
RCI_soil_b = np.array(RCI_soil_b).ravel()
soil_moisture = np.array(soil_moisture).ravel()
soil_bd = soil_bd.ravel()

RCI = np.exp(RCI_soil_a+RCI_soil_b*np.log(soil_moisture/soil_bd*100))          
RCI = pd.DataFrame(RCI)
RCI.to_csv('C:/Users/ruizhen_wang/Desktop/一改/RCI20191005.csv')


RCI_dataset = RCI.reshape(rows,cols)





driver = gdal.GetDriverByName("Gtiff")
outdataset = driver.Create(output_path, cols, rows, 2, gdal.GDT_Float32)
outdataset.SetGeoTransform(geotransform)
outdataset.SetProjection(geoprojection)
outband1 = outdataset.GetRasterBand(1)
outband1.WriteArray(RCI_dataset)
outdataset.FlushCache()
