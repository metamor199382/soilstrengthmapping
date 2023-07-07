# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 13:07:35 2022

@author: ruizhen_wang
"""
import os
import pandas as pd
import numpy as np
from osgeo import gdal
from rosetta import rosetta, SoilData

#通用参数
Slope = gdal.Open('E:/mosaicked_SDR/Slope.tif')
DEM = gdal.Open('E:/mosaicked_SDR/mosaic_20191005.data/DEM.img')
Aspect = gdal.Open('E:/mosaicked_SDR/aspect.tif')

# clay = gdal.Open('E:/mosaicked_SDR/10m 20cm/10m 20cm/Soilgrid_clay.tif')
# silt = gdal.Open('E:/mosaicked_SDR/10m 20cm/10m 20cm/Soilgrid_silt.tif')
# sand = gdal.Open('E:/mosaicked_SDR/10m 20cm/10m 20cm/Soilgrid_sand.tif')
# oc = gdal.Open('E:/mosaicked_SDR/10m 20cm/10m 20cm/Soilgrid_OC.tif')
# bd = gdal.Open('E:/mosaicked_SDR/10m 20cm/10m 20cm/Soilgrid_bd.tif')
# thr = gdal.Open('E:/mosaicked_SDR/10m 0-5cm/thr.tif')
# ths = gdal.Open('E:/mosaicked_SDR/10m 0-5cm/ths.tif')
# ksat = gdal.Open('E:/mosaicked_SDR/10m 0-5cm/ksat.tif')


#影像特定参数
os.chdir('E:/mosaicked_SDR/mosaic_20191005.data')
VSDI = gdal.Open('VSDI.img')
NDVI = gdal.Open('NDVI.img')
NSDSI = gdal.Open('NSDSI3.img')
B8 = gdal.Open('B8.img')


#去空值否则无法运算
def raster_process(dataset):
    dataset = pd.DataFrame(dataset.ReadAsArray())
    dataset = dataset[109::]
    dataset.replace(np.nan, -9999, inplace = True)
    dataset = np.array(dataset).reshape(-1)
    return dataset

NSDSI = raster_process(NSDSI)
NDVI = raster_process(NDVI)
VSDI = raster_process(VSDI)
Slope = raster_process(Slope)
Aspect = raster_process(Aspect)
DEM = raster_process(DEM)
# clay = raster_process(clay)/10
# silt = raster_process(silt)/10
# sand = raster_process(sand)/10
# oc = raster_process(oc)/100
# bd = raster_process(bd)/100
# thr = raster_process(thr)
# ths = raster_process(ths)
# ksat = raster_process(ksat)
B8 = raster_process(B8)
NTR = (1-B8)**2/(2*B8)
NTR = pd.Series(NTR)
NTR[NTR== np.inf] = -9999
NTR = np.array(NTR)
# porosity = 1-(bd/2.65)

# model9 = ['NTR', 'NDVI', 'VSDI', 'NSDSI', 'Slope', 'DEM', 'aspect','clay'+depth, 'sand'+depth,
#           'silt'+depth, 'OC'+depth, 'bulkdensity'+depth, 'porosity'+depth, 'residual water content'+depth, 
#           'saturated water content'+depth, 'saturated hydraulic conductivity'+depth,'SM_'+soil_depth]
#thr, ths, ksat,NTR, NDVI, 
input_data = np.array([VSDI, NSDSI, Slope, DEM, Aspect],dtype='float16').T
input_data = pd.DataFrame(input_data)
input_data1 = input_data.iloc[:30000000].to_csv('E:/mosaicked_SDR/10m 20cm/10m 20cm/Data_for_Predict_20191005_1_model2.csv', index = 0)
input_data2 = input_data.iloc[30000000:60000000].to_csv('E:/mosaicked_SDR/10m 20cm/10m 20cm/Data_for_Predict_20191005_2_model2.csv', index = 0)
input_data3 = input_data.iloc[60000000:].to_csv('E:/mosaicked_SDR/10m 20cm/10m 20cm/Data_for_Predict_20191005_3_model2.csv', index = 0)