import os
import json
import random
from os.path import join, isfile
from typing import Union, Tuple, Dict, List
from glob import glob

import torch
import numpy as np
from torch.utils.data import Dataset
import random
import xarray as xr

##S2 max and min 
GLOBAL_S2_MAX = 28004.
GLOBAL_S2_MIN = 0

##S1 max and min
GLOBAL_S1_MAX = 32767
GLOBAL_S1_MIN = -32768

##DEM max and min
GLOBAL_DEM_MAX= 7604
GLOBAL_DEM_MIN = -71

GLOBAL_azimuth_MAX= 24213
GLOBAL_azimuth_MIN = 0

GLOBAL_wateroccurence_MAX= 255
GLOBAL_wateroccurence_MIN = 0

## Define min-max scaling parameters
RANGE_MIN = -1
RANGE_MAX = 1


##Dataset class for importing the cloudsen12 dataset from cluster
class AllModalityDatasetBandRemoval(Dataset):

    def __init__(self, drop_band_flag, drop_band_count, rotate_flag, label_dir, s1_dir, s2_dir, landcover_dir, azimuth_dir, wateroccurence_dir, DEM_dir) -> None:
        
        super(AllModalityDatasetBandRemoval, self).__init__()
        self.drop_band_flag = drop_band_flag
        self.drop_band_count = drop_band_count
        self.rotate_flag = rotate_flag
        self.labels = xr.open_dataarray(label_dir)
        self.s1 = xr.open_dataarray(s1_dir)
        self.s2 = xr.open_dataarray(s2_dir)
        self.landcover = xr.open_dataarray(landcover_dir)
        self.azimuth = xr.open_dataarray(azimuth_dir)
        self.wateroccurence = xr.open_dataarray(wateroccurence_dir)
        self.elevation = xr.open_dataarray(DEM_dir)
        self.rotation_angles = [0, 90, 180, 270]
    
    def check_negative_strides(self, arr):
        if any(stride < 0 for stride in arr.strides):
            print(f"Array with negative strides found!")
            print(f"Strides: {arr.strides}")

    def __getitem__(self, index) -> Dict['str', torch.Tensor]:
        
        rand_angle = random.choice(self.rotation_angles)
    
        s2_coords = self.s2[index].coords
        #print(s2_coords)

        ##To return the input and target values for the given index
        sen1 = self.s1[index].values
        sen2 = self.s2[index].values
        landcover = self.landcover[index].values
        azimuth = self.azimuth[index].values
        wateroccurence = self.wateroccurence[index].values
        elevation = self.elevation[index].values
        labels = self.labels[index].values

        sen1 = sen1.astype(dtype='float32')
        sen2 = sen2.astype(dtype='float32')
        azimuth = azimuth.astype(dtype='float32')
        wateroccurence = wateroccurence.astype(dtype='float32')
        elevation = elevation.astype(dtype='float32')

        ##Scaling Sentinel 2 values to range [-1,1]
        sen2_std = (sen2 - GLOBAL_S2_MIN) / (GLOBAL_S2_MAX - GLOBAL_S2_MIN)
        sen2 = sen2_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

        if self.drop_band_flag == True:
            if self.drop_band_count == 1:
                sen2[0, :, :] = 0
            elif self.drop_band_count == 2:
                sen2[0:1, :, :] = 0
            elif self.drop_band_count == 3:
                sen2[0:2, :, :] = 0
            elif self.drop_band_count == 4:
                sen2[0:3, :, :] = 0
            elif self.drop_band_count == 5:
                sen2[0:4, :, :] = 0
            elif self.drop_band_count == 6:
                sen2[0:5, :, :] = 0
            elif self.drop_band_count == 7:
                sen2[0:6, :, :] = 0
            elif self.drop_band_count == 8:
                sen2[0:7, :, :] = 0
            elif self.drop_band_count == 9:
                sen2[0:8, :, :] = 0
            elif self.drop_band_count == 10:
                sen2[0:9, :, :] = 0
            elif self.drop_band_count == 11:
                sen2[0:10, :, :] = 0
            elif self.drop_band_count == 12:
                sen2[0:11, :, :] = 0


            

        ##Scaling Sentinel 1 values to range [-1,1]
        sen1_std = (sen1 - GLOBAL_S1_MIN) / (GLOBAL_S1_MAX - GLOBAL_S1_MIN)
        sen1 = sen1_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

        ##Scaling Extra values to range [-1,1]

        azimuth_std = (azimuth - GLOBAL_azimuth_MIN) / (GLOBAL_azimuth_MAX - GLOBAL_azimuth_MIN)
        azimuth = azimuth_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

        wateroccurence_std = (wateroccurence - GLOBAL_wateroccurence_MIN) / (GLOBAL_wateroccurence_MAX - GLOBAL_wateroccurence_MIN)
        wateroccurence = wateroccurence_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

        elevation_std = (elevation - GLOBAL_DEM_MIN) / (GLOBAL_DEM_MAX - GLOBAL_DEM_MIN)
        elevation = elevation_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
        
        if self.rotate_flag==True:
            if rand_angle==0:
                rotated_sen2 = sen2.copy()
                rotated_sen1 = sen1.copy()
                rotated_landcover = landcover.copy()
                rotated_azimuth = azimuth.copy()
                rotated_wateroccurence = wateroccurence.copy()
                rotated_elevation = elevation.copy()
                rotated_label = labels.copy()
            
            elif rand_angle==90:
                ##Sen1 and Sen2 have axes mentioned because they have channels as first dimension
                ## For other arrays, axes are not mentioned as they have only 2 dimensions
                rotated_sen2 = np.rot90(sen2, axes = (1,2)).copy()
                rotated_sen1 = np.rot90(sen1, axes = (1,2)).copy()
                rotated_landcover = np.rot90(landcover, axes = (0,1)).copy()
                rotated_azimuth = np.rot90(azimuth).copy()
                rotated_wateroccurence = np.rot90(wateroccurence).copy()
                rotated_elevation = np.rot90(elevation).copy()
                rotated_label = np.rot90(labels).copy()

            elif rand_angle==180:
                ##Sen1 and Sen2 have axes mentioned because they have channels as first dimension
                ## For other arrays, axes are not mentioned as they have only 2 dimensions
                rotated_sen2 = np.rot90(sen2, 2, axes = (1,2)).copy()
                rotated_sen1 = np.rot90(sen1, 2, axes = (1,2)).copy()
                rotated_landcover = np.rot90(landcover, 2, axes=(0,1)).copy()
                rotated_azimuth = np.rot90(azimuth, 2).copy()
                rotated_wateroccurence = np.rot90(wateroccurence, 2).copy()
                rotated_elevation = np.rot90(elevation, 2).copy()
                rotated_label = np.rot90(labels, 2).copy()

            else:
                ##Sen1 and Sen2 have axes mentioned because they have channels as first dimension
                ## For other arrays, axes are not mentioned as they have only 2 dimensions
                rotated_sen2 = np.rot90(sen2, 3, axes = (1,2)).copy()
                rotated_sen1 = np.rot90(sen1, 3, axes = (1,2)).copy()
                rotated_landcover = np.rot90(landcover, 3, axes=(0,1)).copy()
                rotated_azimuth = np.rot90(azimuth, 3).copy()
                rotated_wateroccurence = np.rot90(wateroccurence, 3).copy()
                rotated_elevation = np.rot90(elevation, 3).copy()
                rotated_label = np.rot90(labels, 3).copy()
        
        else:
            rotated_sen2 = sen2.copy()
            rotated_sen1 = sen1.copy()
            rotated_landcover = landcover.copy()
            rotated_azimuth = azimuth.copy()
            rotated_wateroccurence = wateroccurence.copy()
            rotated_elevation = elevation.copy()
            rotated_label = labels.copy()

        # print(f"checking negative strides for index {index}")
        # print(f"checking S1")
        # self.check_negative_strides(rotated_sen1)
        # print(f"checking S2")
        # self.check_negative_strides(rotated_sen2)
        # print(f"checking landcover")
        # self.check_negative_strides(rotated_landcover)
        # print(f"checking azimuth")
        # self.check_negative_strides(rotated_azimuth)
        # print(f"checking wateroccurence")
        # self.check_negative_strides(rotated_wateroccurence)
        # print(f"checking elevation")
        # self.check_negative_strides(rotated_elevation)
        # print(f"checking label")
        # self.check_negative_strides(rotated_label)

        # Assemble dicts to return inputs and targets
        items = {
            's1': rotated_sen1, 
            's2': rotated_sen2,
            'landcover': rotated_landcover,
            'azimuth': rotated_azimuth,
            'wateroccurence': rotated_wateroccurence,
            'elevation': rotated_elevation,
            'cloud_labels': rotated_label
        }

        return items
    
    def __len__(self) -> int:

        ##To return the number of samples in dataset
        label_vals = self.labels['dim_0'].values

        return label_vals.size
    
    def load_s1(self,idx):

        sen1 = self.s1
        return sen1
    
    