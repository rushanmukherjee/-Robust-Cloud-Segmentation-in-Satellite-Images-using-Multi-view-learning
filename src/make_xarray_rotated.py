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
import pandas as pd
from pathlib import Path

train_shape = (8490, 512, 512)
test_shape = (975, 512, 512)
val_shape = (535, 512, 512)
save_dir = Path("/xarray/train").mkdir(parents=True, exist_ok=True)

# ## Sentinel 2 L1C bands being loaded
# L1C_B1 = np.memmap('/data/train/L1C_B1.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B2 = np.memmap('/data/train/L1C_B2.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B3 = np.memmap('/data/train/L1C_B3.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B4 = np.memmap('/data/train/L1C_B4.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B5 = np.memmap('/data/train/L1C_B5.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B6 = np.memmap('/data/train/L1C_B6.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B7 = np.memmap('/data/train/L1C_B7.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B8 = np.memmap('/data/train/L1C_B8.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B8A = np.memmap('/data/train/L1C_B8A.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B9 = np.memmap('/data/train/L1C_B9.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B10 = np.memmap('/data/train/L1C_B10.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B11 = np.memmap('/data/train/L1C_B11.dat', dtype='int16', mode='r', shape=train_shape)
# L1C_B12 = np.memmap('/data/train/L1C_B12.dat', dtype='int16', mode='r', shape=train_shape)

# L1C0_stacked = np.stack((L1C_B1, L1C_B2, L1C_B3, L1C_B4, L1C_B5, L1C_B6, L1C_B7, L1C_B8, L1C_B8A, L1C_B9, L1C_B10, L1C_B11, L1C_B12), 
#                         axis=1)                  
# print(f"L1C0_stacked shape : {L1C0_stacked.shape}")

# ##rotated the L1C product by 90 degrees
# L1C_B1_90 = np.rot90(L1C_B1, axes=(1,2))
# L1C_B2_90 = np.rot90(L1C_B2, axes=(1,2))
# L1C_B3_90 = np.rot90(L1C_B3, axes=(1,2))
# L1C_B4_90 = np.rot90(L1C_B4, axes=(1,2))
# L1C_B5_90 = np.rot90(L1C_B5, axes=(1,2))
# L1C_B6_90 = np.rot90(L1C_B6, axes=(1,2))
# L1C_B7_90 = np.rot90(L1C_B7, axes=(1,2))
# L1C_B8_90 = np.rot90(L1C_B8, axes=(1,2))
# L1C_B8A_90 = np.rot90(L1C_B8A, axes=(1,2))
# L1C_B9_90 = np.rot90(L1C_B9, axes=(1,2))
# L1C_B10_90 = np.rot90(L1C_B10, axes=(1,2))
# L1C_B11_90 = np.rot90(L1C_B11, axes=(1,2))
# L1C_B12_90 = np.rot90(L1C_B12, axes=(1,2))

# L1C90_stacked = np.stack((L1C_B1_90, L1C_B2_90, L1C_B3_90, L1C_B4_90, L1C_B5_90, L1C_B6_90, L1C_B7_90, L1C_B8_90, L1C_B8A_90, L1C_B9_90, L1C_B10_90, L1C_B11_90, L1C_B12_90), axis=1)
# print(f"L1C90_stacked shape : {L1C90_stacked.shape}")

# ##rotated the L1C product by 180 degrees
# L1C_B1_180 = np.rot90(L1C_B1, 2, axes=(1,2))
# L1C_B2_180 = np.rot90(L1C_B2, 2, axes=(1,2))
# L1C_B3_180 = np.rot90(L1C_B3, 2, axes=(1,2))
# L1C_B4_180 = np.rot90(L1C_B4, 2, axes=(1,2))
# L1C_B5_180 = np.rot90(L1C_B5, 2, axes=(1,2))
# L1C_B6_180 = np.rot90(L1C_B6, 2, axes=(1,2))
# L1C_B7_180 = np.rot90(L1C_B7, 2, axes=(1,2))
# L1C_B8_180 = np.rot90(L1C_B8, 2, axes=(1,2))
# L1C_B8A_180 = np.rot90(L1C_B8A, 2, axes=(1,2))
# L1C_B9_180 = np.rot90(L1C_B9, 2, axes=(1,2))
# L1C_B10_180 = np.rot90(L1C_B10, 2, axes=(1,2))
# L1C_B11_180 = np.rot90(L1C_B11, 2, axes=(1,2))
# L1C_B12_180 = np.rot90(L1C_B12, 2, axes=(1,2))

# L1C180_stacked = np.stack((L1C_B1_180, L1C_B2_180, L1C_B3_180, L1C_B4_180, L1C_B5_180, L1C_B6_180, L1C_B7_180, L1C_B8_180, L1C_B8A_180, L1C_B9_180, L1C_B10_180, L1C_B11_180, L1C_B12_180), axis=1) 
# print(f"L1C180_stacked shape : {L1C180_stacked.shape}")

# # ##rotated the L1C product by 270 degrees
# L1C_B1_270 = np.rot90(L1C_B1, -1, axes=(1,2))
# L1C_B2_270 = np.rot90(L1C_B1, -1, axes=(1,2))
# L1C_B3_270 = np.rot90(L1C_B3, -1, axes=(1,2))
# L1C_B4_270 = np.rot90(L1C_B4, -1, axes=(1,2))
# L1C_B5_270 = np.rot90(L1C_B5, -1, axes=(1,2))
# L1C_B6_270 = np.rot90(L1C_B6, -1, axes=(1,2))
# L1C_B7_270 = np.rot90(L1C_B7, -1, axes=(1,2))
# L1C_B8_270 = np.rot90(L1C_B8, -1, axes=(1,2))
# L1C_B8A_270 = np.rot90(L1C_B8A, -1, axes=(1,2))
# L1C_B9_270 = np.rot90(L1C_B9, -1, axes=(1,2))
# L1C_B10_270 = np.rot90(L1C_B10, -1, axes=(1,2))
# L1C_B11_270 = np.rot90(L1C_B11, -1, axes=(1,2))
# L1C_B12_270 = np.rot90(L1C_B12, -1, axes=(1,2))

# L1C270_stacked = np.stack((L1C_B1_270, L1C_B2_270, L1C_B3_270, L1C_B4_270, L1C_B5_270, L1C_B6_270, L1C_B7_270, L1C_B8_270, L1C_B8A_270, L1C_B9_270, L1C_B10_270, L1C_B11_270, L1C_B12_270), axis=1)
# print(f"L1C270_stacked shape : {L1C270_stacked.shape}")

# # ## Xarray for S2 L1C view
# stacked_s2L1C = np.concatenate((L1C0_stacked, L1C90_stacked, L1C180_stacked, L1C270_stacked))
# print(f"stacked_s2L1C shape: {stacked_s2L1C.shape}")

# s2L1C_xarray = xr.DataArray( 
#     stacked_s2L1C,
#     coords = {
#     "bands" : ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
#     },
#     dims=["size", "bands", "height", "width"]
#     )

# ## Try with new mount
# s2L1C_xarray.to_netcdf("/xarray/train/s2L1C_withrotated.nc")

# ## Sentinel 1 information being loaded
# S1_angle = np.memmap('/data/train/S1_angle.dat', dtype='int16', mode='r', shape=train_shape)
# S1_VH = np.memmap('/data/train/S1_VH.dat', dtype='int16', mode='r', shape=train_shape)
# S1_VV = np.memmap('/data/train/S1_VV.dat', dtype='int16', mode='r', shape=train_shape)

# S10_stacked = np.stack((S1_angle, S1_VH, S1_VV), axis=1)
# print(f"s1_0_stacked shape: {S10_stacked.shape}")

# ## Rotate the S1 product by 90 degrees
# S1_angle_90 = np.rot90(S1_angle, axes=(1,2))
# S1_VH_90 = np.rot90(S1_VH, axes=(1,2))
# S1_VV_90 = np.rot90(S1_VV, axes=(1,2))

# S190_stacked = np.stack((S1_angle_90, S1_VH_90, S1_VV_90), axis=1)
# print(f"s1_90_stacked shape: {S190_stacked.shape}")

# ## Rotate the S1 product by 180 degrees
# S1_angle_180 = np.rot90(S1_angle, 2, axes=(1,2))
# S1_VH_180 = np.rot90(S1_VH, 2, axes=(1,2))
# S1_VV_180 = np.rot90(S1_VV, 2, axes=(1,2))

# S1180_stacked = np.stack((S1_angle_180, S1_VH_180, S1_VV_180), axis=1)
# print(f"s1_180_stacked shape: {S1180_stacked.shape}")

# ## Rotate the S1 product by 270 degrees
# S1_angle_270 = np.rot90(S1_angle, -1, axes=(1,2))
# S1_VH_270 = np.rot90(S1_VH, -1, axes=(1,2))
# S1_VV_270 = np.rot90(S1_VV, -1, axes=(1,2))

# S1270_stacked = np.stack((S1_angle_270, S1_VH_270, S1_VV_270), axis=1)
# print(f"s1_270_stacked shape: {S1270_stacked.shape}")

# stacked_s1 = np.concatenate((S10_stacked, S190_stacked, S1180_stacked, S1270_stacked))
# print(f"stacked_s1 shape: {stacked_s1.shape}")

# s1_xarray = xr.DataArray(
#     stacked_s1,
#     coords = {
#     "bands" : ['angle', 'VH', 'VV']
#     },
#     dims=["size", "bands", "height", "width"]
#     )

# s1_xarray.to_netcdf("/xarray/train/s1_withrotated.nc")

# Extra Information
ex_lc10 = np.memmap('/data/train/EXTRA_lc10.dat', dtype='int16', mode='r', shape=train_shape)
ex_elevation = np.memmap('/data/train/EXTRA_elevation.dat', dtype='int16', mode='r', shape=train_shape)
ex_shwdirection = np.memmap('/data/train/EXTRA_Shwdirection.dat', dtype='int16', mode='r', shape=train_shape)
ex_water = np.memmap('/data/train/EXTRA_ocurrence.dat', dtype='int16', mode='r', shape=train_shape)

# extra0_stacked = np.stack((ex_lc10, ex_elevation, ex_shwdirection, ex_water), axis=1)
# print(f"extra0_stacked shape: {extra0_stacked.shape}")

# ## Rotate the extra information by 90 degrees
# ex_lc10_90 = np.rot90(ex_lc10, axes=(1,2))
# ex_elevation_90 = np.rot90(ex_elevation, axes=(1,2))
# ex_shwdirection_90 = np.rot90(ex_shwdirection, axes=(1,2)) 
# ex_water_90 = np.rot90(ex_water, axes=(1,2))

# extra90_stacked = np.stack((ex_lc10_90, ex_elevation_90, ex_shwdirection_90, ex_water_90), axis=1)
# print(f"extra90_stacked shape: {extra90_stacked.shape}")

# ## Rotate the extra information by 180 degrees
# ex_lc10_180 = np.rot90(ex_lc10, 2, axes=(1,2))
# ex_elevation_180 = np.rot90(ex_elevation, 2, axes=(1,2))
# ex_shwdirection_180 = np.rot90(ex_shwdirection, 2, axes=(1,2))
# ex_water_180 = np.rot90(ex_water, 2, axes=(1,2))

# extra180_stacked = np.stack((ex_lc10_180, ex_elevation_180, ex_shwdirection_180, ex_water_180), axis=1)
# print(f"extra180_stacked shape: {extra180_stacked.shape}")

# ## Rotate the extra information by 270 degrees
# ex_lc10_270 = np.rot90(ex_lc10, -1, axes=(1,2))
# ex_elevation_270 = np.rot90(ex_elevation, -1, axes=(1,2))
# ex_shwdirection_270 = np.rot90(ex_shwdirection, -1, axes=(1,2))
# ex_water_270 = np.rot90(ex_water, -1, axes=(1,2))

# extra270_stacked = np.stack((ex_lc10_270, ex_elevation_270, ex_shwdirection_270, ex_water_270), axis=1)
# print(f"extra270_stacked shape: {extra270_stacked.shape}")

# stacked_extra = np.concatenate((extra0_stacked, extra90_stacked, extra180_stacked, extra270_stacked))
# print(f"stacked_extra shape: {stacked_extra.shape}")


landcoverproduct = xr.DataArray(
    ex_lc10,
    dims = ["size", "height", "width"]
)
DEM = xr.DataArray(
    ex_elevation,
    dims = ["size", "height", "width"]
)
wateroccurence = xr.DataArray(
    ex_water,
    dims = ["size", "height", "width"]
)
azimuth = xr.DataArray(
    ex_shwdirection,
    dims = ["size", "height", "width"]
)

landcoverproduct.to_netcdf("/xarray/train/landcover.nc")
DEM.to_netcdf("/xarray/train/DEM.nc")
wateroccurence.to_netcdf("/xarray/train/wateroccurence.nc")
azimuth.to_netcdf("/xarray/train/azimuth.nc")

# ## Labels  
# train_labels = np.memmap('/data/train/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=train_shape)
# train_labels_90 = np.rot90(train_labels, axes=(1,2))
# train_labels_180 = np.rot90(train_labels, 2, axes=(1,2))
# train_labels_270 = np.rot90(train_labels, -1, axes=(1,2))

# stacked_labels = np.concatenate((train_labels, train_labels_90, train_labels_180, train_labels_270))

# print(f"stacked_labels shape: {stacked_labels.shape}")

# labels_xarray = xr.DataArray(stacked_labels)
# labels_xarray.to_netcdf("/xarray/train/HQlabels_withrotated.nc")

print(f"Saved information to /xarray/train")