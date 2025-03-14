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

## Sentinel 2 L1C bands being loaded
L1C_B1 = np.memmap('/data/train/L1C_B1.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B2 = np.memmap('/data/train/L1C_B2.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B3 = np.memmap('/data/train/L1C_B3.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B4 = np.memmap('/data/train/L1C_B4.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B5 = np.memmap('/data/train/L1C_B5.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B6 = np.memmap('/data/train/L1C_B6.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B7 = np.memmap('/data/train/L1C_B7.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B8 = np.memmap('/data/train/L1C_B8.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B8A = np.memmap('/data/train/L1C_B8A.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B9 = np.memmap('/data/train/L1C_B9.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B10 = np.memmap('/data/train/L1C_B10.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B11 = np.memmap('/data/train/L1C_B11.dat', dtype='int16', mode='r', shape=train_shape)
L1C_B12 = np.memmap('/data/train/L1C_B12.dat', dtype='int16', mode='r', shape=train_shape)

##rotated the L1C product by 90 degrees
L1C_B1_90 = np.rot90(L1C_B1)
L1C_B2_90 = np.rot90(L1C_B2)
L1C_B3_90 = np.rot90(L1C_B3)
L1C_B4_90 = np.rot90(L1C_B4)
L1C_B5_90 = np.rot90(L1C_B5)
L1C_B6_90 = np.rot90(L1C_B6)
L1C_B7_90 = np.rot90(L1C_B7)
L1C_B8_90 = np.rot90(L1C_B8)
L1C_B8A_90 = np.rot90(L1C_B8A)
L1C_B9_90 = np.rot90(L1C_B9)
L1C_B10_90 = np.rot90(L1C_B10)
L1C_B11_90 = np.rot90(L1C_B11)
L1C_B12_90 = np.rot90(L1C_B12)

##rotated the L1C product by 180 degrees
L1C_B1_180 = np.rot90(L1C_B1, 2)
L1C_B2_180 = np.rot90(L1C_B2, 2)
L1C_B3_180 = np.rot90(L1C_B3, 2)
L1C_B4_180 = np.rot90(L1C_B4, 2)
L1C_B5_180 = np.rot90(L1C_B5, 2)
L1C_B6_180 = np.rot90(L1C_B6, 2)
L1C_B7_180 = np.rot90(L1C_B7, 2)
L1C_B8_180 = np.rot90(L1C_B8, 2)
L1C_B8A_180 = np.rot90(L1C_B8A, 2)
L1C_B9_180 = np.rot90(L1C_B9, 2)
L1C_B10_180 = np.rot90(L1C_B10, 2)
L1C_B11_180 = np.rot90(L1C_B11, 2)
L1C_B12_180 = np.rot90(L1C_B12, 2)

##rotated the L1C product by 270 degrees
L1C_B1_180 = np.rot90(L1C_B1, -1)
L1C_B1_180 = np.rot90(L1C_B1, -1)
L1C_B3_180 = np.rot90(L1C_B3, -1)
L1C_B4_180 = np.rot90(L1C_B4, -1)
L1C_B5_180 = np.rot90(L1C_B5, -1)
L1C_B6_180 = np.rot90(L1C_B6, -1)
L1C_B7_180 = np.rot90(L1C_B7, -1)
L1C_B8_180 = np.rot90(L1C_B8, -1)
L1C_B8A_180 = np.rot90(L1C_B8A, -1)
L1C_B9_180 = np.rot90(L1C_B9, -1)
L1C_B10_180 = np.rot90(L1C_B10, -1)
L1C_B11_180 = np.rot90(L1C_B11, -1)
L1C_B12_180 = np.rot90(L1C_B12, -1)

## Sentinel 1 information being loaded
S1_angle = np.memmap('/data/train/S1_angle.dat', dtype='int16', mode='r', shape=train_shape)
S1_VH = np.memmap('/data/train/S1_VH.dat', dtype='int16', mode='r', shape=train_shape)
S1_VV = np.memmap('/data/train/S1_VV.dat', dtype='int16', mode='r', shape=train_shape)

## Extra Information
ex_lc10 = np.memmap('/data/train/EXTRA_lc10.dat', dtype='int16', mode='r', shape=train_shape)
ex_elevation = np.memmap('/data/train/EXTRA_elevation.dat', dtype='int16', mode='r', shape=train_shape)
ex_shwdirection = np.memmap('/data/train/EXTRA_Shwdirection.dat', dtype='int16', mode='r', shape=train_shape)
ex_water = np.memmap('/data/train/EXTRA_ocurrence.dat', dtype='int16', mode='r', shape=train_shape)

#s2_data = xr.DataArray()
y = np.memmap('/data/train/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=train_shape)

metadata = pd.read_csv("/data/train/metadata.csv")

## Xarray for S2 L1C view
stacked_s2_L1C = np.stack((L1C_B1,L1C_B2,L1C_B3,L1C_B4,L1C_B5,L1C_B6,L1C_B7,L1C_B8,L1C_B8A,L1C_B9,L1C_B10,L1C_B11,L1C_B12), axis=1)
s2L1C_xarray = xr.DataArray( 
    stacked_s2_L1C,
    coords = {
    "bands" : ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
    },
    dims=["size", "bands", "height", "width"]
    )

## Xarray for S1 view 
stacked_s1 = np.stack((S1_angle,S1_VH,S1_VV), axis=1)
s1_xarray = xr.DataArray(
    stacked_s1,
    coords = {
    "bands" : ['angle', 'VH', 'VV']
    },
    dims=["size", "bands", "height", "width"]
    )

## Xarray for extra information
stacked_extra = np.stack((ex_lc10, ex_elevation, ex_water, ex_shwdirection), axis=1)
extra_xarray = xr.DataArray(
    stacked_extra,
    coords = {
        "labels" : ['landcoverproduct', 'DEM', 'waterocurrence', 'azimuth']
    },
    dims = ["size", "labels", "height", "width"]
)

## Saving data arrays to disk

save_dir = Path("/xarray_dataset/train").mkdir(parents=True, exist_ok=True)
s2L1C_xarray.to_netcdf("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset/train/s2L1C.nc")
s1_xarray.to_netcdf("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset/train/s1.nc")
extra_xarray.to_netcdf("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset/train/extra.nc")

print(f"all files saved")

## Sentinel 2 L2A bands being loaded
# L2A_B1 = np.memmap('/data/train/L2A_B1.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B2 = np.memmap('/data/train/L2A_B2.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B3 = np.memmap('/data/train/L2A_B3.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B4 = np.memmap('/data/train/L2A_B4.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B5 = np.memmap('/data/train/L2A_B5.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B6 = np.memmap('/data/train/L2A_B6.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B7 = np.memmap('/data/train/L2A_B7.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B8 = np.memmap('/data/train/L2A_B8.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B8A = np.memmap('/data/train/L2A_B8A.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B9 = np.memmap('/data/train/L2A_B9.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B11 = np.memmap('/data/train/L2A_B11.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_B12 = np.memmap('/data/train/L2A_B12.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_AOT = np.memmap('/data/train/L2A_AOT.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_WVP = np.memmap('/data/train/L2A_WVP.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_TCI_B = np.memmap('/data/train/L2A_TCI_B.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_TCI_G = np.memmap('/data/train/L2A_TCI_G.dat', dtype='int16', mode='r', shape=train_shape)
# L2A_TCI_R = np.memmap('/data/train/L2A_TCI_R.dat', dtype='int16', mode='r', shape=train_shape)

## Xarray for S2 L2A view
# stacked_s2_L2A = np.stack((L2A_B1,L2A_B2,L2A_B3,L2A_B4,L2A_B5,L2A_B6,L2A_B7,L2A_B8,L2A_B8A,L2A_B9,L2A_B11,L2A_B12,L2A_AOT, L2A_WVP, L2A_TCI_B, L2A_TCI_G, L2A_TCI_R), axis=1)
# s2L2A_xarray = xr.DataArray(
#     stacked_s2_L2A,
#     coords = {
#     "bands" : ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12', 'AOT', 'WVP', 'TCI_B', 'TCI_G', 'TCI_R']
#     },
#     dims=["size", "bands", "height", "width"]
#     )

# s2L2A_xarray.to_netcdf("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset/train/s2L2A.nc")