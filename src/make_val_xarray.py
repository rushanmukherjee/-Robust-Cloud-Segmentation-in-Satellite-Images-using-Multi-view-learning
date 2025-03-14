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

val_shape = (535, 512, 512)

## Sentinel 2 L1C bands being loaded
L1C_B1 = np.memmap('/data/val/L1C_B1.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B2 = np.memmap('/data/val/L1C_B2.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B3 = np.memmap('/data/val/L1C_B3.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B4 = np.memmap('/data/val/L1C_B4.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B5 = np.memmap('/data/val/L1C_B5.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B6 = np.memmap('/data/val/L1C_B6.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B7 = np.memmap('/data/val/L1C_B7.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B8 = np.memmap('/data/val/L1C_B8.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B8A = np.memmap('/data/val/L1C_B8A.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B9 = np.memmap('/data/val/L1C_B9.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B10 = np.memmap('/data/val/L1C_B10.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B11 = np.memmap('/data/val/L1C_B11.dat', dtype='int16', mode='r', shape=val_shape)
L1C_B12 = np.memmap('/data/val/L1C_B12.dat', dtype='int16', mode='r', shape=val_shape)

## Sentinel 2 L2A bands being loaded
L2A_B1 = np.memmap('/data/val/L2A_B1.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B2 = np.memmap('/data/val/L2A_B2.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B3 = np.memmap('/data/val/L2A_B3.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B4 = np.memmap('/data/val/L2A_B4.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B5 = np.memmap('/data/val/L2A_B5.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B6 = np.memmap('/data/val/L2A_B6.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B7 = np.memmap('/data/val/L2A_B7.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B8 = np.memmap('/data/val/L2A_B8.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B8A = np.memmap('/data/val/L2A_B8A.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B9 = np.memmap('/data/val/L2A_B9.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B11 = np.memmap('/data/val/L2A_B11.dat', dtype='int16', mode='r', shape=val_shape)
L2A_B12 = np.memmap('/data/val/L2A_B12.dat', dtype='int16', mode='r', shape=val_shape)
L2A_AOT = np.memmap('/data/val/L2A_AOT.dat', dtype='int16', mode='r', shape=val_shape)
L2A_WVP = np.memmap('/data/val/L2A_WVP.dat', dtype='int16', mode='r', shape=val_shape)
L2A_TCI_B = np.memmap('/data/val/L2A_TCI_B.dat', dtype='int16', mode='r', shape=val_shape)
L2A_TCI_G = np.memmap('/data/val/L2A_TCI_G.dat', dtype='int16', mode='r', shape=val_shape)
L2A_TCI_R = np.memmap('/data/val/L2A_TCI_R.dat', dtype='int16', mode='r', shape=val_shape)

## Sentinel 1 information being loaded
S1_angle = np.memmap('/data/val/S1_angle.dat', dtype='int16', mode='r', shape=val_shape)
S1_VH = np.memmap('/data/val/S1_VH.dat', dtype='int16', mode='r', shape=val_shape)
S1_VV = np.memmap('/data/val/S1_VV.dat', dtype='int16', mode='r', shape=val_shape)

## Extra Information
ex_lc10 = np.memmap('/data/val/EXTRA_lc10.dat', dtype='int16', mode='r', shape=val_shape)
ex_elevation = np.memmap('/data/val/EXTRA_elevation.dat', dtype='int16', mode='r', shape=val_shape)
ex_shwdirection = np.memmap('/data/val/EXTRA_Shwdirection.dat', dtype='int16', mode='r', shape=val_shape)
ex_water = np.memmap('/data/val/EXTRA_ocurrence.dat', dtype='int16', mode='r', shape=val_shape)

#s2_data = xr.DataArray()
y = np.memmap('/data/val/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=val_shape)
#labels = np.memmap('/data/CloudSEN12-high/val/LABEL_manual_hq.dat', dtype='int16', mode='r+', shape=val_shape)

## Xarray for S2 L1C view
stacked_s2_L1C = np.stack((L1C_B1,L1C_B2,L1C_B3,L1C_B4,L1C_B5,L1C_B6,L1C_B7,L1C_B8,L1C_B8A,L1C_B9,L1C_B10,L1C_B11,L1C_B12), axis=1)
s2L1C_xarray = xr.DataArray(
    stacked_s2_L1C,
    coords = {
    "bands" : ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B10','B11','B12']
    },
    dims=["size", "bands", "height", "width"]
    )

## Xarray for S2 L2A view
stacked_s2_L2A = np.stack((L2A_B1,L2A_B2,L2A_B3,L2A_B4,L2A_B5,L2A_B6,L2A_B7,L2A_B8,L2A_B8A,L2A_B9,L2A_B11,L2A_B12,L2A_AOT, L2A_WVP, L2A_TCI_B, L2A_TCI_G, L2A_TCI_R), axis=1)
s2L2A_xarray = xr.DataArray(
    stacked_s2_L2A,
    coords = {
    "bands" : ['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12', 'AOT', 'WVP', 'TCI_B', 'TCI_G', 'TCI_R']
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
s2L1C_xarray.to_netcdf("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset/val/s2L1C.nc")
print(f"s2l1c saved")
s2L2A_xarray.to_netcdf("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset/val/s2L2A.nc")
print(f"s2l2a saved")
s1_xarray.to_netcdf("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset/val/s1.nc")
print(f"s1 saved")
extra_xarray.to_netcdf("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/xarray_dataset/val/extra.nc")
print(f"extra saved")

print(f"all files saved")