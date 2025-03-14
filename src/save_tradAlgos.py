from torch.utils.data import Dataset
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np

train_shape = (8490, 512, 512)
val_shape = (535, 512, 512)
test_shape = (975, 512, 512)

# train_KappaMask_labels = np.memmap('/data/train/LABEL_kappamask_L1C.dat', dtype='int8', mode='r', shape=train_shape)
# val_KappaMask_labels = np.memmap('/data/val/LABEL_kappamask_L1C.dat', dtype='int8', mode='r', shape=val_shape)
# test_KappaMask_labels = np.memmap('/data/test/LABEL_kappamask_L1C.dat', dtype='int8', mode='r', shape=test_shape)

# KappaMask_train_labels= xr.DataArray(train_KappaMask_labels)
# KappaMask_val_labels= xr.DataArray(val_KappaMask_labels)
# KappaMask_test_labels= xr.DataArray(test_KappaMask_labels)

# KappaMask_train_labels.to_netcdf("/xarray/train/KappaMasklabels.nc")
# KappaMask_val_labels.to_netcdf("/xarray/val/KappaMasklabels.nc")
# KappaMask_test_labels.to_netcdf("/xarray/test/KappaMasklabels.nc")

train_Sen2Cor_labels = np.memmap('/data/train/LABEL_sen2cor.dat', dtype='int8', mode='r+', shape=train_shape)
val_Sen2Cor_labels = np.memmap('/data/val/LABEL_sen2cor.dat', dtype='int8', mode='r+', shape=val_shape)
test_Sen2Cor_labels = np.memmap('/data/test/LABEL_sen2cor.dat', dtype='int8', mode='r+', shape=test_shape)

Sen2Cor_train_labels= xr.DataArray(train_Sen2Cor_labels)
Sen2Cor_val_labels= xr.DataArray(val_Sen2Cor_labels)
Sen2Cor_test_labels= xr.DataArray(test_Sen2Cor_labels)

Sen2Cor_train_labels.to_netcdf("/xarray/train/Sen2Corlabels.nc")
Sen2Cor_val_labels.to_netcdf("/xarray/val/Sen2Corlabels.nc")
Sen2Cor_test_labels.to_netcdf("/xarray/test/Sen2Corlabels.nc")