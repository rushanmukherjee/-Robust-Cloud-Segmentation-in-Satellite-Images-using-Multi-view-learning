import numpy as np
from torch.utils.data import Dataset
import xarray as xr
import pandas as pd
from pathlib import Path
import torch

train_shape = (8490, 512, 512)
train_labels = np.memmap('/data/train/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=train_shape)

val_shape = (535, 512, 512)
val_labels = np.memmap('/data/val/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=val_shape)

test_shape = (975, 512, 512)
test_labels = np.memmap('/data/test/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=test_shape)

xarray_train_labels= xr.DataArray(train_labels)
xarray_val_labels= xr.DataArray(val_labels)
xarray_test_labels= xr.DataArray(test_labels)

xarray_train_labels.to_netcdf("/xarray/train/HQlabels.nc")
xarray_val_labels.to_netcdf("/xarray/val/HQlabels.nc")
xarray_test_labels.to_netcdf("/xarray/test/HQlabels.nc")