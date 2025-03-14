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
import cv2
from PIL import Image

train_shape = (8490, 512, 512)
val_shape = (535, 512, 512)
test_shape = (975, 512, 512)
save_dir = Path("/xarray/train").mkdir(parents=True, exist_ok=True)

#ex_landcover = np.memmap('/data/val/EXTRA_lc10.dat', dtype='int16', mode='r', shape=val_shape)
ex_landcover = np.memmap('/data/test/EXTRA_lc10.dat', dtype='int16', mode='r', shape=test_shape)

# print(np.matrix(ex_landcover[12][508]))
# print(np.matrix(ex_landcover[12][509]))
# print(np.matrix(ex_landcover[12][510]))
# print(np.matrix(ex_landcover[12][511]))
#print(ex_landcover.shape)

depth, height, width = ex_landcover.shape
#print(depth, height, width)

one_hot = np.zeros((height, width, 11), dtype=np.uint8)

final_one_hot = np.array([])
count = 0

for arr in ex_landcover:
    #print(f"arr shape is {arr.shape}")
    for i in range(11):
        one_hot[:, :, i] = (arr == i).astype(np.uint8)
    
    expanded_one_hot = np.expand_dims(one_hot, axis=0)
    if final_one_hot.size == 0:
        final_one_hot = expanded_one_hot
    else:
        final_one_hot = np.concatenate((final_one_hot, expanded_one_hot), axis=0)
    print(f"final_one_hot shape is {final_one_hot.shape}")
    print(f"count is {count}")
    count+=1

print(f"count is {count}")
print(final_one_hot.shape)

final_one_hot = xr.DataArray(final_one_hot, dims=["depth", "height", "width", "channels"])
final_one_hot.to_netcdf("/xarray/test/landcover_onehot.nc")

print("Images saved")