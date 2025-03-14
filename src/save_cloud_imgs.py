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
save_dir = Path("/xarray/train").mkdir(parents=True, exist_ok=True)

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

img1 = L1C_B1[0]
img2 = L1C_B2[0]
img3 = L1C_B3[0]
img4 = L1C_B4[0]
img5 = L1C_B5[0]
img6 = L1C_B6[0]
img7 = L1C_B7[0]
img8 = L1C_B8[0]
img8a = L1C_B8A[0]
img9 = L1C_B9[0]
img10 = L1C_B10[0]
img11 = L1C_B11[0]
img12 = L1C_B12[0]

## Save individual bands
# img = Image.fromarray(img2)
# img.save("/xarray/images/img2.png")

## Save composite RGB Image
# rgb = np.stack((img4, img3, img2), axis=-1)
# rgb = ((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb)) * 255).astype(np.uint8)
# rgb = Image.fromarray(rgb)

# rgb.save("/xarray/images/rgb.png")

## Sentinel 1 information being loaded
S1_angle = np.memmap('/data/train/S1_angle.dat', dtype='int16', mode='r', shape=train_shape)
S1_VH = np.memmap('/data/train/S1_VH.dat', dtype='int16', mode='r', shape=train_shape)
S1_VV = np.memmap('/data/train/S1_VV.dat', dtype='int16', mode='r', shape=train_shape)

s1img1 = S1_angle[0]
s1img2 = S1_VH[0]
s1img2 = s1img2 + 1e-10
s1img3 = S1_VV[0]

##Save 
s1_composite = np.stack((s1img2, s1img3, s1img1), axis=-1)
s1_composite = ((s1_composite - np.min(s1_composite)) / (np.max(s1_composite) - np.min(s1_composite)) * 255).astype(np.uint8)
s1_composite = Image.fromarray(s1_composite)
s1_composite.save("/xarray/images/s1_composite.png")

print("Images saved")