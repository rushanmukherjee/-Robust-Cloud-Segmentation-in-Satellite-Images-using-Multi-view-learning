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
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

train_s2l1c = xr.open_dataset("/xarray/train/s2L1C.nc")

##import the labels for each set 
train_shape = (8490, 512, 512)
val_shape = (535, 512, 512)
test_shape = (975, 512, 512)

y_train = np.memmap('/data/train/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=train_shape)
y_val = np.memmap('/data/val/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=val_shape)
y_test = np.memmap('/data/test/LABEL_manual_hq.dat', dtype='int8', mode='r', shape=test_shape)

train_pixel_count = {0: 0, 1: 0, 2: 0, 3: 0}
val_pixel_count = {0: 0, 1: 0, 2: 0, 3: 0}
test_pixel_count = {0: 0, 1: 0, 2: 0, 3: 0}
y_train = y_train.tolist()
y_val = y_val.tolist()
y_test = y_test.tolist()

for idx in y_train:
    for arr in idx:
        count = Counter(arr)
        #train_pixel_count.update(count)
        train_pixel_count[0]+=count[0]
        train_pixel_count[1]+=count[1]
        train_pixel_count[2]+=count[2]
        train_pixel_count[3]+=count[3]

print("Training dataset Pixel Count")
print(train_pixel_count)
print(50*'-')

for idx in y_val:
    for arr in idx:
        count = Counter(arr)
        #train_pixel_count.update(count)
        val_pixel_count[0]+=count[0]
        val_pixel_count[1]+=count[1]
        val_pixel_count[2]+=count[2]
        val_pixel_count[3]+=count[3]

print("Validation dataset Pixel Count")
print(val_pixel_count)
print(50*'-')

for idx in y_test:
    for arr in idx:
        count = Counter(arr)
        #train_pixel_count.update(count)
        test_pixel_count[0]+=count[0]
        test_pixel_count[1]+=count[1]
        test_pixel_count[2]+=count[2]
        test_pixel_count[3]+=count[3]

print("Test dataset Pixel Count")
print(test_pixel_count)

sns.barplot(x=list(train_pixel_count.keys()), y=list(train_pixel_count.values()), color='violet')
plt.title('Pixel Count Per class in Training Data')
plt.xlabel('Class')
plt.ylabel('Number of Pixels')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/train_pixel_count.png")

sns.barplot(x=list(train_pixel_count.keys()), y=list(val_pixel_count.values()), color='peachpuff')
plt.title('Pixel Count Per class in Training Data')
plt.xlabel('Class')
plt.ylabel('Number of Pixels')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/val_pixel_count.png")

sns.barplot(x=list(train_pixel_count.keys()), y=list(test_pixel_count.values()), color='lawngreen')
plt.title('Pixel Count Per class in Training Data')
plt.xlabel('Class')
plt.ylabel('Number of Pixels')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/test_pixel_count.png")

plt.close()

