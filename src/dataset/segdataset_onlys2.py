import os
import json
import random
from os.path import join, isfile
from typing import Union, Tuple, Dict, List
from glob import glob

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import Dataset
import random
import xarray as xr
from PIL import Image

GLOBAL_S2_MAX = 28004.
GLOBAL_S2_MIN = 0
## Define min-max scaling parameters
RANGE_MIN = -1
RANGE_MAX = 1


##Dataset class for importing the cloudsen12 dataset from cluster
class cloudsenDataset_onlys2(Dataset):

    def __init__(self, rotate_flag, label_dir, s2_dir) -> None:
        
        super(cloudsenDataset_onlys2, self).__init__()
        self.rotate_flag = rotate_flag
        self.labels = xr.open_dataarray(label_dir)
        self.s2 = xr.open_dataarray(s2_dir)
        self.rotation_angles = [0, 90, 180, 270] ## change it to random choice

    def __getitem__(self, index) -> Dict['str', torch.Tensor]:
        
        sen2 = self.s2[index].values
        labels = self.labels[index].values

        ##Converting int16 to float32 for ML pipeline
        sen2 = sen2.astype(dtype='float32')

        ##Normalizing the data
        sen2_std = (sen2 - GLOBAL_S2_MIN) / (GLOBAL_S2_MAX - GLOBAL_S2_MIN)
        sen2_scaled = sen2_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN
        
        #print(f"stride is {sen2_scaled.strides}")
        #copied_sen2 = np.copy(sen2_scaled)

        if self.rotate_flag == True:
            rand_angle = random.choice(self.rotation_angles)
            if rand_angle==0:
                rotated_sen2 = sen2_scaled.copy()
                rotated_label = labels.copy()

            elif rand_angle==90:
                rotated_sen2 = np.rot90(sen2_scaled, axes = (1,2)).copy()
                rotated_label = np.rot90(labels).copy()

            elif rand_angle==180:
                rotated_sen2 = np.rot90(sen2_scaled, 2, axes = (1,2)).copy()
                rotated_label = np.rot90(labels, 2).copy()

            else:
                rotated_sen2 = np.rot90(sen2_scaled, 3, axes = (1,2)).copy()
                rotated_label = np.rot90(labels, 3).copy()
        else:
            rotated_sen2 = sen2_scaled.copy()
            rotated_label = labels.copy()
        
        #print(f"rotated stride is {rotated_sen2.strides}")

        # Assemble dicts to return inputs and targets
        items = {
            's2': rotated_sen2,
            'cloud_labels': rotated_label
        }

        return items
    
    def __len__(self) -> int:

        ##To return the number of samples in dataset
        label_vals = self.labels['dim_0'].values
        true_size = label_vals.size
        return true_size
    