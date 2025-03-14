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


##S1 max and min
GLOBAL_S1_MAX = 32767
GLOBAL_S1_MIN = -32768

## Define min-max scaling parameters
RANGE_MIN = -1
RANGE_MAX = 1


##Dataset class for importing the cloudsen12 dataset from cluster
class cloudsenDataset_onlys1(Dataset):

    def __init__(self, rotate_flag, label_dir, s1_dir) -> None:
        
        super(cloudsenDataset_onlys1, self).__init__()
        self.rotate_flag = rotate_flag
        self.labels = xr.open_dataarray(label_dir)
        self.s1 = xr.open_dataarray(s1_dir)
        self.rotation_angles = [0, 90, 180, 270]

    def __getitem__(self, index) -> Tuple[Dict['str', torch.Tensor], Dict['str', torch.Tensor]]:
        
        sen1 = self.s1[index].values
        labels = self.labels[index].values
        #print(f"seg dataset label datatype is {type(labels)}")

        ##Converting int16 to float32 for ML pipeline
        sen1 = sen1.astype(dtype='float32')

        sen1_std = (sen1 - GLOBAL_S1_MIN) / (GLOBAL_S1_MAX - GLOBAL_S1_MIN)
        sen1_scaled = sen1_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

        if self.rotate_flag == True:
            rand_angle = random.choice(self.rotation_angles)
            if rand_angle==0:
                rotated_sen1 = sen1_scaled.copy()
                rotated_label = labels.copy()

            elif rand_angle==90:
                rotated_sen1 = np.rot90(sen1_scaled, axes = (1,2)).copy()
                rotated_label = np.rot90(labels).copy()

            elif rand_angle==180:
                rotated_sen1 = np.rot90(sen1_scaled, 2, axes = (1,2)).copy()
                rotated_label = np.rot90(labels, 2).copy()

            else:
                rotated_sen1 = np.rot90(sen1_scaled, 3, axes = (1,2)).copy()
                rotated_label = np.rot90(labels, 3).copy()
        else:
            rotated_sen1 = sen1_scaled.copy()
            rotated_label = labels.copy()
        # Assemble dicts to return inputs and targets
        items = {
            's1': rotated_sen1,
            'cloud_labels': rotated_label
        }
        
        return items
    
    def __len__(self) -> int:

        ##To return the number of samples in dataset
        label_vals = self.labels['dim_0'].values
        return label_vals.size
    