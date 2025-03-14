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

##S2 max and min 
GLOBAL_S2_MAX = 28004.
GLOBAL_S2_MIN = 0

##S1 max and min
GLOBAL_S1_MAX = 32767
GLOBAL_S1_MIN = -32768

##Extra max and min
GLOBAL_EXTRA_MAX = 24213
GLOBAL_EXTRA_MIN = -71

## Define min-max scaling parameters
RANGE_MIN = -1
RANGE_MAX = 1


##Dataset class for importing the cloudsen12 dataset from cluster
class cloudsenDataset3modality(Dataset):

    def __init__(self, label_dir, s1_dir, s2_dir, extra_dir) -> None:
        
        super(cloudsenDataset3modality, self).__init__()
        self.labels = xr.open_dataarray(label_dir)
        self.s1 = xr.open_dataarray(s1_dir)
        self.s2 = xr.open_dataarray(s2_dir)
        self.extra = xr.open_dataarray(extra_dir)

    def __getitem__(self, index) -> Tuple[Dict['str', torch.Tensor], Dict['str', torch.Tensor]]:
        
        
        sen1 = self.s1[index].values
        sen2 = self.s2[index].values
        extra = self.extra[index].values
        labels = self.labels[index].values

        sen1 = sen1.astype(dtype='float32')
        sen2 = sen2.astype(dtype='float32')
        extra = extra.astype(dtype='float32')
        #labels = labels.astype(dtype='float32')

        ##Scaling Sentinel 2 values to range [-1,1]
        sen2_std = (sen2 - GLOBAL_S2_MIN) / (GLOBAL_S2_MAX - GLOBAL_S2_MIN)
        sen2_scaled = sen2_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

        ##Scaling Sentinel 1 values to range [-1,1]
        sen1_std = (sen1 - GLOBAL_S1_MIN) / (GLOBAL_S1_MAX - GLOBAL_S1_MIN)
        sen1_scaled = sen1_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

        ##Scaling Extra values to range [-1,1]
        extra_std = (extra - GLOBAL_EXTRA_MIN) / (GLOBAL_EXTRA_MAX - GLOBAL_EXTRA_MIN)
        extra_scaled = extra_std * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

        # Assemble dicts to return inputs and targets
        inputs = {
            's1': sen1_scaled, 
            's2': sen2_scaled,
            'extra': extra_scaled 
        }
        targets = {
            'cloud_labels': labels
        }
        return inputs, targets
    
    def __len__(self) -> int:

        ##To return the number of samples in dataset
        label_vals = self.labels['dim_0'].values

        return label_vals.size
    
    def load_s1(self,idx):

        sen1 = self.s1
        return sen1