import os
import json
import random
from os.path import join, isfile
from typing import Union, Tuple, Dict, List
from glob import glob
import argparse
import yaml
import time

import torch
import numpy as np
from torch.utils.data import Dataset
import random
import xarray as xr
import lightning as L
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from datetime import datetime

## Defining the path to the dataset
train_dataset_dir = "/xarray/train/"
val_dataset_dir = "/xarray/val/"
test_dataset_dir = "/xarray/test/"

labels_train = join(train_dataset_dir, "HQlabels.nc")
labels_val = join(val_dataset_dir, "HQlabels.nc")
labels_test = join(test_dataset_dir, "HQlabels.nc")

label_vals = xr.open_dataarray(labels_train)

idx = label_vals[1].values
print(idx.shape)