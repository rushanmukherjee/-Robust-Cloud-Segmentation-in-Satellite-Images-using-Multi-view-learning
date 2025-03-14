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
from dateutil import parser
from datetime import datetime

df_train = pd.read_csv('/data/train/metadata.csv')
df_val = pd.read_csv('/data/val/metadata.csv')
df_test = pd.read_csv('/data/test/metadata.csv')

train_img_classes = {"cloudy":0, "cloud-free":0, "low-cloudy":0, "mid-cloudy":0, "almost-clear":0}
val_img_classes = {"cloudy":0, "cloud-free":0, "low-cloudy":0, "mid-cloudy":0, "almost-clear":0}
test_img_classes = {"cloudy":0, "cloud-free":0, "low-cloudy":0, "mid-cloudy":0, "almost-clear":0}

for index,row in df_train.iterrows():
    match row["cloud_coverage"]:
        case "cloudy": train_img_classes["cloudy"]+=1
        case "cloud-free": train_img_classes["cloud-free"]+=1
        case "low-cloudy": train_img_classes["low-cloudy"]+=1
        case "mid-cloudy": train_img_classes["mid-cloudy"]+=1
        case "almost-clear": train_img_classes["almost-clear"]+=1

for index,row in df_val.iterrows():
    match row["cloud_coverage"]:
        case "cloudy": val_img_classes["cloudy"]+=1
        case "cloud-free": val_img_classes["cloud-free"]+=1
        case "low-cloudy": val_img_classes["low-cloudy"]+=1
        case "mid-cloudy": val_img_classes["mid-cloudy"]+=1
        case "almost-clear": val_img_classes["almost-clear"]+=1

for index,row in df_test.iterrows():
    match row["cloud_coverage"]:
        case "cloudy": test_img_classes["cloudy"]+=1
        case "cloud-free": test_img_classes["cloud-free"]+=1
        case "low-cloudy": test_img_classes["low-cloudy"]+=1
        case "mid-cloudy": test_img_classes["mid-cloudy"]+=1
        case "almost-clear": test_img_classes["almost-clear"]+=1

print(f"Distribution of Labels in Training Images: {train_img_classes}")
print(f"Distribution of Labels in Training Images: {val_img_classes}")
print(f"Distribution of Labels in Training Images: {test_img_classes}")