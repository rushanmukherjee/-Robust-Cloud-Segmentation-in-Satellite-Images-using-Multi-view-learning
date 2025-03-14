import math
import numpy as np
from os.path import join
import random
from dateutil import parser
from datetime import datetime
import yaml
import argparse
from pathlib import Path

import dataset

# Get unique identifier for this script's execution
dt = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
# Temporary directory to store the figures before copying them to mlflow_logs as artifacts
unique_dir = f"{dt}_{random.randint(0,99)}"
fig_tmpdir = join("/exp/tmp_figs/", unique_dir)
Path(fig_tmpdir).mkdir(parents=True, exist_ok=True)

## Parse arguments
parser = argparse.ArgumentParser()
## Need to add config file as command line input
parser.add_argument("-c", "--Config", help="Name of config file")
args = parser.parse_args()

# Load configs from config file into a configs dictionary
config_dir = "./configs/"
config_file = args.Config
config_path = join(config_dir, config_file)
with open(config_path, 'r') as f:
    configs = yaml.safe_load(f)

## Defining the path to the dataset
train_dataset_dir = "/xarray/train/"
val_dataset_dir = "/xarray/val/"
test_dataset_dir = "/xarray/test/"

s2_train = join(train_dataset_dir, "s2L1C.nc")
s2_val = join(val_dataset_dir, "s2L1C.nc")
s2_test = join(test_dataset_dir, "s2L1C.nc")

s1_train = join(train_dataset_dir, "s1.nc")
s1_val = join(val_dataset_dir, "s1.nc")
s1_test = join(test_dataset_dir, "s1.nc")

labels_train = join(train_dataset_dir, "HQlabels.nc")
labels_val = join(val_dataset_dir, "HQlabels.nc")
labels_test = join(test_dataset_dir, "HQlabels.nc")

LC_train = join(train_dataset_dir, "landcover.nc")
LC_val = join(val_dataset_dir, "landcover.nc")
LC_test = join(test_dataset_dir, "landcover.nc")

ele_train = join(train_dataset_dir, "DEM.nc")
ele_val = join(val_dataset_dir, "DEM.nc")
ele_test = join(test_dataset_dir, "DEM.nc")

wateroccurence_train = join(train_dataset_dir, "wateroccurence.nc")
wateroccurence_val = join(val_dataset_dir, "wateroccurence.nc")
wateroccurence_test = join(test_dataset_dir, "wateroccurence.nc")

azimuth_train = join(train_dataset_dir, "azimuth.nc")
azimuth_val = join(val_dataset_dir, "azimuth.nc")
azimuth_test = join(test_dataset_dir, "azimuth.nc")


# Choose appropriate Dataset class based on config
dataset_name = configs['DATASET_NAME']
dataset_mod = getattr(dataset, dataset_name)

# Set up train/val/test dataset objects
train_dataset = dataset_mod(rotate_flag = True, label_dir=labels_train, s2_dir=s2_train, s1_dir=s1_train, landcover_dir=LC_train, 
                            azimuth_dir=azimuth_train, wateroccurence_dir=wateroccurence_train, DEM_dir=ele_train)
print(100*'-')
print(f"Number of data samples in training set = {len(train_dataset)}")
print(100*'-')
val_dataset = dataset_mod(rotate_flag = False, label_dir=labels_val, s2_dir=s2_val, s1_dir=s1_val, landcover_dir=LC_val, 
                          azimuth_dir=azimuth_val, wateroccurence_dir=wateroccurence_val, DEM_dir=ele_val)
print(f"Number of data samples in validation set = {len(val_dataset)}")
print(100*'-')
test_dataset = dataset_mod(rotate_flag = False, label_dir=labels_test, s2_dir=s2_test, s1_dir=s1_test, landcover_dir=LC_test,
                           azimuth_dir=azimuth_test, wateroccurence_dir=wateroccurence_test, DEM_dir=ele_test)
print(f"Number of data samples in test set = {len(test_dataset)}")
print(100*'-')

for item in test_dataset:
    for key in item:
        isnan = np.isnan(item[key])
        if isnan is True:
            print(f"Key: {key}, isnan: {isnan}")
