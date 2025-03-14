from typing import List,Dict
from os.path import join
import dataset
import yaml
import argparse

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

labels_train = join(train_dataset_dir, "HQlabels.nc")
labels_val = join(val_dataset_dir, "HQlabels.nc")
labels_test = join(test_dataset_dir, "HQlabels.nc")

# Choose appropriate Dataset class based on config
dataset_name = configs['DATASET_NAME']
dataset_mod = getattr(dataset, dataset_name)

drop_band_flag = configs['DROP_BAND_FLAG']
drop_band_count = configs['DROP_BAND_COUNT']

# Set up train/val/test dataset objects
test_dataset = dataset_mod(drop_band_flag = drop_band_flag, drop_band_count = drop_band_count, rotate_flag = False, label_dir=labels_test, s1_dir=s1_train, s2_dir=s2_test, landcover_dir=LC_test, azimuth_dir=azimuth_test, wateroccurence_dir=wateroccurence_test, DEM_dir=ele_test)
print(f"Number of data samples in test set = {len(test_dataset)}")

