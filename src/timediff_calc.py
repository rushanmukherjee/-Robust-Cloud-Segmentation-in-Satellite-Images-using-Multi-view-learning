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

#ds = xr.open_dataset("xarray_dataset/train/s2L1C.nc")

test_shape = (975, 512, 512)
#metadata = np.memmap('/data/test/metadata.csv', dtype='int16', mode='r', shape=test_shape)

df_train = pd.read_csv('/data/train/metadata.csv')
df_val = pd.read_csv('/data/val/metadata.csv')
df_test = pd.read_csv('/data/test/metadata.csv')
#print(df.to_string())

diff_train = []
diff_val = []
diff_test = []

date_format = '%Y-%m-%d'

for index,row in df_train.iterrows():
    
    ## Extract only s1 and s2 date information from metadata
    s2_row = row['s2_date'].split("T")[0]
    s1_row = row['s1_date'].split("T")[0]
    
    ## Convert date in string to datetime object
    s2_date = datetime.strptime(s2_row, date_format)
    s1_date = datetime.strptime(s1_row, date_format)

    ## Find the difference in dates of capture between s1 and s2 image
    day_diff = str(s2_date-s1_date)
    day_diff = int(day_diff.split(" ")[0])
    diff_train.append(day_diff)

for index,row in df_val.iterrows():
    
    ## Extract only s1 and s2 date information from metadata
    s2_row = row['s2_date'].split("T")[0]
    s1_row = row['s1_date'].split("T")[0]
    
    ## Convert date in string to datetime object
    s2_date = datetime.strptime(s2_row, date_format)
    s1_date = datetime.strptime(s1_row, date_format)

    ## Find the difference in dates of capture between s1 and s2 image
    day_diff = str(s2_date-s1_date)
    day_diff = int(day_diff.split(" ")[0])
    diff_val.append(day_diff)

for index,row in df_test.iterrows():
    
    ## Extract only s1 and s2 date information from metadata
    s2_row = row['s2_date'].split("T")[0]
    s1_row = row['s1_date'].split("T")[0]
    
    ## Convert date in string to datetime object
    s2_date = datetime.strptime(s2_row, date_format)
    s1_date = datetime.strptime(s1_row, date_format)

    ## Find the difference in dates of capture between s1 and s2 image
    day_diff = str(s2_date-s1_date)
    day_diff = int(day_diff.split(" ")[0])
    diff_test.append(day_diff)

# print(diff_train.size)
# print(diff_val.size)
# print(diff_test.size)

plot_data = {
    'Train': diff_train,
    'Validation': diff_val,
    'Test': diff_test
}


fig = plt.figure(figsize =(10, 7))
## Plotting a Boxplot
sns.boxplot(data=plot_data) 
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/time_diff_all.png")
plt.close()

## Plotting a histogram of Time Delay
fig = plt.figure(figsize=(11,7))
sns.histplot(data=diff_train)
plt.title('Time difference in days between S2 and S1 image capture')
plt.xlabel('Difference in Days')
plt.ylabel('Number of Images')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/time_diff_train.png")
plt.close()

fig = plt.figure(figsize=(11,7))
sns.histplot(data=diff_val)
plt.title('Time difference in days between S2 and S1 image capture')
plt.xlabel('Difference in Days')
plt.ylabel('Number of Images')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/time_diff_val.png")
plt.close()

fig = plt.figure(figsize=(11,7))
sns.histplot(data=diff_test)
plt.title('Time difference in days between S2 and S1 image capture')
plt.xlabel('Difference in Days')
plt.ylabel('Number of Images')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/time_diff_test.png")
plt.close()


