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

train_difficulty = {1.0:0, 2.0:0, 3.0:0, 4.0:0}
val_difficulty = {1.0:0, 2.0:0, 3.0:0, 4.0:0}
test_difficulty = {1.0:0, 2.0:0, 3.0:0, 4.0:0}

for index,row in df_train.iterrows():
    match row["difficulty"]:
        case 1.0: train_difficulty[1.0]+=1
        case 2.0: train_difficulty[2.0]+=1
        case 3.0: train_difficulty[3.0]+=1
        case 4.0: train_difficulty[4.0]+=1

for index,row in df_val.iterrows():
    match row["difficulty"]:
        case 1.0: val_difficulty[1.0]+=1
        case 2.0: val_difficulty[2.0]+=1
        case 3.0: val_difficulty[3.0]+=1
        case 4.0: val_difficulty[4.0]+=1

for index,row in df_test.iterrows():
    match row["difficulty"]:
        case 1.0: test_difficulty[1.0]+=1
        case 2.0: test_difficulty[2.0]+=1
        case 3.0: test_difficulty[3.0]+=1
        case 4.0: test_difficulty[4.0]+=1

##getting the sum of the dictionary values
sum_train = sum(train_difficulty.values())
sum_val = sum(val_difficulty.values())
sum_test = sum(test_difficulty.values())

##calculating relative frequencies of values
for key in train_difficulty:
    train_difficulty[key] = round((train_difficulty[key]/sum_train)*100,2)

for key in val_difficulty:
    val_difficulty[key] = round((val_difficulty[key]/sum_val)*100,2)

for key in test_difficulty:
    test_difficulty[key] = round((test_difficulty[key]/sum_test)*100,2)

print(f"Train Dataset Difficulty Counts: {train_difficulty.values()}")
print(f"Validation Dataset Counts: {val_difficulty.values()}")
print(f"Test Dataset Counts: {test_difficulty.values()}")

fig = plt.figure(figsize=(8,5))
sns.barplot(x=list(train_difficulty.keys()), y=list(train_difficulty.values()))
plt.title('Image Labelling Difficulty in Percentage(%)')
plt.xlabel('Ascending order of Difficulty')
plt.ylabel('Relative Percentage w.r.t Total Dataset')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/difficulty_plots/train_diff.png")
plt.close()

sns.barplot(x=list(val_difficulty.keys()), y=list(val_difficulty.values()))
plt.title('Image Labelling Difficulty in Percentage(%)')
plt.xlabel('Ascending order of Difficulty')
plt.ylabel('Relative Percentage w.r.t Total Dataset')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/difficulty_plots/val_diff.png")
plt.close()

sns.barplot(x=list(test_difficulty.keys()), y=list(test_difficulty.values()))
plt.title('Image Labelling Difficulty in Percentage(%)')
plt.xlabel('Ascending order of Difficulty')
plt.ylabel('Relative Percentage w.r.t Total Dataset')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/difficulty_plots/test_diff.png")
plt.close()
