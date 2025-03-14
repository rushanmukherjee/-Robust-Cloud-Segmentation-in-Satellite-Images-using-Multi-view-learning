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
from collections import Counter

df_train = pd.read_csv('/data/train/metadata.csv')
df_val = pd.read_csv('/data/val/metadata.csv')
df_test = pd.read_csv('/data/test/metadata.csv')

anno_list_train = []
anno_list_val = []
anno_list_test = []

for index,row in df_train.iterrows():
    name = row['annotator_name']
    anno_list_train.append(name)

for index,row in df_val.iterrows():
    name = row['annotator_name']
    anno_list_val.append(name)

for index,row in df_test.iterrows():
    name = row['annotator_name']
    anno_list_test.append(name)

##Count the number of occurences of each annotator and put them in a dictionary
anno_dict_train = dict(Counter(anno_list_train))
anno_dict_val = dict(Counter(anno_list_val))
anno_dict_test = dict(Counter(anno_list_test))

# ##Sort these dictionaries alphabetically
# anno_train_sorted = dict(sorted(anno_dict_train.items()))
# anno_val_sorted = dict(sorted(anno_dict_val.items()))
# anno_test_sorted = dict(sorted(anno_dict_test.items()))

print(anno_dict_train)
print(anno_dict_val)
print(anno_dict_test)

# fig = plt.figure()
# ax = fig.add_subfigure(111)
# sub1 = ax.bar()

fig = plt.figure(figsize=(15,7))
sns.barplot(x=list(anno_dict_train.keys()), y=list(anno_dict_train.values()), align='edge', width=0.3)
plt.xticks(rotation='vertical')
plt.title('Images labelled by each Annotator')
plt.xlabel('Annotator Names')
plt.ylabel('Image Count')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/annotator_plots/anno_train.png")
plt.close()

fig = plt.figure(figsize=(15,7))
sns.barplot(x=list(anno_dict_val.keys()), y=list(anno_dict_val.values()), align='edge', width=0.3)
plt.xticks(rotation='vertical')
plt.title('Images labelled by each Annotator')
plt.xlabel('Annotator Names')
plt.ylabel('Image Count')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/annotator_plots/anno_val.png")
plt.close()

fig = plt.figure(figsize=(15,7))
sns.barplot(x=list(anno_dict_test.keys()), y=list(anno_dict_test.values()), align='edge', width=0.3)
plt.xticks(rotation='vertical')
plt.title('Images labelled by each Annotator')
plt.xlabel('Annotator Names')
plt.ylabel('Image Count')
plt.savefig("/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/plots/annotator_plots/anno_test.png")
plt.close()