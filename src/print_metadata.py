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

print(df_test.to_string())