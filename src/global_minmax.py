from models import EdgeVit
import torch
import numpy as np
import xarray as xr
import os

s2_train = xr.open_dataarray("/xarray/train/s2L1C.nc")
label_train = xr.open_dataarray("/xarray/train/HQlabels.nc")
s2_val = xr.open_dataarray("/xarray/val/s2L1C.nc")

# long_label = label_train
# long_label = torch.tensor(long_label)
# long_label = long_label.long()
# print(long_label)

s1_train = xr.open_dataarray("/xarray/train/s1.nc")
s1_val = xr.open_dataarray("/xarray/val/s1.nc")

extra_train = xr.open_dataarray("/xarray/train/extra.nc")
extra_val = xr.open_dataarray("/xarray/val/extra.nc")

extra_max = 0
extra_min = 0

for idx in extra_train.values:
    for i in idx:
        
        max = np.max(i)
        if max > extra_max:
            extra_max = max
        
        min = np.min(i)
        if min < extra_min:
            extra_min = min

print(f"max train is {extra_max}")
print(f"min train is {extra_min}")

for idx in extra_val.values:
    for i in idx:
        
        max = np.max(i)
        if max > extra_max:
            extra_max = max
        
        min = np.min(i)
        if min < extra_min:
            extra_min = min

print(f"max val is {extra_max}")
print(f"min val is {extra_min}")


#print(s2_train)
# print(type(s2_train))
# print(type(label_train))

# len = s2_train['size'].sizes
# xarray_size = s2_train['size'].values
# num_elements = xarray_size.size

# labels = label_train['dim_0'].sizes
# labels_vals = label_train['dim_0'].values
# print(num_elements)
# print(type(num_elements))
# print(labels_vals)
# print(type(labels_vals))
# print(labels_vals.size)
# float_array = s2_train[0][0].values
# float_array = float_array.astype(dtype='float32')
# print(float_array)
# print(type(float_array))

# s2_train_height = s2_train['height'].values
# s2_train_width = s2_train['width'].values

# max_height = np.max(float_array)
# max_width = np.max(s2_train_width)
# print(max_height)

# s2_max = 0
# s2_min = 0

# for idx in s2_train.values:
#     for i in idx:
        
#         max = np.max(i)
#         if max > s2_max:
#             s2_max = max
        
#         min = np.min(i)
#         if min < s2_min:
#             s2_min = min

#     print(f"max val is {s2_max}")
#     print(f"min val is {s2_min}")
    
# for idx in s2_val.values:
#     for i in idx:
        
#         max = np.max(i)
#         if max > s2_max:
#             s2_max = max
        
#         min = np.min(i)
#         if min < s2_min:
#             s2_min = min

#     print(f"max val is {s2_max}")
#     print(f"min val is {s2_min}")

# s1_max = 0
# s1_min = 0

# for idx in s1_train.values:
#     for i in idx:
        
#         max = np.max(i)
#         if max > s1_max:
#             s1_max = max
        
#         min = np.min(i)
#         if min < s1_min:
#             s1_min = min

# print(f"max train is {s1_max}")
# print(f"min train is {s1_min}")
    
# for idx in s1_val.values:
#     for i in idx:
        
#         max = np.max(i)
#         if max > s1_max:
#             s1_max = max
        
#         min = np.min(i)
#         if min < s1_min:
#             s1_min = min

# print(f"max val is {s1_max}")
# print(f"min val is {s1_min}")