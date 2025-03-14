import numpy as np
import xarray as xr

landcover_train = xr.open_dataarray("/xarray/train/landcover.nc")
landcover_val = xr.open_dataarray("/xarray/val/landcover.nc")

landcover_max = 0
landcover_min = 0

for idx in landcover_train.values:
    for i in idx:
        
        max = np.max(i)
        if max > landcover_max:
            landcover_max = max
        
        min = np.min(i)
        if min < landcover_min:
            landcover_min = min

print(f"max train is {landcover_max}")
print(f"min train is {landcover_min}")

for idx in landcover_val.values:
    for i in idx:
        
        max = np.max(i)
        if max > landcover_max:
            landcover_max = max
        
        min = np.min(i)
        if min < landcover_min:
            landcover_min = min

print(f"max val is {landcover_max}")
print(f"min val is {landcover_min}")
