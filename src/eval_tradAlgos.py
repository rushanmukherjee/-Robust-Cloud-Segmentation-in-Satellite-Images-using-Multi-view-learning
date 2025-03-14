from os.path import join
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import torch
from torchmetrics.segmentation import MeanIoU
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall

## Classification Specific Metrics
accuracy = MulticlassAccuracy(num_classes = 4)
precision = MulticlassPrecision(num_classes = 4)
recall = MulticlassRecall(num_classes = 4)
f1score = MulticlassF1Score(num_classes = 4)
#dice = Dice(num_classes = 4)
## Segmentation Specific Metric
meaniou = MeanIoU(num_classes = 4, include_background = True) ##Also check for include background set as True
## Combine all metrics in a list to be put inside lightning module

metrics_dict = {
    #'MeanIOU': meaniou,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1Score': f1score}
metric_collection = MetricCollection(metrics_dict)

## Defining the path to the dataset
train_dataset_dir = "/xarray/train/"
val_dataset_dir = "/xarray/val/"
test_dataset_dir = "/xarray/test/"

labels_test = join(test_dataset_dir, "HQlabels.nc")
label_xr = xr.open_dataarray(labels_test)
label = label_xr.values
print(f"Shape of the label: {label.shape}")

KappaMask_test = join(test_dataset_dir, "KappaMasklabels.nc")
KappaMask_xr = xr.open_dataarray(KappaMask_test)
KappaMask = KappaMask_xr.values
print(f"Shape of the KappaMask: {KappaMask.shape}")

kp_clear = (KappaMask == 1)
kp_thin = (KappaMask == 3)
kp_thick = (KappaMask == 4)
kp_shadow = (KappaMask == 2)

KappaMask[kp_clear] = 0
KappaMask[kp_thin] = 2
KappaMask[kp_thick] = 1
KappaMask[kp_shadow] = 3
KappaMask[KappaMask == 5] = 0

Sen2Cor_test = join(test_dataset_dir, "Sen2Corlabels.nc")
Sen2Cor_xr = xr.open_dataarray(Sen2Cor_test)
Sen2Cor = Sen2Cor_xr.values
print(f"Shape of the Sen2Cor: {Sen2Cor.shape}")

thin_mask = (Sen2Cor == 10)
thick_mask = (Sen2Cor == 8) | (Sen2Cor == 9)
clear_mask = (Sen2Cor == 4) | (Sen2Cor == 2) | (Sen2Cor == 5) | (Sen2Cor == 6) | (Sen2Cor == 11)

Sen2Cor[thick_mask] = 1
Sen2Cor[clear_mask] = 0
Sen2Cor[thin_mask] = 2
Sen2Cor[Sen2Cor == 7] = 0

#len = labels_test['dim_0'].values.size
#print(f"length of dataset: {len}")

local_save_dir = "/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/test_predictions"
Path(local_save_dir).mkdir(parents=True, exist_ok=True)

# Save test set predictions as artifacts
KappaMask_metrics_list = []

##Save metrics for KappaMask
print("Saving metrics for KappaMask")
for index in range(len(label)): 

    HQLabel = label[index]
    KappaMaskLabel = KappaMask[index]

    hqlabel_torch = torch.tensor(HQLabel)
    hqlabel_torch = hqlabel_torch.type(torch.uint8)

    kappamask_torch = torch.tensor(KappaMaskLabel)
    kappamask_torch = kappamask_torch.type(torch.uint8)
    
    #print(f" HQ Label {HQLabel}")
    #print(f" Kappamask label {KappaMaskLabel[2]}")	
    #print(f" sen2cor label {Sen2Cor[index]}")

    metric_per_sample = metric_collection(kappamask_torch, hqlabel_torch)
    KappaMask_metrics_list.append(metric_per_sample)

metrics_df = pd.DataFrame(KappaMask_metrics_list)

metrics_df.to_csv(join(f"{local_save_dir}", "KappaMask_metrics.csv"))



##Save metrics for Sen2Cor
print("Saving metrics for Sen2Cor")
Sen2Cor_metrics_list = []
for index in range(len(label)): 

    HQLabel = label[index]
    Sen2CorLabel = Sen2Cor[index]

    hqlabel_torch = torch.tensor(HQLabel)
    hqlabel_torch = hqlabel_torch.type(torch.uint8)

    sen2cor_torch = torch.tensor(Sen2CorLabel)
    sen2cor_torch = sen2cor_torch.type(torch.uint8)
    
    #print(f" HQ Label {HQLabel}")
    #print(f" Kappamask label {KappaMaskLabel[2]}")	
    #print(f" sen2cor label {Sen2Cor[index]}")

    metric_per_sample = metric_collection(sen2cor_torch, hqlabel_torch)
    Sen2Cor_metrics_list.append(metric_per_sample)

sen2cormetrics_df = pd.DataFrame(Sen2Cor_metrics_list)

sen2cormetrics_df.to_csv(join(f"{local_save_dir}", "Sen2Cor_metrics.csv"))

