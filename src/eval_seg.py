import os
import json
import random
from os.path import join, isfile
from typing import Union, Tuple, Dict, List
from glob import glob
import argparse
import yaml
import time
import shutil
import mlflow
import re

from PIL import Image
import cv2
print(cv2.__version__)
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset
import random
import xarray as xr
import lightning as L
import pandas as pd
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
from datetime import datetime
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics.segmentation import MeanIoU
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall, Dice
from lightning.pytorch.loggers import MLFlowLogger

## Mention self created Losses, Datasets and Architectures
import models
import dataset
from models import SegmentationModel
from loss_functions import DiceLoss

# Parse arguments
parser = argparse.ArgumentParser()
## This script requires the name of the config file as command line argument
parser.add_argument("-c", "--Config", help="Name of config file")
parser.add_argument("-r", "--MLFlowRunDir", help="Path to mlflow run directory")
args = parser.parse_args()

# Load configs from config file into a configs dictionary
config_dir = "./configs/"
config_file = args.Config
config_path = join(config_dir, config_file)
with open(config_path, 'r') as f:
    configs = yaml.safe_load(f)
    
# Load checkpoints from saved training run directory
run_dir = args.MLFlowRunDir

# Load best checkpoint path
ckpts = os.listdir(join(run_dir, "checkpoints"))

#print(f"Checkpoints found: {ckpts}")

#ckpts.remove("last.ckpt")
#pattern = r"val_loss=(\d+\.\d+).ckpt"
best_ckpt = ckpts[0]
print(f"Best checkpoint: {best_ckpt}")
best_ckpt_path = join(run_dir, "checkpoints", best_ckpt)
print(f"Best checkpoint path: {best_ckpt_path}")

# Load run_id of mlflow training run
meta_path = join(run_dir, "meta.yaml")
with open(meta_path, 'r') as f:
    meta_cfgs = yaml.load(f, Loader=yaml.FullLoader)
run_id = meta_cfgs['run_id']
experiment_id = meta_cfgs['experiment_id']
# Set up mlflow to resume training experiment to log 
save_dir = "/netscratch/rmukherjee/mlflow_logs/"
mlflow.set_tracking_uri(f"file:{save_dir}")
# Set current experiment_id and run_id
mlflow.set_experiment(experiment_id=experiment_id)

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

# Set up train/val/test dataset objects
train_dataset = dataset_mod(rotate_flag = True, label_dir=labels_train, s1_dir=s1_train, s2_dir=s2_train, landcover_dir=LC_train, azimuth_dir=azimuth_train, wateroccurence_dir=wateroccurence_train, DEM_dir=ele_train)
print(100*'-')
print(f"Number of data samples in training set = {len(train_dataset)}")
print(100*'-')
val_dataset = dataset_mod(rotate_flag = False, label_dir=labels_val, s1_dir=s1_train, s2_dir=s2_val, landcover_dir=LC_val, azimuth_dir=azimuth_val, wateroccurence_dir=wateroccurence_val, DEM_dir=ele_val)
print(f"Number of data samples in validation set = {len(val_dataset)}")
print(100*'-')
test_dataset = dataset_mod(rotate_flag = False, label_dir=labels_test, s1_dir=s1_train, s2_dir=s2_test, landcover_dir=LC_test, azimuth_dir=azimuth_test, wateroccurence_dir=wateroccurence_test, DEM_dir=ele_test)
print(f"Number of data samples in test set = {len(test_dataset)}")

# Get the training configurations from config
NUM_GPUS = configs['NUM_GPUS']
PRECISION = configs['PRECISION']
ACC_GRAD = configs['ACC_GRAD']
LEARNING_RATE = configs['LEARNING_RATE']
BATCH_SIZE = configs['BATCH_SIZE']
NUM_WORKERS = configs['NUM_WORKERS']
EPOCHS = configs['EPOCHS']
CRITERION = configs['CRITERION']

## Setup Loss function and other Metrics
if CRITERION == 'DiceLoss':
    criterion = DiceLoss()
elif CRITERION == 'CrossEntropyLoss':
    criterion = CrossEntropyLoss() 

##Setup architecture class
get_arch_name = configs['ARCH_NAME']
arch_module = getattr(models, get_arch_name)
model_arch = arch_module(
    #num_input_channels = num_input_channels
)

## Setup optimizer
optimizer = Adam(model_arch.parameters(),lr=LEARNING_RATE)

# Setup the DataModule using the three datasets
datamodule = dataset.Cloudsen12DataModule(
    train_dataset=train_dataset, 
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    batch_size=BATCH_SIZE, 
    num_workers=NUM_WORKERS,
)

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
    'MeanIOU': meaniou,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1Score': f1score}
metric_collection = MetricCollection(metrics_dict)

##Setup the model class
model_name = configs['MODEL_NAME']
## Load the model module based on the model name
model_module = getattr(models, model_name)
model = model_module(
    architecture=model_arch,
    criterion=criterion,
    optimizer=optimizer,
    metric_collection=metric_collection,
)
model.eval()
print(100*'-')
print(f"Model has been built:\n{model.model}")

checkpoint_cb = ModelCheckpoint(
    #dirpath= model_dirpath,
    save_last=True,
    save_top_k=1,
    monitor='val_loss',
    filename='{epoch}_{val_loss:.4f}',
)
earlystopping_cb = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    verbose= False,
    patience= EPOCHS//4, # no. of non-improving epochs to wait before stopping training
    check_finite= True, # stops training if NaN is encountered in loss
)
callbacks_list = [
    checkpoint_cb,
    earlystopping_cb,
]

## Setup Lightning trainer
trainer = L.Trainer(
    accumulate_grad_batches=ACC_GRAD,
    max_epochs=EPOCHS,
    accelerator='gpu', devices=NUM_GPUS,
    callbacks=callbacks_list,
    log_every_n_steps=100,
    enable_progress_bar=False,
    num_sanity_val_steps=0,
    precision=PRECISION,
)

# Evalute the model with the same mlflow run_id
with mlflow.start_run(run_id=run_id):
    # Save architecture as artifact
    model_str = str(model)
    with open(join(f"{run_dir}", "artifacts", "model_architecture.txt"), "w") as f:
        f.write(model_str)
        
    # Log config file as artifact
    mlflow.log_artifact(local_path=config_path)

    # Evaluate model on test set
    print(100*'-')
    print("Evaluating the trained model on test set...")
    test_metrics = trainer.test(
        model=model,
        ckpt_path=best_ckpt_path,
        dataloaders=datamodule.test_dataloader(),
    )
    mlflow.log_metrics(test_metrics[0])

    # Plot predictions on test set
    print(100*'-')
    print("Inference using trained model on test set...")
    predictions_test = trainer.predict(
        model=model, 
        ckpt_path=best_ckpt_path,
        dataloaders=datamodule.test_dataloader(),
    )
    
    print(100*'-')
    
    ## Establish Colour List
    colors_rgb = {
        0: [0, 230, 230],    # Clear - transparent blue
        1: [255, 255, 0],    # Thick Clouds - semi-transparent yellow
        2: [51, 255, 51],    # Thin Clouds - semi-transparent green
        3: [230, 0, 38]      # Cloud Shadow - semi-transparent red
    }
    
    local_save_dir = "/netscratch/rmukherjee/thesis/cloud-fusion-rushan-mukherjee/cloudseg/test_predictions"
    Path(local_save_dir).mkdir(parents=True, exist_ok=True)

    
    # Save test set predictions as artifacts
    metrics_list = []
    
    # Save predictions as images
    print("Saving test set predictions as images...")
    for batch_idx in range(len(predictions_test)):
        #print("for batch indx", batch_idx, "there are ",predictions_test[batch_idx]['cloud_labels'].shape[0], "samples")
    	
        for i in range(predictions_test[batch_idx].shape[0]):
            
            current_index = batch_idx * BATCH_SIZE + i
            #print("current index is ", current_index)

            ## Shape of pred_i is [4, 512, 512] 
            ## Pred_i is torch tensor
            pred_i = predictions_test[batch_idx][i]
            pred_i = pred_i.argmax(dim=0) ## Transforms into shape [512, 512]

            pred_i_torch = pred_i.type(torch.uint8)

            ## Shape of target_i is [512, 512]
            ## Target_i is numpy array
            target_i = test_dataset[current_index]['cloud_labels']
            target_i_torch = torch.tensor(target_i)
            target_i_torch = target_i_torch.type(torch.uint8)

            
            ## Make composite RGB Test Image
            ## Test Image is a numpy array
            test_image = test_dataset[current_index]['s2']
            rgb = np.stack((test_image[3], test_image[2], test_image[1]), axis=-1)

            ## Make RGB Composite Image
            rgb = ((rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb)) * 255).astype(np.uint8)
            rgb_tensor = torch.tensor(rgb)
            rgb_tensor = rgb_tensor.permute(2, 0, 1)
            #print(f"RGB Tensor shape: {rgb_tensor.shape}")
            #rgb = Image.fromarray(rgb)

            ## Make Overlay Image
            mask = np.zeros_like(rgb, dtype=np.uint8) #This creates a blank mask with same dimensions as the image
            
            for class_code, color in colors_rgb.items():

                binary_mask = (pred_i == class_code).numpy().astype(np.uint8)

                for j in range(3):
                    mask[:, :, j] += binary_mask * color[j]

            ##Blend the mask with the original image
            alpha = 0.3
            overlay = cv2.addWeighted(mask, alpha, rgb, 1 - alpha, 0)

            #print(f"Overlay shape: {overlay.shape}")
            overlay_img = Image.fromarray(overlay)
            rbg_img = Image.fromarray(rgb)
            ## Save RGB Image
            
            
            ## Save Overlay Image
            if current_index % 100 == 0:
                rbg_img.save(join(f"{local_save_dir}", "AMG_ad", f"test_batch_{batch_idx}_sample_{i}_overlay.png"))

            ##Calculate metrics for each sample
            metric_per_sample = metric_collection(pred_i_torch.unsqueeze(0), target_i_torch.unsqueeze(0))
            #print(metric_per_sample)
            
            metrics_list.append(metric_per_sample)
            #print(type(metric_per_sample))
            #print(f"Metrics for sample {i} in batch {batch_idx}: {metric_per_sample}")
        
    
   # assert(False)

    #metrics_df = pd.DataFrame(metrics_list)

    #metrics_df.to_csv(join(f"{local_save_dir}", "AMG_ad", "AMG_AD_testMetrics.csv"))

    
    #metrics_df.to_csv(join(local_save_dir, "S2_only_test_metrics.csv"))



