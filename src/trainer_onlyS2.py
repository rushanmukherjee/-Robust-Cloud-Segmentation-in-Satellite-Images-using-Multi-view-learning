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

import torch
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

# Choose appropriate Dataset class based on config
dataset_name = configs['DATASET_NAME']
dataset_mod = getattr(dataset, dataset_name)

# Set up train/val/test dataset objects
train_dataset = dataset_mod(rotate_flag=True, label_dir=labels_train, s2_dir=s2_train)
print(100*'-')
print(f"Number of data samples in training set = {len(train_dataset)}")
print(100*'-')
val_dataset = dataset_mod(rotate_flag=False, label_dir=labels_val, s2_dir=s2_val)
print(f"Number of data samples in validation set = {len(val_dataset)}")
print(100*'-')
test_dataset = dataset_mod(rotate_flag=False, label_dir=labels_test, s2_dir=s2_test)
print(f"Number of data samples in test set = {len(test_dataset)}")

# Get the training configurations from config
NUM_GPUS = configs['NUM_GPUS']
PRECISION = configs['PRECISION']
ACC_GRAD = configs['ACC_GRAD']
LEARNING_RATE = configs['LEARNING_RATE']
BATCH_SIZE = configs['BATCH_SIZE']
NUM_WORKERS = configs['NUM_WORKERS']
WEIGHT_DECAY = configs['WEIGHT_DECAY']
CRITERION = configs['CRITERION']

## Setup Loss function and other Metrics
if CRITERION == 'CrossEntropyLoss':
    criterion = CrossEntropyLoss()
elif CRITERION == 'DiceLoss':
    criterion = DiceLoss()

##Setup architecture class
get_arch_name = configs['ARCH_NAME']
arch_module = getattr(models, get_arch_name)
model_arch = arch_module(
    #num_input_channels = num_input_channels
)

## Setup optimizer
optimizer = Adam(model_arch.parameters(),lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
meaniou = MeanIoU(num_classes = 4, include_background = False) ##Also check for include background set as True
## Combine all metrics in a list to be put inside lightning module
metrics_list = [meaniou,accuracy,precision,recall,f1score]#,dice] 
metric_collection = MetricCollection(metrics_list)

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
print(100*'-')
print(f"Model has been built:\n{model.model}")

## _ architecture as artifact
model_str = str(model)
with open(join(fig_tmpdir, "model_architecture.txt"), "w") as f:
    f.write(model_str)

# Setup the logger
save_dir = "/mlflow_logs/"
experiment_name = f"{model_name}"
Path(save_dir).mkdir(parents=True, exist_ok=True)
mlf_logger = MLFlowLogger(
    experiment_name=experiment_name,
    save_dir=save_dir,
    tracking_uri=f"file:{save_dir}"
)
model_dirpath = join(save_dir, "checkpoints")
Path(model_dirpath).mkdir(parents=True, exist_ok=True)
# Define the callbacks during training
EPOCHS = configs['EPOCHS']
checkpoint_cb = ModelCheckpoint(
    save_last=False,
    save_top_k=1,
    monitor='val_loss',
    filename='{epoch}_{val_loss:.4f}',
)
earlystopping_cb = EarlyStopping(
    monitor= 'val_loss',
    mode= 'min',
    verbose= True,
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
    logger=mlf_logger,
    callbacks=callbacks_list,
    log_every_n_steps=100,
    enable_progress_bar=False,
    num_sanity_val_steps=0,
    precision=PRECISION,
)

# Save hyperparameters to mlflow
hyperparameters = dict(
    max_epochs = EPOCHS,
    learning_rate = LEARNING_RATE,
    batch_size = BATCH_SIZE,
    effective_batch_size = BATCH_SIZE*ACC_GRAD,
    arch_name = get_arch_name,
    model_name = model_name,
    weight_decay = WEIGHT_DECAY
)
trainer.logger.log_hyperparams(hyperparameters)

#TODO program model training and plotting
## Training the model 

# Train the model
print(100*'-')
print("Training the model")

start_time = time.time()
trainer.fit(
    model=model, 
    train_dataloaders=datamodule.train_dataloader(), 
    val_dataloaders=datamodule.val_dataloader()
)
training_time = time.time() - start_time
print(f'Time taken for training = {training_time} seconds...')
trainer.logger.log_metrics({"training_time": training_time})

# Save figures to mlruns run directory
mlf_experiment = mlf_logger.experiment
mlf_experiment.log_artifacts(run_id=mlf_logger._run_id, local_dir=fig_tmpdir)

# Save config file as an mlflow artifact
mlf_experiment.log_artifact(run_id=mlf_logger._run_id, local_path=config_path)

# Delete the temporary directory
shutil.rmtree(fig_tmpdir)