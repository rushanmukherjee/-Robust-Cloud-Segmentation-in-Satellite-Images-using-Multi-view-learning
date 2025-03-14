from typing import Optional, Union, Dict, Tuple
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import Module
import lightning as L
import numpy as np

## Pytorch Lightning Model Class used to configure the model for training and inference
class MultiInput_SegmentationModel(L.LightningModule):
    def __init__(
            self, 
            architecture: Module,
            criterion,
            optimizer: torch.optim.Optimizer,
            metric_collection: torchmetrics.MetricCollection,

        ) -> None:
        super(MultiInput_SegmentationModel, self).__init__()
        self.save_hyperparameters(ignore=["architecture", "metric_collection"])
        
        self.model = architecture
        self.optimizer = optimizer
        
        # Loss functions
        self.criterion = criterion

        # Other metrics to be tracked on train, val and test sets
        self.train_metrics = metric_collection.clone(prefix="train_")
        self.val_metrics = metric_collection.clone(prefix='val_')
        self.test_metrics = metric_collection.clone(prefix='test_')
        # Lists for storing and computing losses
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def training_step(self, batch, batch_idx):
        x1 = batch['s1']
        x2 = batch['s2']
        x3 = batch['landcover']
        x4 = batch['azimuth']
        x5 = batch['wateroccurence']
        x6 = batch['elevation']

        y_cloud_gt = batch['cloud_labels'].long()
        y_hat = self.model(x1, x2, x3, x4, x5, x6)
        cloud_loss = self.criterion(y_hat['cloud_labels'], y_cloud_gt)
        self.training_step_outputs.append(cloud_loss.item())
        
        # Compute metrics
        y_class = y_hat['cloud_labels'].argmax(dim=1)
        self.train_metrics.update(y_class, y_cloud_gt)
        return cloud_loss
    
    def on_train_epoch_end(self) -> None:
        # Compute loss
        loss_per_epoch = torch.Tensor(self.training_step_outputs).mean()
        print(f'After epoch {self.current_epoch}, train_loss = {loss_per_epoch}')
        self.log('train_loss', loss_per_epoch)
        self.training_step_outputs.clear() # used to free up memory
        # Compute metrics

        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        x1 = batch['s1']
        x2 = batch['s2']
        x3 = batch['landcover']
        x4 = batch['azimuth']
        x5 = batch['wateroccurence']
        x6 = batch['elevation']

        y_cloud_gt = batch['cloud_labels'].long()
        y_hat = self.model(x1, x2, x3, x4, x5, x6)
        cloud_loss = self.criterion(y_hat['cloud_labels'], y_cloud_gt)
        self.validation_step_outputs.append(cloud_loss.item())
        
        # Compute metrics
        y_class = y_hat['cloud_labels'].argmax(dim=1)
        self.val_metrics.update(y_class, y_cloud_gt)
        
        return cloud_loss
        

    def on_validation_epoch_end(self) -> None:
        # Compute loss
        loss_per_epoch = torch.Tensor(self.validation_step_outputs).mean()
        print(f'After epoch {self.current_epoch}, val_loss = {loss_per_epoch}')
        self.log('val_loss', loss_per_epoch)
        self.validation_step_outputs.clear() # used to free up memory
        
        # Compute metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        x1 = batch['s1']
        x2 = batch['s2']
        x3 = batch['landcover']
        x4 = batch['azimuth']
        x5 = batch['wateroccurence']
        x6 = batch['elevation']

        y_cloud_gt = batch['cloud_labels'].long()
        y_hat = self.model(x1, x2, x3, x4, x5, x6)
        cloud_loss = self.criterion(y_hat['cloud_labels'], y_cloud_gt)
        self.test_step_outputs.append(cloud_loss.item())
        
        # Compute metrics
        y_class = y_hat['cloud_labels'].argmax(dim=1)
        self.test_metrics.update(y_class, y_cloud_gt)
        return cloud_loss

    
    def on_test_epoch_end(self) -> None:
        # Compute loss
        loss_per_epoch = torch.Tensor(self.test_step_outputs).mean()
        self.log('test_loss', loss_per_epoch)
        self.test_step_outputs.clear() # free memory
        
        # Compute metrics
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()
    
    def predict_step(self, batch, batch_idx=None):
        x = batch['s2']
        x1 = batch['s1']
        x2 = batch['s2']
        x3 = batch['landcover']
        x4 = batch['azimuth']
        x5 = batch['wateroccurence']
        x6 = batch['elevation']

        output = self.model(x1, x2, x3, x4, x5, x6)
        # sigmoid = nn.Sigmoid()
        # sigmoid_output = sigmoid(output)

        softmax = nn.Softmax(dim=1)
        softmax_output = softmax(output['cloud_labels'])
        return softmax_output
