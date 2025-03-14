from typing import Optional, Union, Dict, Tuple
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import Module
import lightning as L
import numpy as np

## Pytorch Lightning Model Class used to configure the model for training and inference
class SegmentationModel_S1(L.LightningModule):
    def __init__(
            self, 
            architecture: Module, 
            criterion, 
            optimizer: torch.optim.Optimizer,
            metric_collection: torchmetrics.MetricCollection,
        ) -> None:
        super(SegmentationModel_S1, self).__init__()
        self.save_hyperparameters(ignore=["architecture", "metric_collection"])
        
        self.model = architecture
        self.criterion = criterion
        self.optimizer = optimizer
        # Other metrics to be tracked on train, val and test sets
        self.train_metrics = metric_collection.clone(prefix="train_")
        self.val_metrics   = metric_collection.clone(prefix='val_')
        self.test_metrics  = metric_collection.clone(prefix='test_')
        # Lists for storing and computing losses
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer

    def training_step(self, batch, batch_idx):
        
        x = batch['s1']
        #print(x)
        y = batch['cloud_labels']
        #print(y)
        y_gt = y.long()
        y_hat = self.model(x)
        predicted = y_hat['cloud_labels']

        ##Loss Calculation where predicted shape is [BxCxHxW] and y_gt shape is [BxHxW]
        loss = self.criterion(predicted, y_gt)
        self.training_step_outputs.append(loss.item())
        
        
        # Apply softmax activation to raw outputs of the model
        #y_softmax = nn.Softmax(predicted)
        # print(f"datatype of softmax is {type(y_softmax)}")
        # one_hot_gt = torch.nn.functional.one_hot(y_gt, num_classes=4)
        # print(one_hot_gt.shape)
        # reshaped_gt = y_gt.permute(0,3,1,2)
        # print(f"reshaped gt shape is {reshaped_gt.shape}")

        y_class = predicted.argmax(dim=1)
        self.train_metrics.update(y_class, y_gt)
        return loss
    
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
        
        x = batch['s1']
        y = batch['cloud_labels']
        y_gt = y.long()
        #y_gt = torch.tensor(y_gt, dtype=torch.long)
        y_hat = self.model(x)
        predicted = y_hat['cloud_labels']

        ##Loss Calculation where predicted shape is [BxCxHxW] and y_gt shape is [BxHxW]
        val_loss = self.criterion(predicted, y_gt)
        self.validation_step_outputs.append(val_loss.item())
        
        ##Apply softmax activation to raw outputs of the model
        #y_softmax = nn.Softmax(predicted)
        
        y_class = predicted.argmax(dim=1)
        self.val_metrics.update(y_class, y_gt)
        return val_loss
    
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
        x = batch['s1']
        y = batch['cloud_labels']
        y_gt = y.long()
        #y_gt = torch.tensor(y_gt, dtype=torch.long)
        y_hat = self.model(x)
        predicted = y_hat['cloud_labels']

        ##Loss Calculation where predicted shape is [BxCxHxW] and y_gt shape is [BxHxW]
        test_loss = self.criterion(predicted, y_gt)

        #print(f"test_loss is {test_loss}")
        #print(f"test_loss item is {test_loss.item()}")

        self.test_step_outputs.append(test_loss.item())
        
        # Apply softmax activation to raw outputs of the model
        #y_softmax = nn.Softmax(y_hat)
        
        #print(f"predicted shape is {predicted.shape}")

        y_class = predicted.argmax(dim=1)
        #print(f"y_class shape is {y_class.shape}")
        self.test_metrics.update(y_class, y_gt)
        return test_loss
    
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
        x = batch['s1']
        output = self.model(x)
        # sigmoid = nn.Sigmoid()
        # sigmoid_output = sigmoid(output)

        softmax = nn.Softmax(dim=1)
        softmax_output = softmax(output['cloud_labels'])
        return softmax_output
