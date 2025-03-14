from typing import Optional, Union, Dict, Tuple
import torch
import torch.nn as nn
import torchmetrics
from torch.nn import Module
import lightning as L
import numpy as np

## Pytorch Lightning Model Class used to configure the model for training and inference
class Multitask_SegmentationModel(L.LightningModule):
    def __init__(
            self, 
            architecture: Module, 
            optimizer: torch.optim.Optimizer,
            metric_collection: torchmetrics.MetricCollection,
            cloudseg_weight: float,
            LC_weight: float,
            ele_weight: float,
            S1_weight: float,

        ) -> None:
        super(Multitask_SegmentationModel, self).__init__()
        self.save_hyperparameters(ignore=["architecture", "metric_collection"])
        
        self.model = architecture
        self.optimizer = optimizer
        
        self.cloudseg_weight = cloudseg_weight
        self.LC_weight = LC_weight
        self.ele_weight = ele_weight
        self.S1_weight = S1_weight
        
        self.criterion_entropy = nn.CrossEntropyLoss()
        self.criterion_mse = nn.MSELoss()

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
        x = batch['s2']
        y_cloud_gt = batch['cloud_labels'].long()
        y_LC_gt = batch['landcover'].float()
        y_ele_gt = batch['elevation'].float()
        y_S1_gt = batch['s1'].float()
        y_hat = self.model(x)

        ##Convert One-hot Encoded Landcover
        y_LC_gt = y_LC_gt.permute(0, 3, 1, 2)
        #print(f"predicted shape {y_hat['landcover'].shape}")
        #print(f"y_LC_gt shape {y_LC_gt.shape}")
        #y_LC_target_classes = y_LC_target.argmax(dim=1)
        #y_hat['elevation'] = y_hat['elevation'].squeeze(1)
        y_ele_gt = y_ele_gt.unsqueeze(1)

        ##Individual Loss calculation
        LC_loss = self.criterion_entropy(y_hat['landcover'], y_LC_gt)
        #print(f"LC loss {LC_loss}")
        cloud_loss = self.criterion_entropy(y_hat['cloud_labels'], y_cloud_gt)
        ele_loss = self.criterion_mse(y_hat['elevation'], y_ele_gt)
        S1_loss = self.criterion_mse(y_hat['s1'], y_S1_gt)

        #Weighted Loss calculation
        loss_main = (self.cloudseg_weight * cloud_loss + self.LC_weight * LC_loss 
        + self.ele_weight * ele_loss + self.S1_weight * S1_loss)

        self.training_step_outputs.append(loss_main.item())

        # y_class = predicted.argmax(dim=1)
        # self.train_metrics.update(y_class, y_gt)
        # print(f"loss_main {loss_main}")

        return {
            'loss': loss_main,
            'cloud_loss': cloud_loss,
            'LC_loss': LC_loss,
            'ele_loss': ele_loss,
            'S1_loss': S1_loss
        }
    
    def on_train_epoch_end(self) -> None:
        # Compute loss
        #print(f"training_step_outputs {self.training_step_outputs}")
        loss_per_epoch = torch.Tensor(self.training_step_outputs).mean()
        print(f'After epoch {self.current_epoch}, train_loss = {loss_per_epoch}')
        self.log('train_loss', loss_per_epoch)
        self.training_step_outputs.clear() # used to free up memory
        # Compute metrics

        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()
    
    def validation_step(self, batch, batch_idx):
        x = batch['s2']
        y_cloud_gt = batch['cloud_labels'].long()
        y_LC_gt = batch['landcover'].float()
        y_ele_gt = batch['elevation'].float()
        y_S1_gt = batch['s1'].float()
        y_hat = self.model(x)

        ##Convert One-hot Encoded Landcover
        y_LC_gt = y_LC_gt.permute(0, 3, 1, 2)
        #y_LC_target_classes = y_LC_target.argmax(dim=1)
        y_hat['elevation'] = y_hat['elevation'].squeeze(1)

        ##Individual Loss calculation
        LC_loss = self.criterion_entropy(y_hat['landcover'], y_LC_gt)
        cloud_loss = self.criterion_entropy(y_hat['cloud_labels'], y_cloud_gt)
        ele_loss = self.criterion_mse(y_hat['elevation'], y_ele_gt)
        S1_loss = self.criterion_mse(y_hat['s1'], y_S1_gt)

        ##Weighted Loss calculation
        loss_main = (self.cloudseg_weight * cloud_loss + self.LC_weight * LC_loss
        + self.ele_weight * ele_loss + self.S1_weight * S1_loss)

        self.validation_step_outputs.append(loss_main.item())

        # y_class = cloudseg_predicted.argmax(dim=1)
        # self.val_metrics.update(y_class, y_gt)

        return {
            'loss': loss_main,
            'cloud_loss': cloud_loss,
            'LC_loss': LC_loss,
            'ele_loss': ele_loss,
            'S1_loss': S1_loss
        }
        

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
        x = batch['s2']
        y_cloud_gt = batch['cloud_labels'].long()
        y_LC_gt = batch['landcover'].long()
        y_ele_gt = batch['elevation'].float()
        y_S1_gt = batch['s1'].float()
        y_hat = self.model(x)

        ##Convert One-hot Encoded Landcover
        y_LC_gt = y_LC_gt.permute(0, 3, 1, 2)
        #y_LC_target_classes = y_LC_target.argmax(dim=1)
        y_hat['elevation'] = y_hat['elevation'].squeeze(1)

        ##Individual Loss calculation
        cloud_loss = self.criterion_entropy(y_hat['cloud_labels'], y_cloud_gt)
        LC_loss = self.criterion_entropy(y_hat['landcover'], y_LC_gt)
        ele_loss = self.criterion_mse(y_hat['elevation'], y_ele_gt)
        S1_loss = self.criterion_mse(y_hat['s1'], y_S1_gt)

        ##Weighted Loss calculation
        loss_main = (self.cloudseg_weight * cloud_loss + self.LC_weight * LC_loss
        + self.ele_weight * ele_loss + self.S1_weight * S1_loss)

        self.test_step_outputs.append(loss_main.item())
        
        # y_class = predicted.argmax(dim=1)
        # self.test_metrics.update(y_class, y_gt)

        return {
            'loss': loss_main,
            'cloud_loss': cloud_loss,
            'LC_loss': LC_loss,
            'ele_loss': ele_loss,
            'S1_loss': S1_loss
        }
    
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
        output = self.model(x)
        # sigmoid = nn.Sigmoid()
        # sigmoid_output = sigmoid(output)

        softmax = nn.Softmax(dim=1)
        softmax_output = softmax(output)
        return softmax_output
