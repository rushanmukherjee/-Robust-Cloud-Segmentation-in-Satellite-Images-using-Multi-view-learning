# -Robust-Cloud-Segmentation-in-Satellite-Images-using-Multi-view-learning
This repository contains the document and code for my master thesis at Saarland University. 

Structure of repository 
--plots 
--src
  -- dataset
  -- models
  -- training files

Dataset folder consists of dataset classes for single-view and multi-view models.
Models folder consist of both the model class and architecture files.
To Train a model and save checkpoints, one can select from the train_modality.py files
Eval_modality.py evaluates trained model on independent test set and outputs metrics, plots and metadata files. 
