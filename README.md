## Overview

This repository contains code for multi-class classification of brain tumor MRI images using Transfer Learning with [EfficientNetB1](https://arxiv.org/abs/1905.11946). The model is trained to classify tumors into three categories: Glioma, Meningioma, and Pituitary.

## Dataset

The preprocessed dataset is available [here](https://drive.google.com/drive/folders/1_E4ZD6WvLLwj2WGuQbQ3YZ28p9FtYpZ1?usp=sharing). It is already split into training, validation, and test sets.  
*If you need to reproduce the preprocessing steps*, follow the optional steps below using scripts `1_Convert_mat_to_jpg_Split.py`, `2_Crop_Brain.py`, and `3_Data_Augmentation.py`.


## Repository Structure

| File Name                     | Description                                  |
|-------------------------------|----------------------------------------------|
| `1_Convert_mat_to_jpg_Split.py` | Converts .mat files to JPEG and splits dataset |
| `2_Crop_Brain.py`              | Crops brain regions using Canny edge detection |
| `3_Data_Augmentation.py`       | Applies data augmentation to training data   |
| `4_Train_Test.py`              | Main script for training/testing the model   |


## Results

The model outputs:
  - Training/validation accuracy and loss curves
  - Test accuracy, ROC-AUC scores
  - Confusion matrix and classification report


