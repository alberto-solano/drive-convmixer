# Deep learning architectures for diagnosis of diabetic retinopathy

Alberto Solano $^{1}$, Kevin N. Dietrich $^{1}$, Marcelino Martínez-Sober $^{1}$, Regino Barranquero-Cardeñosa $^{1}$, Jorge Vila-Tomás $^{2}$\*, Pablo Hernández-Cámara $^{2}$\*

> $^{1}$ Intelligent Data Analysis Laboratory, ETSE (Engineering School), Universitat de València, 46100 Burjassot, Spain

> $^{2}$ Image Processing Lab., Universitat de València, 46980 Paterna, Spain

## Repository Index

1. [Abstract](#abstract)
2. [Repository Organization](#repository-organization)
3. [Model Weights](#model-weights)

## Abstract

For many years, convolutional neural networks dominated the field of computer vision, not least in the medical field, where problems such as image segmentation were addressed by such networks as the U-Net. The arrival of self-attention based networks to the field of computer vision through ViTs seems to have changed the trend of using standard convolutions. Throughout this work, we apply different architectures such as U-Net, ViTs and ConvMixer, to compare their performance on a medical semantic segmentation problem.
All the models have been trained from scratch on the DRIVE dataset and evaluated on its private counterpart to assess which of the models performed better in the segmentation problem.
Our major contribution is showing that the best performing model (ConvMixer) is the one that shares the approach from the ViT (processing images as patches) while maintaining the foundational blocks (convolutions) from the U-Net. 
This mixture doesn't only produce better results ($DICE=0.83$) than both ViTs ($0.80$/$.077$ for UNETR/SWIN-Unet) and the U-Net ($0.82$) on their own, but reduces considerably the number of parameters (2.97M against 104M/27M and 31M respectively), showing that there is no need to systematically use large models for solving image problems where smaller architectures
with the optimal pieces can get better results.

## Repository organization

This repository contains all the code and data needed to reproduce the results shown in the paper.

The `Data` folder contains the different splits (training, validation and test) used, while the `Code` folder is organized as follows:

- `Checkpoint_<model>`: Results of each model training.
- `Models/`: PyTorch definition of the models used.
- `Train/`: Training script of each model.
- `Utils/dataset.py`: PyTorch dataset and augmentations used.
- `Utils/utils.py`: Utilities to process the data and manage the training.

## Model weights

- **ConvMixer (Light)**: [Weights](https://drive.google.com/file/d/1OJyf7EOdn4tIL85qHbhvAHILX4NNURbQ/view?usp=share_link) | [Parameters](https://drive.google.com/file/d/1a0P9SghXbVaRTtCd6lX-auk0tWVu3xY4/view?usp=share_link)

- **ConvMixer**: [Weights](https://drive.google.com/file/d/1lXsy8lZoffX6hsEBE0YyU7lMym259LSi/view?usp=share_link) | [Parameters](https://drive.google.com/file/d/1gB3jipjE5Cz2_c6kNipbuXKWyS3EMsws/view?usp=share_link)