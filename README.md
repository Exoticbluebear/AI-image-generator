# AI-image-generator
This is a project I did in Colab as part of my AI course to generate Abstract Expressionism art

Abstract Art Generation Using DCGAN in PyTorch

Project Overview

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate abstract art. The goal is to train a generator model that can produce visually appealing abstract images by learning from a dataset of existing art images.

Key Components

Dependencies
The project relies on several Python libraries, including:

os, numpy, pandas for file handling and data manipulation.
torch for building and training the neural network.
torchvision for image processing and handling datasets.
matplotlib for visualizing images.
cv2 for additional image processing if needed.
Data Preparation
Dataset: The project utilizes the ImageFolder class to load images from a specified directory. Images are resized to 64x64 pixels and normalized.
DataLoader: A DataLoader is used to efficiently load batches of data for training.
Model Architecture
Two neural networks are defined:

Generator: Transforms random noise (latent vectors) into images using transposed convolution layers.
Discriminator: Distinguishes between real images from the dataset and fake images produced by the generator using convolutional layers.
Training Process
Training Loop: The training process alternates between training the discriminator and the generator. The discriminator learns to distinguish real from fake images, while the generator aims to produce convincing fake images.
Loss Functions: Binary cross-entropy loss is used for both the generator and discriminator.
Image Generation
The model saves generated images at regular intervals during training, allowing for monitoring of the generator's progress.
Usage Instructions

Setup Environment: Ensure you have Python and the required libraries installed.
Dataset Preparation: Place your dataset of art images in the specified directory.
Run the Code: Execute the script to train the DCGAN. Adjust hyperparameters like learning rate and number of epochs as needed.
Generated Art: Check the generated directory for images created by the generator during training.

