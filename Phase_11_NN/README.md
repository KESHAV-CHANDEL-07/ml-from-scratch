# My CNN from Scratch Journey

This repository contains my exploration and hands-on implementation of a Convolutional Neural Network (CNN) from scratch using **NumPy** and **PyTorch**. The goal of this project is to understand the inner workings of CNNs, from convolution to fully connected layers, and to implement gradient descent for training.

---

## Overview

Convolutional Neural Networks (CNNs) are a class of deep learning models commonly used for image classification and recognition. Unlike standard neural networks, CNNs leverage **convolutional layers**, **activation functions**, and **pooling layers** to efficiently learn spatial hierarchies in images.

In this project, I implemented the following components from scratch:

1. **Data Loading**  
   - MNIST dataset (handwritten digits 0–9)
   - Preprocessing: normalization and reshaping

2. **Forward Pass Components**
   - **Convolution Layer**: Sliding window convolution over input images with multiple kernels
   - **ReLU Activation**: Introduced non-linearity
   - **Max Pooling**: Downsampling feature maps
   - **Flattening**: Conversion of pooled feature maps into a 1D vector
   - **Dense Layer**: Fully connected layer for classification
   - **Softmax**: Probability distribution over classes

3. **Training Components**
   - Implemented **cross-entropy loss** for classification
   - Added **gradient descent updates** for weights and biases
   - Training loop with **epochs**, manually updating parameters
   - CUDA-enabled using **PyTorch** for GPU acceleration

---

## Dependencies

The project uses the following Python libraries:

- [PyTorch](https://pytorch.org/) (for CUDA support and tensor operations)
- [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist) (for loading MNIST dataset)
- [Matplotlib](https://matplotlib.org/) (optional, for visualizing images)

Install dependencies via pip:

```bash
pip install torch tensorflow matplotlib

