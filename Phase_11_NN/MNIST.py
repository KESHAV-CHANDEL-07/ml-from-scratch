# import numpy as np
# from tensorflow.keras.datasets import mnist
# import matplotlib.pyplot as plt

# # Load MNIST data
# (X_train, y_train), (_, _) = mnist.load_data()

# images = X_train.astype(np.float32) / 255.0
# label = y_train

# images = images.reshape(-1,28, 28, 1)

# #plt.imshow(image.squeeze(), cmap='gray')
# #plt.title(f"Label: {label}")
# #plt.show()

# def Conv2D(image , kernel):
#     batch_size,H,W,C1 =image.shape
#     num_kernels,kH, kW, _ =kernel.shape
#     Out_H = H - kH + 1
#     Out_W = W - kW + 1

#     output = np.zeros((batch_size,Out_H ,Out_W,num_kernels))
#     for n in range(batch_size):
#         image = images[n]
#         for k in range(num_kernels):
#             kernel = kernels[k]        
#             for i in range(Out_H):
#                 for j in range(Out_W):
#                     region = image[i:i+kH , j:j+kW, :]
#                     output[n,i,j,k] = np.sum(region*kernel)

#     return output
# def relu(x):
#     return np.maximum(0, x)

# kernels = np.random.randn(8, 3, 3, 1).astype(np.float32)


# conv_output = Conv2D(images, kernels)

# #plt.imshow(conv_output, cmap='gray')
# #plt.title("After Convolution")
# #plt.show()

# relu_output = relu(conv_output)
# #plt.imshow(relu_output, cmap='gray')
# #plt.title("After ReLU")
# #plt.show()

# def maxpooling(feature_map , size = 2, stride =2 ):
#     batch_size,H,W,C= feature_map.shape
#     Out_H = ((H - size )//stride) +1
#     Out_W = ((W - size )//stride) +1
#     pooled = np.zeros((batch_size,Out_H , Out_W ,C))
#     for n in range (batch_size):
#         for c in range(C):
#             for i in range(Out_H):
#                 for j in range(Out_W):
#                     region = feature_map[n,i*stride:i*stride+size , j*stride: j*stride+size, c]
#                     pooled[n,i, j,c] = np.max(region)
#     return pooled

# pooled_output = maxpooling(relu_output)
# #plt.imshow(pooled_output,cmap = "gray")
# #plt.title("After Pooling")
# #plt.show()

# def flatten(feature_maps):
#     batch_size = feature_maps.shape[0]
#     return feature_maps.reshape(batch_size, -1)
# flattened_output = flatten(pooled_output)
# print(flattened_output.shape)

# def dense_layer(X, w ,b):
#     return np.dot(X,w.T ) +b
# def softmax(logits):
#     # logits: (batch_size, num_classes)
#     exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
#     return exps / np.sum(exps, axis=1, keepdims=True)


# input_len = flattened_output.shape[1]

# # Random weights and bias for 10 output neurons
# np.random.seed(42)
# weights = np.random.randn(10, input_len) 
# bias = np.random.randn(10)                
# logits = dense_layer(flattened_output, weights, bias)
# probs = softmax(logits)

# print("Predicted classes:", np.argmax(probs, axis=1))


import torch
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Pick device: CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load MNIST data
(X_train, y_train), (_, _) = mnist.load_data()

images = torch.tensor(X_train[0:64], dtype=torch.float32, device=device) / 255.0
labels = torch.tensor(y_train[0:64], dtype=torch.long, device=device)

images = images.unsqueeze(-1)  # shape (N, 28, 28, 1)

# Convolution
def Conv2D(image, kernel):
    batch_size, H, W, C1 = image.shape
    num_kernels, kH, kW, _ = kernel.shape
    Out_H = H - kH + 1
    Out_W = W - kW + 1

    output = torch.zeros((batch_size, Out_H, Out_W, num_kernels), device=device)
    for n in range(batch_size):
        img = image[n]
        for k in range(num_kernels):
            kern = kernel[k]
            for i in range(Out_H):
                for j in range(Out_W):
                    region = img[i:i+kH, j:j+kW, :]
                    output[n, i, j, k] = torch.sum(region * kern)
    return output

def relu(x):
    return torch.clamp(x, min=0.0)

# 8 kernels of size 3x3
kernels = torch.randn(8, 3, 3, 1, dtype=torch.float32, device=device)

conv_output = Conv2D(images, kernels)
relu_output = relu(conv_output)

# Max pooling
def maxpooling(feature_map, size=2, stride=2):
    batch_size, H, W, C = feature_map.shape
    Out_H = ((H - size) // stride) + 1
    Out_W = ((W - size) // stride) + 1
    pooled = torch.zeros((batch_size, Out_H, Out_W, C), device=device)
    for n in range(batch_size):
        for c in range(C):
            for i in range(Out_H):
                for j in range(Out_W):
                    region = feature_map[n, i*stride:i*stride+size,
                                         j*stride:j*stride+size, c]
                    pooled[n, i, j, c] = torch.max(region)
    return pooled

pooled_output = maxpooling(relu_output)

# Flatten
def flatten(feature_maps):
    batch_size = feature_maps.shape[0]
    return feature_maps.reshape(batch_size, -1)

flattened_output = flatten(pooled_output)
print(flattened_output.shape)

# Dense layer + softmax
def dense_layer(X, w, b):
    return X @ w.T + b  # (batch, num_classes)

def softmax(logits):
    exps = torch.exp(logits - torch.max(logits, dim=1, keepdim=True).values)
    return exps / torch.sum(exps, dim=1, keepdim=True)

input_len = flattened_output.shape[1]

torch.manual_seed(42)
weights = torch.randn(10, input_len, device=device)
bias = torch.randn(10, device=device)

logits = dense_layer(flattened_output, weights, bias)
probs = softmax(logits)

print("Predicted classes:", torch.argmax(probs, dim=1))





# import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # Small GPU test
# x = torch.rand((1000, 1000), device=device)
# y = torch.mm(x, x)   # matrix multiply on GPU
# print("Test complete, result shape:", y.shape)
