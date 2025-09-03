import torch
import numpy as np
from emnist import extract_training_samples, extract_test_samples

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
X_train, Y_train = extract_training_samples('digits')
X_test, Y_test = extract_test_samples('digits')


images = X_train[... ,torch.newaxis].astype(torch.float32)/255.0

def Conv2D(image,kernel,padding = 1):
    n_samples,H,W,C=image.shape
    num_kernel, kH, kW, _=kernel.shape
    H_out = H + 2*padding - kH + 1
    W_out = W + 2*padding - kW + 1

    images_padded = torch.pad(images, ((0,0),(padding,padding),(padding,padding),(0,0)), mode='constant')

    output = torch.zeros((n_samples,H_out,W_out,num_kernel))
    for n in range(n_samples):
        img = images_padded[n]
        for k in range(num_kernel):
            kern = kernel[k]
            for i in range(H_out):
                for j in range(W_out):
                    region = img[i:i+kH , j:j+kW , :]
                    output[n,i,j,k] = torch.sum(region*kern)
    return output
kernels =torch.random.randn(8,3,3,1).astype(torch.float32)

def ReLU(x):
    return torch.maximum(0, x)

def MaxPool2D(feature_map, pool_size=2, stride=2):
    n_samples, H, W, C = feature_map.shape
    H_out = (H - pool_size)//stride + 1
    W_out = (W - pool_size)//stride + 1

    output = torch.zeros((n_samples, H_out, W_out, C), dtype=torch.float32)

    for n in range(n_samples):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i*stride
                    w_start = j*stride
                    window = feature_map[n, h_start:h_start+pool_size, w_start:w_start+pool_size, c]
                    output[n, i, j, c] = torch.max(window)
    return output

def Flatten(feature_map):
    n_samples = feature_map.shape[0]
    return feature_map.reshape(n_samples, -1)


def Dense(x, w, b):
    return torch.dot(x, w) + b

def Softmax(x):
    exp_x = torch.exp(x - torch.max(x, axis=1, keepdims=True))
    return exp_x / torch.sum(exp_x, axis=1, keepdims=True)


# Convolution
conv_out = Conv2D(images[:5], kernels, padding=1)

# ReLU
relu_out = ReLU(conv_out)

#MaxPooling
pool_out = MaxPool2D(relu_out, pool_size=2, stride=2)

# Flatten
flat_out = Flatten(pool_out)

# Dense layer (example: 10 outputs)
weights = torch.random.randn(flat_out.shape[1], 10).astype(torch.float32)
bias = torch.random.randn(10).astype(torch.float32)
dense_out = Dense(flat_out, weights, bias)

#Softmax
predictions = Softmax(dense_out)

print(predictions.shape)


