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

#Pick device: CUDA if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# # Load MNIST data
# (X_train, y_train), (_, _) = mnist.load_data()

# images = torch.tensor(X_train[0:12], dtype=torch.float32, device=device) / 255.0
# labels = torch.tensor(y_train[0:12], dtype=torch.long, device=device)

# images = images.unsqueeze(-1)  # shape (N, 28, 28, 1)
# plt.imshow(images[0, :, :, 0].cpu(), cmap="gray")  # first image, first channel
# plt.title("After Pooling - Image 0, Channel 0")
# plt.show()
# plt.imshow(images[1, :, :, 0].cpu(), cmap="gray")  # first image, first channel
# plt.title("After Pooling - Image 0, Channel 0")
# plt.show()
# plt.imshow(images[2, :, :, 0].cpu(), cmap="gray")  # first image, first channel
# plt.title("After Pooling - Image 0, Channel 0")
# plt.show()
# plt.imshow(images[3, :, :, 0].cpu(), cmap="gray")  # first image, first channel
# plt.title("After Pooling - Image 0, Channel 0")
# plt.show()
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

#  8 kernels of size 3x3
# kernels = torch.randn(8, 3, 3, 1, dtype=torch.float32, device=device, requires_grad=True)

# conv_output = Conv2D(images, kernels)
# relu_output = relu(conv_output)

#  Max pooling
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

# pooled_output = maxpooling(relu_output)
# plt.imshow(pooled_output[0, :, :, 0].cpu(), cmap="gray")
# plt.title("After Pooling - Image 0, Channel 0")
# plt.show()
# plt.imshow(pooled_output[1, :, :, 0].cpu(), cmap="gray")  
# plt.title("After Pooling - Image 0, Channel 0")
# plt.show()
# plt.imshow(pooled_output[2, :, :, 0].cpu(), cmap="gray") 
# plt.title("After Pooling - Image 0, Channel 0")
# plt.show()
# plt.imshow(pooled_output[3, :, :, 0].cpu(), cmap="gray")  
# plt.title("After Pooling - Image 0, Channel 0")
# plt.show()


# Flatten
def flatten(feature_maps):
    batch_size = feature_maps.shape[0]
    return feature_maps.reshape(batch_size, -1)

# flattened_output = flatten(pooled_output)
# print(flattened_output.shape)

#Dense layer + softmax
def dense_layer(X, w, b):
    return X @ w.T + b  # (batch, num_classes)

# torch.manual_seed(42)
# weights = torch.randn(10,1352,dtype=torch.float32, device=device,requires_grad=True)
# bias = torch.randn(10, dtype=torch.float32, device=device,requires_grad=True)

# epochs = 50
# learning_rate = 0.01
# criterion = torch.nn.CrossEntropyLoss()

def forward(x):
    conv_out = Conv2D(x, kernels)
    relu_out = relu(conv_out)
    pooled = maxpooling(relu_out)
    flat = flatten(pooled)
    logits = dense_layer(flat, weights, bias)
    return logits


# for epoch in range(epochs):
#     logits = forward(images)
#     loss = criterion(logits, labels)

    # Backpropagation
    # loss.backward()

    # with torch.no_grad():
    #     kernels -= learning_rate * kernels.grad
    #     weights -= learning_rate * weights.grad
    #     bias -= learning_rate * bias.grad
        
    #     # Zero gradients for next step
    #     kernels.grad.zero_()
    #     weights.grad.zero_()
    #     bias.grad.zero_()
    
    # if epoch % 5 == 0:
    #     pred_classes = torch.argmax(logits, dim=1)
    #     acc = (pred_classes == labels).float().mean()
    #     print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc*100:.2f}%")


# logits = forward(images)
# pred_classes = torch.argmax(logits, dim=1)
# print("Predicted classes:", pred_classes)
# print("True labels:", labels)

# torch.save({
#     "kernels": kernels,
#     "weights": weights,
#     "bias": bias
# }, "mnist_scratch_model.pth")

# print("Model saved as mnist_scratch_model.pth")



# Load trained parameters from .pth file
checkpoint = torch.load("mnist_scratch_model.pth", map_location=device)
kernels = checkpoint["kernels"]
weights = checkpoint["weights"]
bias = checkpoint["bias"]

print("Trained model loaded!")


import cv2
import numpy as np

# Load your image
img = cv2.imread(r"C:\Users\kesha\OneDrive\Desktop\Projects\ml-from-scratch\Phase_11_NN\digit.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))                          # resize to MNIST size
img = torch.tensor(img, dtype=torch.float32, device=device) / 255.0
img = img.unsqueeze(0).unsqueeze(-1)  # shape (1, 28, 28, 1)

# Get predictions
logits = forward(img)
probs = torch.nn.functional.softmax(logits, dim=1)
pred_class = torch.argmax(probs, dim=1)

print("Predicted digit:", pred_class.item())
