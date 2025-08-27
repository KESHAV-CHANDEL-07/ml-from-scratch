import numpy as np
import matplotlib.pyplot as plt

# Create a simple 5x5 grayscale image (1 channel)
image = np.array([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 1],
    [1, 0, 1, 2, 1],
    [2, 3, 0, 1, 0],
    [1, 2, 1, 0, 2]
], dtype=np.float32)

# Plot the input image
plt.imshow(image)
plt.imshow(image,cmap = "grey")
plt.title("Input")
plt.colorbar()
plt.show()

filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

# Input and filter dimensions
H, W = image.shape
kH, kW = filter.shape
out_H = H - kH + 1
out_W = W - kW + 1

# Output feature map
output = np.zeros((out_H, out_W))

# Perform convolution
for i in range(out_H):
    for j in range(out_W):
        region = image[i:i+kH, j:j+kW]
        output[i, j] = np.sum(region * filter)

print("Convolved Output:\n", output)

plt.imshow(output, cmap='gray')
plt.title("Output Feature Map (After Convolution)")
plt.colorbar()
plt.show()

#batch normalization
def batch_norm(X):
    mean = np.mean(X)
    std = np.std(X)
    X_norm = (X - mean) / (std + 1e-5)
    return X_norm

bn_output = batch_norm(output)
print("\nAfter Batch Normalization:\n", bn_output)     #not affect so much due to small dataset

plt.imshow(bn_output, cmap='gray')
plt.title("Output Feature Map (After BN)")
plt.colorbar()
plt.show()


#relu for 0-1
output = np.array([
    [-4., -2., 3.],
    [ 0., -2., 1.],
    [ 2.,  2., -1.]
])

relu_output = np.maximum(0, output)
print("After ReLU:\n", relu_output)

plt.imshow(relu_output, cmap='gray')
plt.title("Output Feature Map (After ReLU)")
plt.colorbar()
plt.show()