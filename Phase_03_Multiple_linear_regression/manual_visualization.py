import numpy as np
import matplotlib.pyplot as plt

# Step 1: Prepare data
x1 = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1,1)
x2 = np.array([2, 3, 3, 4, 4]).reshape(-1,1)
y = np.array([500000, 700000, 900000, 1100000, 1300000]).reshape(-1,1)

X = np.hstack((x1, x2))
print("Step 1: Combined Feature Matrix X:")
print("Shape:", X.shape)
print(X)
input("Press Enter to continue...\n")

# Step 2: Standardize features
x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)
print("Step 2: Mean of X:", x_mean)
print("Standard deviation of X:", x_std)
input("Press Enter to continue...\n")

X_scaled = (X - x_mean) / x_std
print("Standardized X (X_scaled):")
print("Shape:", X_scaled.shape)
print(X_scaled)
input("Press Enter to continue...\n")

# Step 3: Standardize target
y_mean = np.mean(y)
y_std = np.std(y)
print("Mean of y:", y_mean)
print("Standard deviation of y:", y_std)
input("Press Enter to continue...\n")

y_scaled = (y - y_mean) / y_std
print("Standardized y (y_scaled):")
print("Shape:", y_scaled.shape)
print(y_scaled)
input("Press Enter to continue...\n")

# Step 4: Initialize weights and bias
np.random.seed(42)
w = np.random.randn(2)
b = np.random.randn()
print("Initial weights w:", w)
print("Initial bias b:", b)
input("Press Enter to continue...\n")

# Step 5: Define functions
def predict(X_input):
    return X_input @ w + b

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 6: Training loop
learning_rate = 0.01
epochs = 1000
losses = []

for i in range(epochs):
    y_pred = predict(X_scaled)
    loss = compute_loss(y_scaled, y_pred)
    losses.append(loss)

    error = y_scaled.flatten() - y_pred  # shape (5,)
    dw = -2 * X_scaled.T @ error / len(X_scaled)  # shape (2,)
    db = -2 * np.mean(error)

    if i % 500 == 0:
        print(f"\nEpoch {i}")
        print("Prediction y_pred:", y_pred)
        print("Error:", error)
        print("Gradient dw:", dw)
        print("Gradient db:", db)
        print("Loss:", loss)
        input("Press Enter to continue...\n")

    w -= learning_rate * dw
    b -= learning_rate * db

# Final output
print("\nFinal weights w:", w)
print("Final bias b:", b)
print("Final loss:", losses[-1])
