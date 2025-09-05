import numpy as np
import matplotlib.pyplot as plt

# Inputs (Features)
x1 = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1,1)  # Size in sq feet
x2 = np.array([2, 3, 3, 4, 4]).reshape(-1,1)                # Number of bedrooms

# Output (Target)
y = np.array([500000, 700000, 900000, 1100000, 1300000]).reshape(-1,1)  # Price in ₹

# Combine features
X = np.hstack((x1, x2))  # Shape: (5,2)

# Standardize inputs
x_mean = np.mean(X, axis=0)
x_std = np.std(X, axis=0)
X_scaled = (X - x_mean) / x_std

# Standardize target
y_mean = np.mean(y)
y_std = np.std(y)
y_scaled = ((y - y_mean) / y_std).reshape(-1, 1)

# Initialize parameters
np.random.seed(42)
w = np.random.randn(2)
b = np.random.randn()

# Prediction function
def predict(X_input):
    return X_input @ w + b

# Loss function (MSE)
def compute_loss(y, y_predicted):
    return np.mean((y - y_predicted) ** 2)

# Training settings
learning_rate = 0.01
epochs = 10000
losses = []

# Training loop
for i in range(epochs):
    y_predicted = predict(X_scaled).reshape(-1, 1)
    loss = compute_loss(y_scaled, y_predicted)
    losses.append(loss)

    error = (y_scaled - y_predicted).flatten()  # (5,)
    dw = -2 * X_scaled.T @ error / len(X_scaled)  # shape: (2,)
    db = -2 * np.mean(error)

    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 1000 == 0:
        print(f"Epoch {i}: Loss = {loss:.6f}")

# Final parameters
print(f"\nFinal Weights: {w}")
print(f"Final Bias: {b}")
print(f"Final Loss: {losses[-1]:.6f}")

# De-standardize predictions
y_pred_scaled = predict(X_scaled).reshape(-1, 1)
y_pred = y_pred_scaled * y_std + y_mean

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(x1, y, color='blue', label='Actual Prices')
plt.plot(x1, y_pred, color='orange', label='Predicted Prices', linewidth=2)
plt.xlabel('Size in sq feet')
plt.ylabel('Price in ₹')
plt.title('House Price Prediction')
plt.legend()
plt.grid(True)
plt.show()

# Plot loss over epochs
plt.figure(figsize=(10, 4))
plt.plot(losses, color='orange')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
