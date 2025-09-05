import numpy as np
import matplotlib.pyplot as plt

# 1. Original data
x = np.array([500, 1000, 1500, 2000, 2500]).reshape(-1, 1)  # House size (sq. ft)
y = np.array([700000, 900000, 1100000, 1300000, 1500000]).reshape(-1, 1)  # Prices (₹)

# 2. Feature scaling (Standardization)
x_mean = np.mean(x)
x_std = np.std(x)
x_scaled = (x - x_mean) / x_std

y_mean = np.mean(y)
y_std = np.std(y)
y_scaled = (y - y_mean) / y_std

# 3. Initialize parameters
np.random.seed(42)
w = np.random.randn() * 0.01
b = np.random.randn() * 0.01

# 4. Model and loss functions
def predict(x_input):
    return w * x_input + b

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 5. Training settings
learning_rate = 0.0009
epochs = 1000
losses = []

# 6. Training loop
for i in range(epochs):
    y_pred = predict(x_scaled)
    loss = compute_loss(y_scaled, y_pred)
    losses.append(loss)

    dw = -2 * np.mean((y_scaled - y_pred) * x_scaled)
    db = -2 * np.mean(y_pred - y_scaled)

    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 200 == 0:
        print(f"Epoch {i:>4}: w = {w:.4f}, b = {b:.4f}, loss = {loss:.6f}")
        print(f"Loss at epoch {i}: {losses[-1]:.6f}")

# 7. Final output
print(f"\nFinal parameters: w = {w:.4f}, b = {b:.4f}")
print(f"Final loss: {losses[-1]:.6f}")

# 8. Prediction for a new house
def predict_price(size_sqft):
    x_new_scaled = (size_sqft - x_mean) / x_std
    y_new_scaled = predict(x_new_scaled)
    y_new = y_new_scaled * y_std + y_mean
    return y_new

size = 3000
predicted_price = predict_price(size)
print(f"\nPredicted price for {size} sq.ft house: ₹{predicted_price:.2f}")

# 9. Plot: Prediction vs Actual
plt.figure(figsize=(8, 5))
plt.scatter(x, y, color='blue', label='Original Data')
plt.plot(x, predict(x_scaled) * y_std + y_mean, color='red', label='Fitted Line')
plt.title('House Price Prediction')
plt.xlabel('House Size (sq ft)')
plt.ylabel('Price (₹)')
plt.legend()
plt.grid(True)
plt.show()

# 10. Plot: Loss over epochs
plt.figure(figsize=(8, 4))
plt.plot(range(epochs), losses, color='violet')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Save model parameters to a file after training
np.savez("model_params.npz", w=w, b=b, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
print("Model parameters saved to model_params.npz")


