import numpy as np
import matplotlib.pyplot as plt

# 1. Input data
x_original = np.array([150, 160, 170, 180, 190]).reshape(-1, 1)
y = np.array([50, 56, 63, 70, 77]).reshape(-1, 1)

# 2. Normalize x
x_mean = np.mean(x_original)
x_std = np.std(x_original)
x = (x_original - x_mean) / x_std  # x is now scaled to ~[-1, 1]

# 3. Initialize parameters
np.random.seed(42)
w = np.random.randn() * 0.01
b = np.random.randn() * 0.01

# 4. Prediction and loss
def predict(x):
    return w * x + b

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 5. Training loop
learning_rate = 0.01
epochs = 10100
losses = []

for i in range(epochs):
    y_pred = predict(x)
    loss = compute_loss(y, y_pred)
    losses.append(loss)

    # Gradients
    dw = -2 * np.mean(x * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    # Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 100 == 0:
        print(f"Epoch {i}: loss = {loss:.6f}, w = {w:.4f}, b = {b:.4f}")

# 6. Final prediction using original x
x_test = (x_original - x_mean) / x_std
y_final = predict(x_test)

print(f"\nFinal Loss: {losses[-1]:.6f}")
print(f"Final Weights: w = {w:.4f}, b = {b:.4f}")

# 7. Plot prediction vs actual
plt.figure(figsize=(8, 5))
plt.scatter(x_original, y, color='blue', label='Actual Data')
plt.plot(x_original, y_final, color='red', label='Predicted Line')
plt.title("Height vs Weight - Linear Regression")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.legend()
plt.grid(True)
plt.show()

# 8. Plot loss over time
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.show()
