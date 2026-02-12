import numpy as np
import matplotlib.pyplot as plt

# 1. Generate data
x = np.linspace(-2, 2, 10000).reshape(-1, 1)  # shape (100, 1)
y = (x ** 2).reshape(-1, 1)                 # shape (100, 1)

# 2. Polynomial features: x and x^2
X = np.hstack((x, x**2))  # shape (100, 2)

# 3. Standardize inputs and outputs
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y_scaled = (y - y_mean) / y_std

# 4. Initialize weights and bias
np.random.seed(42)
w = np.random.randn(2)
b = np.random.randn()

# 5. Prediction function
def predict(X_input):
    return np.dot(X_input, w) + b  # returns shape (100,)

# 6. Loss function
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 7. Training loop
learning_rate = 0.001
epochs = 1000
losses = []

for i in range(epochs):
    y_pred = predict(X_scaled).reshape(-1, 1)  # shape (100, 1)
    error = y_scaled - y_pred

    loss = compute_loss(y_scaled, y_pred)
    losses.append(loss)

    dw = -2 * np.mean(error * X_scaled, axis=0)  # shape (2,)
    db = -2 * np.mean(error)

    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 200 == 0:
        print(f"Epoch {i}: Loss = {loss:.6f}")

# 8. Final results
print(f"\nFinal Weights: {w}")
print(f"Final Bias: {b}")
print(f"Final Loss: {losses[-1]:.6f}")

# 9. Rescale prediction to original y
y_pred_scaled = predict(X_scaled).reshape(-1, 1)
y_pred = y_pred_scaled * y_std + y_mean

# 10. Plot the original data and model prediction
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Original Data', alpha=0.5)
plt.plot(x, y_pred, color='red', label='Predicted Curve', linewidth=2)
plt.title('Polynomial Regression with Gradient Descent')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# 11. Plot the loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses, color='orange')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 12. Save model parameters
np.savez("model_params.npz", w=w, b=b, x_mean=X_mean, x_std=X_std, y_mean=y_mean, y_std=y_std)

# 13. Predict for a new value
def predict_value(x_input_scalar):
    x_input_poly = np.array([x_input_scalar, x_input_scalar**2])
    x_scaled = (x_input_poly - X_mean) / X_std
    y_scaled = predict(x_scaled)
    y = y_scaled * y_std + y_mean
    return y

# Example usage
int=float(input("Enter a value for x to predict y: "))
new_x = int
predicted_y = predict_value(new_x)
print(f"\nPrediction for x = {new_x}: y = {predicted_y:.4f}")

np.savez("model_params.npz", w=w, b=b, x_mean=X_mean, x_std=X_std, y_mean=y_mean, y_std=y_std)
print("Model parameters saved to model_params.npz")
import numpy as np

# Load the saved model parameters
data = np.load("model_params.npz")
w = data["w"]
b = data["b"]
x_mean = data["x_mean"]
x_std = data["x_std"]
y_mean = data["y_mean"]
y_std = data["y_std"]

# Prediction function
def predict(X_input):
    X_scaled = (X_input - x_mean) / x_std
    y_scaled = np.dot(X_scaled, w) + b
    y = y_scaled * y_std + y_mean
    return y

int=float(input("Enter a value for x (e.g., 1.5): "))
new_x=int
# Example input (e.g., predict for x = 1.5)
x = np.array([[new_x, new_x**2]])  # include both x and x^2 for polynomial
prediction = predict(x)
print(f"Predicted y for x=1.5: {prediction[0]:.2f}")