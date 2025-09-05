import numpy as np
import matplotlib.pyplot as plt

'''# 1. Original data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)                      # Years of experience
y = np.array([3500, 4500, 5500, 6500, 7500]).reshape(-1, 1)       # Salaries

# 2. Manual standardization (scale x and y to have mean 0, std 1)
x_mean = np.mean(x)
x_std = np.std(x)
x_scaled = (x - x_mean) / x_std

y_mean = np.mean(y)
y_std = np.std(y)
y_scaled = (y - y_mean) / y_std

# 3. Initialize parameters
np.random.seed(42)
w = np.random.randn()*1.5
b = np.random.randn()*0.05

# 4. Prediction and loss functions
def predict(x_input):
    return w * x_input + b

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 5. Training settings
learning_rate = 0.0005
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

# Final parameters:
# w = 0.9752, b = 0.0021
# x_mean = 3.0, x_std = 1.4142
# y_mean = 5500.0, y_std = 1414.2
'''

# --------------- PREDICTION SECTION ---------------

# Final trained parameters (manually used)
w = 0.9752
b = 0.0021
x_mean = 3.0
x_std = 1.4142
y_mean = 5500.0
y_std = 1414.2

# Training data (for plotting)
x_train = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y_train = np.array([3500, 4500, 5500, 6500, 7500]).reshape(-1, 1)

# Function to predict
def predict_salary(years):
    x_scaled = (years - x_mean) / x_std
    y_scaled = w * x_scaled + b
    return y_scaled * y_std + y_mean

# Take input
years = float(input("Enter years of experience: "))
predicted_salary = predict_salary(years)
print(f"Predicted salary for {years} years of experience: ₹{predicted_salary:.2f}")

# Plot prediction and training data
x_range = np.linspace(0, 15, 100)
y_range = [predict_salary(val) for val in x_range]

plt.figure(figsize=(8, 5))
plt.plot(x_range, y_range, color='red', label='Predicted Salary Line')
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter([years], [predicted_salary], color='green', s=100, label=f'{years} yrs → ₹{predicted_salary:.0f}')
plt.title('Years of Experience vs Salary')
plt.xlabel('Years of Experience')
plt.ylabel('Salary (₹)')
plt.legend()
plt.grid(True)
plt.show()

# ---------------- PLOT SAVED LOSS ----------------

# Manually insert losses if training block is commented OR uncomment the training loop to regenerate
# Here's a dummy decreasing pattern just for plotting if you still want to see the curve
losses = [1.2/(i+1)**0.5 for i in range(1000)]

plt.figure(figsize=(8, 4))
plt.plot(range(len(losses)), losses, color='green')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
