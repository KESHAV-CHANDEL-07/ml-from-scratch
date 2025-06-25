import numpy as np
import matplotlib.pyplot as plt

# 1. Load trained model parameters
params = np.load("model_params.npz")
w = params['w']
b = params['b']
x_mean = params['x_mean']
x_std = params['x_std']
y_mean = params['y_mean']
y_std = params['y_std']

# 2. Define prediction function
def predict_value(sq_feet):
    x_scaled = (sq_feet - x_mean) / x_std
    y_scaled = w * x_scaled + b
    y = y_scaled * y_std + y_mean
    return y

# 3. Take input and make prediction
sq_feet = float(input("Enter house size in square feet: "))
predicted_price = predict_value(sq_feet)
print(f"Predicted House Price: ₹{predicted_price:.2f}")

# 4. Plot the prediction
x_range = np.linspace(0, 5000, 100)
y_range = [predict_value(val) for val in x_range]
plt.plot(x_range, y_range, color='blue', label='Prediction Line')
plt.scatter([sq_feet], [predicted_price], color='green', s=100, label=f'{sq_feet} sq.ft → ₹{predicted_price:.0f}')
plt.grid(True)
plt.legend()
plt.show()