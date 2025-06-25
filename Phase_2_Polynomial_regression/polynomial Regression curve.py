import numpy as np
import matplotlib.pyplot as plt

# 1. Data: Car age vs price
x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)  # Car age
y = np.array([800, 680, 540, 400, 300, 220, 160]).reshape(-1, 1)  # Price in ₹

min_age = np.min(x)
max_age = np.max(x)

# 2. Polynomial features: x and x^2
X = np.hstack((x, x**2))  # shape: (7, 2)

# 3. Feature scaling
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

# 5. Prediction and loss functions
def predict(X_input):
    return X_input @ w + b

def compute_loss(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)

# 6. Training loop
learning_rate = 0.009
epochs = 10000
losses = []

for i in range(epochs):
    y_predicted = predict(X_scaled)
    loss = compute_loss(y_scaled.flatten(), y_predicted)
    losses.append(loss)

    error = (y_scaled.flatten() - y_predicted)  # shape (7,)
    dw = -2 * np.mean(error[:, np.newaxis] * X_scaled, axis=0)  # shape (2,)
    db = -2 * np.mean(error)

    w -= learning_rate * dw
    b -= learning_rate * db

    if i % 200 == 0:
        print(f"Epoch {i}: Loss = {loss:.6f}")

# 7. Final results
print(f"\nFinal Weights: {w}")
print(f"Final Bias: {b}")
print(f"Final Loss: {losses[-1]:.6f}")

# 8. Rescale predictions to original y
y_pred_scaled = predict(X_scaled).reshape(-1, 1)
y_pred = y_pred_scaled * y_std + y_mean

# 9. Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Actual Data', s=100)
plt.plot(x, y_pred, color='blue', label='Polynomial Fit', linewidth=2)
plt.title('Car Age vs Price')
plt.xlabel('Car Age (Years)')
plt.ylabel('Price (₹)')
plt.legend()
plt.grid(True)
plt.show()

# 10. Plotting the loss over epochs
plt.figure(figsize=(10, 4))
plt.plot(losses, color='orange')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# 11. Predicting price for a new car age
car_age = float(input('Enter the age of car in years: '))
age_scaled = (car_age - X_mean[0]) / X_std[0]
age_squared_scaled = (car_age ** 2 - X_mean[1]) / X_std[1]
new_X_scaled = np.array([[age_scaled, age_squared_scaled]])

predicted_price_scaled = predict(new_X_scaled)
predicted_price = predicted_price_scaled * y_std + y_mean

if car_age > max_age or car_age < min_age:
    print("Warning: Prediction may be inaccurate (outside training range)")

else:
    print(f"Predicted price for a {car_age} year old car: ₹{predicted_price[0]:.2f}")

#Using scikit-learn for polynomial regression

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures, StandardScaler
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline

# # 1. Data: Car age vs price
# x = np.array([1, 2, 3, 4, 5, 6, 7]).reshape(-1, 1)
# y = np.array([800, 680, 540, 400, 300, 220, 160]).reshape(-1, 1)

# # 2. Create a pipeline with standardization + polynomial features + linear regression
# degree = 2
# model = make_pipeline(
#     StandardScaler(),                 # Scale features
#     PolynomialFeatures(degree),      # x, x^2
#     LinearRegression()               # Fit regression
# )

# # 3. Train the model
# model.fit(x, y)

# # 4. Predict values
# x_plot = np.linspace(0, 10, 100).reshape(-1, 1)
# y_pred = model.predict(x_plot)

# # 5. Visualize
# plt.figure(figsize=(8, 5))
# plt.scatter(x, y, color='red', label='Actual Data')
# plt.plot(x_plot, y_pred, color='blue', label=f'Polynomial Degree {degree}')
# plt.xlabel('Car Age (Years)')
# plt.ylabel('Price (₹)')
# plt.title('Polynomial Regression using scikit-learn')
# plt.legend()
# plt.grid(True)
# plt.show()

# # 6. Predict a new car's price
# new_age = float(input("Enter car age in years: "))
# predicted_price = model.predict(np.array([[new_age]]))[0][0]
# print(f"Predicted price for a {new_age} year old car: ₹{predicted_price:.2f}")
# Note: This code uses scikit-learn for polynomial regression, which simplifies the process significantly.




    