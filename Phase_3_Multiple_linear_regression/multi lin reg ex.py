import numpy as np
import matplotlib.pyplot as plt

# Training Data
study_hours = np.array([2, 4, 6, 8, 10]).reshape(-1, 1)
sleep_hours = np.array([8, 7, 6, 5, 4]).reshape(-1, 1)
marks = np.array([50, 60, 70, 80, 90]).reshape(-1, 1)

# Combine features into one matrix X
X = np.hstack((study_hours, sleep_hours))

# Standardize Features
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scaled = (X - X_mean) / X_std

# Standardize Target
y = marks
y_mean = np.mean(y)
y_std = np.std(y)
y_scaled = (y - y_mean) / y_std

# Initialize weights and bias
np.random.seed(42)
w = np.random.randn(2)
b = np.random.randn()

# Prediction Function
def predict(X_input):
    return X_input @ w + b

# Loss Function
def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Training
learning_rate = 0.01
epochs = 10000
for epoch in range(epochs):
    y_pred = predict(X_scaled)
    loss = compute_loss(y_scaled.flatten(), y_pred)
    error = y_scaled.flatten() - y_pred

    dw = -2 * X_scaled.T @ error / len(X_scaled)
    db = -2 * np.mean(error)

    w -= learning_rate * dw
    b -= learning_rate * db

# --------- Prediction ---------
input_study = float(input("Enter number of study hours: "))
input_sleep = float(input("Enter number of sleep hours: "))

# Scale new input
new_X = np.array([[input_study, input_sleep]])
new_X_scaled = (new_X - X_mean) / X_std

# Predict scaled -> unscale
predicted_scaled = predict(new_X_scaled)
predicted_mark = predicted_scaled * y_std + y_mean

if not (2 <= input_study <= 16) or not (2 <= input_sleep <= 8):
    print("⚠️ Warning: Input is outside training range. Prediction may be inaccurate.")
else:
    print(f"\nPredicted Marks: {predicted_mark[0]:.2f}")

# --------- Plotting ---------
# Predict on training data (to compare)
y_pred_train_scaled = predict(X_scaled)
y_pred_train = y_pred_train_scaled * y_std + y_mean

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(marks, label='Actual Marks', marker='o', color='blue')
plt.plot(y_pred_train, label='Predicted Marks', marker='x', linestyle='--', color='orange')
plt.plot(predicted_mark, label='Actual Marks', marker='*', color='blue')
plt.title("Actual vs Predicted Marks")
plt.xlabel("Sample Index")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 Target
target_marks = 100
fixed_sleep = 6

# Step 1: Standardize the fixed sleep hours
sleep_scaled = (fixed_sleep - X_mean[1]) / X_std[1]

# Step 2: Calculate scaled target output
target_scaled = (target_marks - y_mean) / y_std

# Step 3: Solve for required study_scaled
study_scaled = (target_scaled - (w[1] * sleep_scaled) - b) / w[0]

# Step 4: Convert study_scaled back to original hours
required_study_hours = study_scaled * X_std[0] + X_mean[0]

print(f"\n📘 To score 100 marks with {fixed_sleep} hours of sleep,")
print(f"🕒 You need to study for approximately {required_study_hours:.50f} hours.")
