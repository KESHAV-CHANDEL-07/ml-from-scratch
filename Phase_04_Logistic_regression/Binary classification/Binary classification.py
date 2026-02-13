import numpy as np

# Inputs
X_raw = np.array([
    [2, 9],
    [4, 8],
    [6, 7],
    [8, 6],
    [10, 5]
], dtype=float)

# Targets (0 = Fail, 1 = Pass)
y = np.array([[0], [0], [1], [1], [1]])

print("Step 1: Raw X:")
print("Shape:", X_raw.shape)
print(X_raw)
input("\nPress Enter to continue...")

print("y (targets):")
print("Shape:", y.shape)
print(y)
input("\nPress Enter to continue...")

# Step 2: Standardize features
x_mean = np.mean(X_raw, axis=0)
x_std = np.std(X_raw, axis=0)
X = (X_raw - x_mean) / x_std

print("\nStep 2: Standardized X")
print("Mean:", x_mean)
print("Std:", x_std)
print("Shape:", X.shape)
print(X)
input("\nPress Enter to continue...")

np.random.seed(42)
w = np.random.randn(2)
b = np.random.randn()

print("\nStep 3: Initial Weights and Bias")
print("w:", w)  # shape (2,)
print("b:", b)  # scalar
input("\nPress Enter to continue...")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X_input):
    z = X_input @ w + b  # shape: (m,)
    return sigmoid(z) 

def compute_loss(y_true, y_pred):   #Binary cross entropy loss function 
    eps = 1e-15  # to prevent log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true.flatten() * np.log(y_pred) + (1 - y_true.flatten()) * np.log(1 - y_pred))

learning_rate = 0.1
epochs = 1000
losses = []

for i in range(epochs):
    y_pred = predict(X)
    loss = compute_loss(y, y_pred)
    losses.append(loss)

    error = y_pred - y.flatten()  # shape: (m,)

    # Gradients
    dw = X.T @ error / len(X)  # shape: (2,)
    db = np.mean(error)

    # Optional: Print at checkpoints
    if i % 200 == 0:
        print(f"\nEpoch {i}")
        print("Prediction:", y_pred)
        print("Error:", error)
        print("dw:", dw)
        print("db:", db)
        print("Loss:", loss)
        input("\nPress Enter to continue...")

    # Parameter update
    w -= learning_rate * dw
    b -= learning_rate * db

print("\nFinal weights:", w)
print("Final bias:", b)
print("Final loss:", losses[-1])

def predict_class(X_input):
    return (predict(X_input) >= 0.5).astype(int)

# Try predicting on training data
print("\nPredicted classes:", predict_class(X))
print("Actual classes:   ", y.flatten())
