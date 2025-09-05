import numpy as np

# Realistic used car dataset
# Features: [engine_size (L), mileage (km), seats, age (years)]
X_raw = np.array([
    [1.2, 40000, 5, 5],
    [1.5, 25000, 5, 3],
    [2.0, 60000, 7, 6],
    [1.0, 10000, 4, 2],
    [1.3, 80000, 5, 7],
    [1.8, 30000, 5, 4],
    [2.2, 20000, 7, 3]
], dtype=float)

y_raw = np.array([[4.5], [6.0], [5.5], [7.0], [3.8], [6.2], [6.8]])  # Prices in lakhs

X_test = np.array([
    [0.8, 18000, 5, 15],
    [2.0, 90000, 7, 6]
], dtype=float)

# Standardize features
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X = (X_raw - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Standardize target
y_mean = np.mean(y_raw)
y_std = np.std(y_raw)
y_scaled = (y_raw - y_mean) / y_std

def train_ridge(lambda_, learning_rate=0.0001, epochs=1000):
    np.random.seed(42)
    w = np.random.randn(X.shape[1])
    b = np.random.randn()

    def predict(X_input):
        return X_input @ w + b

    for _ in range(epochs):
        y_pred = predict(X)
        error = y_pred - y_scaled.flatten()
        dw = (X.T @ error) / len(X) + 2 * lambda_ * w
        db = np.mean(error)
        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b

# Train and predict
lambdas = [0, 1, 100]
results = {}

for lam in lambdas:
    w, b = train_ridge(lambda_=lam)
    y_test_scaled = X_test_scaled @ w + b
    y_test_unscaled = y_test_scaled * y_std + y_mean
    results[lam] = y_test_unscaled

# Print predictions
print("\nPredicted Prices for Test Cars (in lakhs):")
for i, x in enumerate(X_test):
    formatted = ", ".join(f"{v:.2f}" for v in x)
    print(f"\nInput [{formatted}]:")
    for lam in lambdas:
        print(f"  λ = {lam:<3}: Predicted Price = {results[lam][i]:.2f} lakhs")