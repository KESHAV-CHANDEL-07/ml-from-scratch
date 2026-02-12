import numpy as np

# Training data
X_raw = np.array([
    [1000, 2],
    [1200, 3],
    [1500, 3],
    [1800, 4],
    [2000, 4],
    [2200, 5],
], dtype=float)

y_raw = np.array([[30], [35], [45], [55], [60], [70]])

# Test data (unseen)
X_test = np.array([
    [1600, 3],
    [1900, 4],
    [2100, 5]
], dtype=float)

# Standardize X
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X = (X_raw - X_mean) / X_std
X_test_scaled = (X_test - X_mean) / X_std

# Standardize y
y_mean = np.mean(y_raw)
y_std = np.std(y_raw)
y_scaled = (y_raw - y_mean) / y_std

# Store predictions
results = {}

# Training function
def train_ridge(lambda_, learning_rate=0.0001, epochs=1000):
    np.random.seed(42)
    w = np.random.randn(2)
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

# Train and predict for different lambdas
lambdas = [0, 1, 100]
for lam in lambdas:
    w, b = train_ridge(lambda_=lam)
    y_test_scaled = X_test_scaled @ w + b
    y_test_unscaled = y_test_scaled * y_std + y_mean
    results[lam] = y_test_unscaled

# Print final predictions
print("\nPredictions on New Test Data (in lakhs):")
for i, x in enumerate(X_test):
    print(f"\nInput {x}:")
    for lam in lambdas:
        print(f"  λ = {lam:<3}: Predicted Price = {results[lam][i]:.2f} lakhs")
