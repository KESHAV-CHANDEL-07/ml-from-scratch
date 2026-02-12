import numpy as np

# Car dataset: [engine_size, mileage, seats, age]
X_raw = np.array([
    [1.2, 40000, 5, 5],
    [1.5, 25000, 5, 3],
    [2.0, 60000, 7, 6],
    [1.0, 10000, 4, 2],
    [1.3, 80000, 5, 7],
    [1.8, 30000, 5, 4],
    [2.2, 20000, 7, 3]
], dtype=float)

y_raw = np.array([[4.5], [6.0], [5.5], [7.0], [3.8], [6.2], [6.8]])  # in lakhs

# Standardize features and target
X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X = (X_raw - X_mean) / X_std

y_mean = np.mean(y_raw)
y_std = np.std(y_raw)
y_scaled = (y_raw - y_mean) / y_std

def train_lasso(X, y_scaled, lambda_, learning_rate=0.0001, epochs=1000):
    np.random.seed(42)
    w = np.random.randn(X.shape[1])
    b = np.random.randn()

    def predict(X_input):
        return X_input @ w + b

    for i in range(epochs):
        y_pred = predict(X)
        error = y_pred - y_scaled.flatten()

        dw = (X.T @ error) / len(X) + lambda_ * np.sign(w)
        db = np.mean(error)

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            loss = np.mean(error ** 2) + lambda_ * np.sum(np.abs(w))
            print(f"Epoch {i}, Loss: {loss:.4f}")

    return w, b

# Run for different lambda values
lambdas = [0, 0.1, 1, 10]
results = {}

for lam in lambdas:
    w, b = train_lasso(X, y_scaled, lambda_=lam)
    y_pred_scaled = X @ w + b
    y_pred_unscaled = y_pred_scaled * y_std + y_mean
    results[lam] = y_pred_unscaled

# Print predictions
print("\nLasso Predictions (in lakhs):")
for i, x in enumerate(X_raw):
    formatted = ", ".join(f"{val:.2f}" for val in x)
    print(f"\nInput [{formatted}]:")
    for lam in lambdas:
        print(f"  λ = {lam:<4}: Predicted Price = {results[lam][i]:.2f} lakhs")
