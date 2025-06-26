import numpy as np

# Features: [study hours, sleep hours, school rank, mobile usage, extra classes, junk food]
X_raw = np.array([
    [4, 7, 8, 2, 1, 3],
    [2, 6, 5, 6, 0, 5],
    [3, 8, 9, 1, 1, 2],
    [5, 6, 6, 4, 1, 6],
    [1, 5, 4, 7, 0, 7],
    [4, 7, 7, 2, 1, 2],
    [3, 7, 6, 3, 1, 3]
], dtype=float)

y_raw = np.array([[88], [62], [90], [70], [50], [85], [77]])  # Exam scores

X_mean = np.mean(X_raw, axis=0)
X_std = np.std(X_raw, axis=0)
X = (X_raw - X_mean) / X_std

y_mean = np.mean(y_raw)
y_std = np.std(y_raw)
y_scaled = (y_raw - y_mean) / y_std

def train_lasso(X, y_scaled, lambda_, learning_rate=0.01, epochs=1000):
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

lambdas = [0, 0.1, 1, 10]
results = {}

for lam in lambdas:
    w, b = train_lasso(X, y_scaled, lambda_=lam)
    y_pred_scaled = X @ w + b
    y_pred_unscaled = y_pred_scaled * y_std + y_mean
    results[lam] = y_pred_unscaled

# Print predictions
print("\nPredicted Exam Scores (Lasso):")
for i, x in enumerate(X_raw):
    formatted = ", ".join(f"{val:.2f}" for val in x)
    print(f"\nInput [{formatted}]:")
    for lam in lambdas:
        print(f"  λ = {lam:<4}: Predicted Score = {results[lam][i]:.2f}")

# Predict for a new student
X_test = np.array([[3, 7, 9, 3, 1, 3]])
X_test_scaled = (X_test - X_mean) / X_std

print("\nNew student input:", X_test[0])
for lam in lambdas:
    w, b = train_lasso(X, y_scaled, lambda_=lam)
    y_pred = X_test_scaled @ w + b
    y_final = y_pred * y_std + y_mean
    print(f"  λ = {lam:<4}: Predicted Final Score = {y_final[0]:.2f}")
