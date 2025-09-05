import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1.2, 40000, 5, 5],
    [1.5, 25000, 5, 3],
    [2.0, 60000, 7, 6],
    [1.0, 10000, 4, 2],
    [1.3, 80000, 5, 7],
    [1.8, 30000, 5, 4],
    [2.2, 20000, 7, 3],
], dtype=float)

# Prices in lakhs (₹)
y = np.array([[4.5], [6.0], [5.5], [7.0], [3.8], [6.2], [6.8]])

X_mean = np.mean(X, axis = 0)
X_std = np.std(X, axis = 0)
X_scaled =(X -X_mean)/X_std

y_mean =np.mean(y)
y_std = np.std(y)
y_scaled = (y - y_mean) / y_std

np.random.seed(42)
w = np.random.randn(4)
b = np.random.randn(1)
