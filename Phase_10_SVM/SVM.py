import numpy as np
import matplotlib.pyplot as plt

# ---------------- Create a toy dataset ----------------
X = np.array([
    [2, 2],
    [2, 3],
    [3, 3],
    [6, 6],
    [7, 7],
    [8, 6]
])
y = np.array([-1, -1, -1, 1, 1, 1])   # labels: -1 (blue), +1 (red)

# ---------------- SVM from scratch ----------------
class SVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# ---------------- Train and Predict ----------------
svm = SVM(lr=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)
predictions = svm.predict(X)
print("Predictions:", predictions)

# ---------------- Accuracy Calculation ----------------
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ---------------- Visualize ----------------
def visualize_svm(X, y, model):
    def hyperplane(x, w, b, offset):
        return (-w[0] * x + b + offset) / w[1]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')

    ax = plt.gca()
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])

    x1_1 = hyperplane(x0_1, model.w, model.b, 0)
    x1_2 = hyperplane(x0_2, model.w, model.b, 0)

    x1_1_m = hyperplane(x0_1, model.w, model.b, -1)
    x1_2_m = hyperplane(x0_2, model.w, model.b, -1)

    x1_1_p = hyperplane(x0_1, model.w, model.b, 1)
    x1_2_p = hyperplane(x0_2, model.w, model.b, 1)

    plt.plot([x0_1, x0_2], [x1_1, x1_2], "y--")   # decision boundary
    plt.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "k") # margin -1
    plt.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "k") # margin +1

    plt.show()

visualize_svm(X, y, svm)
