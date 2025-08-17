import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- Generate a bigger dataset ----------------
X, y = make_classification(n_samples=500, n_features=2, n_classes=2,
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Convert labels {0,1} to {-1, +1}
y = np.where(y == 0, -1, 1)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- SVM class (same as before) ----------------
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

# ---------------- Train & Evaluate ----------------
svm = SVM(lr=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X_train, y_train)

train_preds = svm.predict(X_train)
test_preds = svm.predict(X_test)

print("Train Accuracy:", accuracy_score(y_train, train_preds)*100)
print("Test Accuracy:", accuracy_score(y_test, test_preds)*100)

# ---------------- Plot Decision Boundary ----------------
def plot_decision_boundary(X, y, model):
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', s=20)

    # Create mesh grid
    x0 = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100)
    x1 = -(model.w[0]/model.w[1]) * x0 + model.b/model.w[1]

    # Margin lines
    margin_plus = -(model.w[0]/model.w[1]) * x0 + (model.b+1)/model.w[1]
    margin_minus = -(model.w[0]/model.w[1]) * x0 + (model.b-1)/model.w[1]

    plt.plot(x0, x1, 'k-', lw=2, label="Decision Boundary")
    plt.plot(x0, margin_plus, 'g--', label="Margin +1")
    plt.plot(x0, margin_minus, 'r--', label="Margin -1")

    plt.legend()
    plt.title("SVM Decision Boundary")
    plt.show()

plot_decision_boundary(X, y, svm)
