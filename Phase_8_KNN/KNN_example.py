# knn_classifier.py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
k = 3
class KNNclassifier:
    def __init__(self,k):
        self.k = k

    def fit(self,x,y):
        self.X_train=x
        self.y_train=y
    def predict(self,X):
        prediction = [self._predict_one(x) for x in X]
        return np.array(prediction)
    def _predict_one(self,x):
        #calculate eucledian distance
        distance = np.sqrt(np.sum((self.X_train - x)**2 ,axis=1))
        k_indices = np.argsort(distance)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
    

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=100, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, random_state=42    
    )
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit model
    model = KNNclassifier(k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = np.mean(predictions == y_test)*100
    print(f"Accuracy: {accuracy:.2f}")

    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=20)
    plt.title(f"KNN Decision Boundary (k={model.k})")
    plt.show()
    
