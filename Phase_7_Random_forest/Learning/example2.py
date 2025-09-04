import numpy as np
import pandas as pd

# Load and clean the dataset
column_names = [
    "age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"
]

df = pd.read_csv(
    "heart.csv",
    na_values="?"
)

df.dropna(inplace=True)
df = df.astype(float)

X = df.drop("target", axis=1).values
y = df["target"].astype(int).values

# Manual Stratified Train-Test Split
def stratified_split(X, y, test_ratio=0.2):
    unique_classes = np.unique(y)
    X_train, y_train, X_test, y_test = [], [], [], []

    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - test_ratio))
        X_train.append(X[idx[:split]])
        y_train.append(y[idx[:split]])
        X_test.append(X[idx[split:]])
        y_test.append(y[idx[split:]])

    return (
        np.vstack(X_train),
        np.concatenate(y_train),
        np.vstack(X_test),
        np.concatenate(y_test)
    )

X_train, y_train, X_test, y_test = stratified_split(X, y, test_ratio=0.2)

# Decision Tree Functions (Gini based)
def gini(y):
    classes = np.unique(y)
    g = 1.0
    for c in classes:
        p = np.sum(y == c) / len(y)
        g -= p ** 2
    return g

def best_split(X, y):
    best_feature, best_thresh, best_gini = None, None, 1.0
    n_samples, n_features = X.shape

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for thresh in thresholds:
            left_mask = X[:, feature] <= thresh
            right_mask = X[:, feature] > thresh
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            y_left, y_right = y[left_mask], y[right_mask]
            g = (len(y_left)*gini(y_left) + len(y_right)*gini(y_right)) / len(y)
            if g < best_gini:
                best_gini = g
                best_feature = feature
                best_thresh = thresh

    return best_feature, best_thresh

def build_tree(X, y, max_depth=5, depth=0):
    if len(set(y)) == 1 or depth == max_depth:
        return int(np.bincount(y).argmax())

    feature, thresh = best_split(X, y)
    if feature is None:
        return int(np.bincount(y).argmax())

    left_idx = X[:, feature] <= thresh
    right_idx = X[:, feature] > thresh

    left_subtree = build_tree(X[left_idx], y[left_idx], max_depth, depth+1)
    right_subtree = build_tree(X[right_idx], y[right_idx], max_depth, depth+1)

    return {
        "feature": feature,
        "thresh": thresh,
        "left": left_subtree,
        "right": right_subtree
    }

def predict_tree(tree, x):
    while isinstance(tree, dict):
        if x[tree["feature"]] <= tree["thresh"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree

# Random Forest from Scratch
def bootstrap(X, y):
    idx = np.random.choice(len(X), size=len(X), replace=True)
    return X[idx], y[idx]

def build_forest(X, y, n_trees=10, max_depth=5):
    forest = []
    for _ in range(n_trees):
        X_s, y_s = bootstrap(X, y)
        tree = build_tree(X_s, y_s, max_depth)
        forest.append(tree)
    return forest

def predict_forest(forest, X):
    predictions = []
    for tree in forest:
        preds = np.array([predict_tree(tree, row) for row in X])
        predictions.append(preds)
    predictions = np.array(predictions)
    final_preds = []
    for i in range(X.shape[0]):
        vote = np.bincount(predictions[:, i], minlength=5)
        final_preds.append(np.argmax(vote))
    return np.array(final_preds)

# Train and Evaluate
forest = build_forest(X_train, y_train, n_trees=15, max_depth=5)
y_pred = predict_forest(forest, X_test)

# Accuracy
acc = np.mean(y_pred == y_test)
print("Accuracy:", round(acc, 4))

# Per-class metrics
def classification_report(y_true, y_pred):
    classes = np.unique(y_true)
    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        print(f"Class {cls} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}")

classification_report(y_test, y_pred)
