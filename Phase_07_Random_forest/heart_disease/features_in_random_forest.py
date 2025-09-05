import numpy as np
import pandas as pd
from graphviz import Digraph

# Load dataset (heart.csv should have header row)
df = pd.read_csv(r"c:\Users\kesha\OneDrive\Desktop\Projects\ml-from-scratch\Phase_7_Random_forest\dataset\heart.csv")
df = df.dropna()
df = df.astype(float)

X = df.drop("target", axis=1).values
y = df["target"].astype(int).values

# Manual Stratified Split
def stratified_split(X, y, test_ratio=0.2):
    X_train, y_train, X_test, y_test = [], [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        np.random.shuffle(idx)
        split = int(len(idx) * (1 - test_ratio))
        X_train.append(X[idx[:split]])
        y_train.append(y[idx[:split]])
        X_test.append(X[idx[split:]])
        y_test.append(y[idx[split:]])
    return (
        np.vstack(X_train), np.concatenate(y_train),
        np.vstack(X_test), np.concatenate(y_test)
    )

X_train, y_train, X_test, y_test = stratified_split(X, y)

# Gini Function
def gini(y):
    classes = np.unique(y)
    return 1.0 - sum((np.sum(y == c) / len(y)) ** 2 for c in classes)

# Best Split Function
def best_split(X, y):
    best_feature, best_thresh, best_gini = None, None, 1.0
    for feature in range(X.shape[1]):
        for thresh in np.unique(X[:, feature]):
            left = y[X[:, feature] <= thresh]
            right = y[X[:, feature] > thresh]
            if len(left) == 0 or len(right) == 0:
                continue
            g = (len(left)*gini(left) + len(right)*gini(right)) / len(y)
            if g < best_gini:
                best_gini = g
                best_feature = feature
                best_thresh = thresh
    return best_feature, best_thresh

# Build Decision Tree
def build_tree(X, y, max_depth=5, depth=0):
    if len(set(y)) == 1 or depth == max_depth:
        return int(np.bincount(y).argmax())
    feature, thresh = best_split(X, y)
    if feature is None:
        return int(np.bincount(y).argmax())
    left = X[:, feature] <= thresh
    right = X[:, feature] > thresh
    return {
        "feature": feature,
        "thresh": thresh,
        "left": build_tree(X[left], y[left], max_depth, depth + 1),
        "right": build_tree(X[right], y[right], max_depth, depth + 1)
    }

# Tree Prediction
def predict_tree(tree, x):
    while isinstance(tree, dict):
        if x[tree["feature"]] <= tree["thresh"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree

# Random Forest

def bootstrap(X, y):
    idx = np.random.choice(len(X), len(X), replace=True)
    return X[idx], y[idx]

def build_forest(X, y, n_trees=10, max_depth=5):
    return [build_tree(*bootstrap(X, y), max_depth=max_depth) for _ in range(n_trees)]

def predict_forest(forest, X):
    all_preds = np.array([[predict_tree(tree, x) for x in X] for tree in forest])
    return np.apply_along_axis(lambda x: np.bincount(x, minlength=5).argmax(), axis=0, arr=all_preds)

# Accuracy and Report
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def classification_report(y_true, y_pred):
    for cls in np.unique(y_true):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1 = 2*p*r / (p + r + 1e-9)
        print(f"Class {cls} | Precision: {p:.2f} | Recall: {r:.2f} | F1: {f1:.2f}")

# Visualization

def visualize_tree(tree, feature_names, class_names, filename="scratch_tree", format="pdf"):
    dot = Digraph()
    node_id = [0]

    def add_nodes(node, parent_id=None, edge_label=""):
        curr_id = str(node_id[0])
        node_id[0] += 1
        if isinstance(node, dict):
            feat = feature_names[node["feature"]]
            label = f"{feat} <= {node['thresh']:.2f}"
            dot.node(curr_id, label, shape="box", style="filled", fillcolor="#e6f2ff")
            if parent_id is not None:
                dot.edge(parent_id, curr_id, label=edge_label)
            add_nodes(node["left"], curr_id, "True")
            add_nodes(node["right"], curr_id, "False")
        else:
            label = f"Class {class_names[node]}"
            dot.node(curr_id, label, shape="ellipse", style="filled", fillcolor="#d5f5e3")
            if parent_id is not None:
                dot.edge(parent_id, curr_id, label=edge_label)

    add_nodes(tree)
    dot.render(filename, format=format, cleanup=True)
    dot.view()

# Run everything
forest = build_forest(X_train, y_train, n_trees=10, max_depth=5)
y_pred = predict_forest(forest, X_test)
print("Accuracy:", round(accuracy(y_test, y_pred), 3))
classification_report(y_test, y_pred)

y_pred = predict_forest(forest, X_test)
print("Accuracy:", round(accuracy(y_test, y_pred), 3))
classification_report(y_test, y_pred)

# Confusion Matrix

import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight") 
plt.show() 


# # Visualize one tree
# feature_names = list(df.drop("target", axis=1).columns)
# class_names = sorted([str(c) for c in np.unique(y)])
# visualize_tree(forest[0], feature_names, class_names)