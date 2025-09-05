import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Load and clean the data
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df = pd.read_csv("processed.cleveland.data", header=None, names=column_names, na_values="?")
df_clean = df.dropna()

X = df_clean.drop("target", axis=1).values.astype(float)
y = df_clean["target"].values.astype(int)

# 2. Stratified Split using sklearn
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Decision Tree from Scratch
def build_tree(x, y, max_depth=5, depth=0):
    n_samples, n_features = x.shape
    if len(np.unique(y)) == 1:
        return int(y[0])
    if depth >= max_depth or len(y) < 2:
        return int(np.bincount(y).argmax())

    best_gini = 1
    best_feature, best_thresh = None, None

    for feature in range(n_features):
        thresholds = np.unique(x[:, feature])
        for thresh in thresholds:
            left_idx = x[:, feature] <= thresh
            right_idx = x[:, feature] > thresh

            y_left, y_right = y[left_idx], y[right_idx]
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            def gini(y_part):
                p = np.bincount(y_part, minlength=5) / len(y_part)
                return 1 - np.sum(p**2)

            weighted_gini = (len(y_left)*gini(y_left) + len(y_right)*gini(y_right)) / len(y)
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_thresh = thresh

    if best_feature is None:
        return int(np.bincount(y).argmax())

    left_idx = x[:, best_feature] <= best_thresh
    right_idx = x[:, best_feature] > best_thresh

    left_subtree = build_tree(x[left_idx], y[left_idx], max_depth, depth+1)
    right_subtree = build_tree(x[right_idx], y[right_idx], max_depth, depth+1)

    return {"feature": best_feature, "thresh": best_thresh,
            "left": left_subtree, "right": right_subtree}

def predict_tree(tree, x_row):
    while isinstance(tree, dict):
        if x_row[tree["feature"]] <= tree["thresh"]:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree

def bootstrap(x, y):
    n = len(x)
    idx = np.random.choice(n, size=n, replace=True)
    return x[idx], y[idx]

def build_random_forest(x, y, n_trees=5, max_depth=5):
    forest = []
    for _ in range(n_trees):
        x_s, y_s = bootstrap(x, y)
        tree = build_tree(x_s, y_s, max_depth)
        forest.append(tree)
    return forest

def predict_random_forest(forest, x_test):
    predictions = []
    for tree in forest:
        preds = [predict_tree(tree, row) for row in x_test]
        predictions.append(preds)
    predictions = np.array(predictions)
    final_preds = []
    for i in range(len(x_test)):
        votes = predictions[:, i]
        final_preds.append(np.bincount(votes).argmax())
    return np.array(final_preds)

# 4. Train your Random Forest
forest = build_random_forest(X_train, y_train, n_trees=10, max_depth=5)

# 5. Evaluate
y_pred = predict_random_forest(forest, X_test)

print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 6. Feature Importance (approximate based on tree splits)
def feature_importance(forest, n_features):
    importance = np.zeros(n_features)

    def count_splits(tree):
        if isinstance(tree, dict):
            importance[tree["feature"]] += 1
            count_splits(tree["left"])
            count_splits(tree["right"])

    for tree in forest:
        count_splits(tree)
    
    importance = importance / np.sum(importance)  # normalize
    return importance

fi = feature_importance(forest, X.shape[1])
print("\n🔥 Feature Importances:")
for name, score in zip(column_names[:-1], fi):
    print(f"{name:12}: {score:.4f}")
