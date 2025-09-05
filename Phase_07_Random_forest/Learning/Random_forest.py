import numpy as np

# decision tree from before
def build_tree(x, y, max_depth=2, depth=0):
    n_samples, n_features = x.shape

    if len(np.unique(y)) == 1:
        return int(y[0])
    if depth >= max_depth or len(y) < 2:
        return int(np.bincount(y).argmax())

    best_gini = 1
    best_feature = None
    best_thresh = None

    for feature in range(n_features):
        thresholds = np.unique(x[:, feature])
        for thresh in thresholds:
            left_idx = x[:, feature] <= thresh
            right_idx = x[:, feature] > thresh

            y_left = y[left_idx]
            y_right = y[right_idx]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            p1_left = np.sum(y_left == 1) / len(y_left)
            p0_left = 1 - p1_left
            gini_left = 1 - (p1_left**2 + p0_left**2)

            p1_right = np.sum(y_right == 1) / len(y_right)
            p0_right = 1 - p1_right
            gini_right = 1 - (p1_right**2 + p0_right**2)

            weighted_gini = (len(y_left)*gini_left + len(y_right)*gini_right) / len(y)

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_thresh = thresh

    if best_feature is None:
        return int(np.bincount(y).argmax())

    left_idx = x[:, best_feature] <= best_thresh
    right_idx = x[:, best_feature] > best_thresh

    x_left = x[left_idx]
    y_left = y[left_idx]
    x_right = x[right_idx]
    y_right = y[right_idx]

    left_subtree = build_tree(x_left, y_left, max_depth, depth+1)
    right_subtree = build_tree(x_right, y_right, max_depth, depth+1)

    return {
        "feature": best_feature,
        "thresh": best_thresh,
        "left": left_subtree,
        "right": right_subtree
    }

def predict_tree(tree, x_row):
    while isinstance(tree, dict):
        f = tree["feature"]
        t = tree["thresh"]
        if x_row[f] <= t:
            tree = tree["left"]
        else:
            tree = tree["right"]
    return tree

# bootstrap sampling
def bootstrap(x, y):
    n = len(x)
    idx = np.random.choice(n, size=n, replace=True)
    return x[idx], y[idx]

# random forest
def build_random_forest(x, y, n_trees=5, max_depth=2):
    forest = []
    for i in range(n_trees):
        x_s, y_s = bootstrap(x, y)
        tree = build_tree(x_s, y_s, max_depth=max_depth)
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
        counts = np.bincount(votes)
        final_preds.append(np.argmax(counts))
    return np.array(final_preds)

# sample data
x = np.array([
    [1, 3],
    [2, 2],
    [3, 6],
    [4, 5]
])
y = np.array([0, 0, 1, 1])

# train random forest
forest = build_random_forest(x, y, n_trees=5, max_depth=2)

# test
x_test = np.array([
    [1, 2],
    [3, 4],
    [6, 2]
])
preds = predict_random_forest(forest, x_test)

print("Predictions:", preds)
