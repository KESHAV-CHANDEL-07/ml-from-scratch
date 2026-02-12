import numpy as np

def entropy(labels):
    classes,counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    ent = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    print(f"Entropy for labels {labels} = {ent:.4f}")
    return ent

def information_gain(parent_labels, left_labels, right_labels):
    parent_entropy = entropy(parent_labels)
    left_entropy = entropy(left_labels)
    right_entropy = entropy(right_labels)
    left_weight = len(left_labels) / len(parent_labels)
    right_weight = len(right_labels) / len(parent_labels)
    weighted_entropy = left_weight * left_entropy + right_weight * right_entropy
    ig = parent_entropy - weighted_entropy
    print(f"IG = {parent_entropy:.4f} - {weighted_entropy:.4f} = {ig:.4f}")
    return ig

def best_split(X, y, feature_indices):
    best_gain = 0
    best_feature = None
    best_splits = None

    print("\n== Testing all features ==")
    for feature in feature_indices:
        values = np.unique(X[:, feature])
        print(f" Feature {feature} values: {values}")
        for val in values:
            left_idx = X[:, feature] == val
            right_idx = X[:, feature] != val
            left_labels = y[left_idx]
            right_labels = y[right_idx]
            print(f"  Testing split: feature={feature} == {val}")
            print(f"   Left labels: {left_labels}")
            print(f"   Right labels: {right_labels}")
            gain = information_gain(y, left_labels, right_labels)
            print(f"   Gain for split feature={feature} == {val} : {gain:.4f}")
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_splits = (val, left_idx, right_idx)
                print(f"   => New best gain {best_gain:.4f} on feature {feature} == {val}")
                
    return best_feature, best_splits

def build_tree(X, y, feature_indices, depth=0):
    indent = "  " * depth
    print(f"{indent}Building tree at depth {depth}, labels={y}")
    if len(np.unique(y)) == 1:
        print(f"{indent}Pure node found with class {y[0]}")
        return {'type': 'leaf', 'class': y[0]}
    if len(feature_indices) == 0:
        majority = np.bincount(y).argmax()
        print(f"{indent}No features left, majority class is {majority}")
        return {'type': 'leaf', 'class': majority}

    best_feature, best_splits = best_split(X, y, feature_indices)
    if best_feature is None:
        majority = np.bincount(y).argmax()
        print(f"{indent}No good split, majority class is {majority}")
        return {'type': 'leaf', 'class': majority}

    val, left_idx, right_idx = best_splits
    print(f"{indent}Best split: feature {best_feature} == {val}")
    left_subtree = build_tree(X[left_idx], y[left_idx], feature_indices, depth+1)
    right_subtree = build_tree(X[right_idx], y[right_idx], feature_indices, depth+1)

    return {
        'type': 'node',
        'feature': best_feature,
        'value': val,
        'left': left_subtree,
        'right': right_subtree
    }

def predict(tree, sample):
    if tree['type'] == 'leaf':
        return tree['class']
    else:
        if sample[tree['feature']] == tree['value']:
            return predict(tree['left'], sample)
        else:
            return predict(tree['right'], sample)

# Training data

X = np.array([
    [0, 0, 0],  # Sunny, High, False
    [0, 0, 1],
    [1, 0, 0],
    [2, 1, 0],
    [2, 1, 1],
    [2, 0, 0],
    [1, 1, 1],
])
y = np.array([0, 0, 1, 1, 0, 1, 1])

#Test
# Build the decision tree
feature_indices = [0,1,2]
tree = build_tree(X, y, feature_indices)

test_data = [
    [0, 0, 1],
    [1, 0, 0],
    [2, 1, 0],
    [2, 0, 1]
]
for sample in test_data:
    result = predict(tree, sample)
    print(f"{sample} => PlayTennis={result}")
