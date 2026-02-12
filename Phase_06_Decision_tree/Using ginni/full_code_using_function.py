import numpy as np

# sample data
x = np.array([
    [1, 3],
    [2, 2],
    [3, 6],
    [4, 5]
])
y = np.array([0, 0, 1, 1])

# decision tree recursive builder
def build_tree(x, y, depth=0, max_depth=2):
    n_samples, n_features = x.shape
    
    best_gini = 1
    best_feature = None
    best_thresh = None
    
    for feature in range(n_features):
        thresholds = np.unique(x[:, feature])
        print(f"\n{'  '*depth}Checking feature {feature} with thresholds {thresholds}")
        
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
            
            print(f"{'  '*depth}  thresh {thresh}: gini_left {gini_left:.3f}, gini_right {gini_right:.3f}, weighted {weighted_gini:.3f}")
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_thresh = thresh
                
    if best_feature is None:
        print(f"{'  '*depth}No valid split at this depth")
        return
    
    print(f"\n{'  '*depth}Best split at depth {depth}: feature {best_feature}, threshold {best_thresh}, Gini {best_gini:.3f}")
    
    # split with best
    left_idx = x[:, best_feature] <= best_thresh
    right_idx = x[:, best_feature] > best_thresh
    
    x_left, y_left = x[left_idx], y[left_idx]
    x_right, y_right = x[right_idx], y[right_idx]
    
    # print left branch info
    p1_left = np.sum(y_left == 1) / len(y_left)
    p0_left = 1 - p1_left
    gini_left = 1 - (p1_left**2 + p0_left**2)
    if gini_left == 0 or depth + 1 == max_depth:
        print(f"{'  '*(depth+1)}Left branch is pure or max depth with labels {y_left}")
    else:
        build_tree(x_left, y_left, depth + 1, max_depth)
        
    # print right branch info
    p1_right = np.sum(y_right == 1) / len(y_right)
    p0_right = 1 - p1_right
    gini_right = 1 - (p1_right**2 + p0_right**2)
    if gini_right == 0 or depth + 1 == max_depth:
        print(f"{'  '*(depth+1)}Right branch is pure or max depth with labels {y_right}")
    else:
        build_tree(x_right, y_right, depth + 1, max_depth)

# start building the tree
build_tree(x, y)
