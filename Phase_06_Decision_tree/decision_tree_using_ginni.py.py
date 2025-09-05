import numpy as np

x = np.array([
    [1,3],
    [2,2],
    [3,6],
    [4,5]    
])

y=np.array([0,0,1,1])

max_depth=2
depth =0
n_samples,n_features =x.shape

best_gini =1
best_feature=None
best_thresh=None

for feature in range(n_features):
    thresholds = np.unique(x[:, feature])
    print(f"\nChecking feature {feature} with thresholds {thresholds}")

    for thresh in thresholds:
        left_idx = x[:,feature]  <= thresh
        right_idx = x[:, feature] > thresh 

        y_left = y[left_idx]
        y_right = y[right_idx]

        print (f" threshold :{thresh}")
        print (f"y_left:{y_left}")
        print (f"y_right:{y_right}")
        
        if len(y_left) ==0 or len(y_right) == 0:
            continue

        p1_left =np.sum(y_left == 1)/len(y_left)
        p0_left = 1 - p1_left
        gini_left =1 -(p1_left**2 +p0_left**2)

        p1_right = np.sum(y_right == 1)/len(y_right)
        p0_right = 1- p1_right
        gini_right =1 -(p1_right**2 + p0_right**2)

        weighted_gini = (len(y_left)*gini_left +len(y_right)*gini_right)/len(y)

        if weighted_gini < best_gini:
            best_gini = weighted_gini
            best_feature = feature
            best_thresh = thresh

# after finishing searching all features:
print("\nBest split chosen at this level:")
print(f"  Best feature index: {best_feature}")
print(f"  Best threshold: {best_thresh}")
print(f"  Best Gini: {best_gini:.3f}")      

left_idx = x[:,best_feature]  <= best_thresh
right_idx = x[:, best_feature] > best_thresh 

x_left = x[left_idx]
y_left = y[left_idx]
x_right = x[right_idx]
y_right = y[right_idx]

print("\nLeft branch:")
print(x_left)
print(y_left)

print("\nRight branch:")
print(x_right)
print(y_right)