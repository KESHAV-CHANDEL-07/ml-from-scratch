import numpy as np

x = np.array([
    [90, 18],
    [120, 25],
    [150, 28],
    [180, 33],
    [200, 36],
    [220, 40]
], dtype=float)

y = np.array([[0], [0], [1], [1], [1], [1]])

x_mean = np.mean(x, axis=0)
x_std = np.std(x, axis=0)
X_scaled = (x - x_mean) / x_std

np.random.seed(42)
w = np.random.randn(2)
b = np.random.randn()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(x_input):
    z = x_input @ w + b
    return sigmoid(z)

def compute_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true.flatten() * np.log(y_pred) + (1 - y_true.flatten()) * np.log(1 - y_pred))

def predict_class(X_input):
    return (predict(X_input) >= 0.5).astype(int)

learning_rate = 0.1
epochs = 1000
losses = []

for i in range(epochs):
    y_pred = predict(X_scaled)
    loss = compute_loss(y_pred, y)
    losses.append(loss)

    error = y_pred - y.flatten()
    dw = X_scaled.T @ error / len(y)
    db = np.mean(error)

    w -= learning_rate * dw
    b -= learning_rate * db

print("Final weights:", w)
print("Final bias:", b)
print("Final loss:", losses[-1])

y_pred_class = predict_class(X_scaled)
print("Predicted classes:", y_pred_class)
print("Actual classes:   ", y.flatten())

# Accuracy calculation
def compute_accuracy(y_true, y_pred_class):
    y_true = y_true.flatten()
    correct = np.sum(y_true == y_pred_class)
    return correct / len(y_true)

accuracy = compute_accuracy(y, y_pred_class)
print("Accuracy:", accuracy * 100, "%")

# Confusion matrix
def compute_confusion_matrix(y_true, y_pred_class):
    y_true = y_true.flatten()
    y_pred_class = y_pred_class.flatten()

    TP = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred_class[i] == 1:
         TP += 1

    TN = 0
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred_class[i] == 0:
            TN += 1
    FP = np.sum((y_true == 0) & (y_pred_class == 1))
    FN = np.sum((y_true == 1) & (y_pred_class == 0))             


    return TP, TN, FP, FN

TP, TN, FP, FN = compute_confusion_matrix(y, y_pred_class)
print("Confusion Matrix")
print("-----------------")
print(f"TP (True Positives):   {TP}")
print(f"TN (True Negatives):   {TN}")
print(f"FP (False Positives):  {FP}")
print(f"FN (False Negatives):  {FN}")

def standardize_input(glucose, bmi):
    x = np.array([[glucose, bmi]])
    return (x - x_mean) / x_std

def predict_user(glucose, bmi):
    x_std = standardize_input(glucose, bmi)
    pred = predict_class(x_std)
    if pred[0] == 1:
        print("diabetic")
    else:
        print("non-diabetic")
   
print("\nPredict Patient Diabetes Status")
g = float(input("Enter glucose level (e.g., 150): "))
b_input = float(input("Enter BMI (e.g., 30): "))
if(40 <= g <= 300) and (10 <= b_input <= 70):
    predict_user(g, b_input) 
else:
     print("Warning:glucose or BMI value seems unrealistic.") 