import numpy as np

x=np.array([[90, 18],   # Low glucose, low BMI → likely not diabetic
            [120, 25],  # Normal range → not diabetic
            [150, 28],  # Pre-diabetic
            [180, 33],  # High glucose, moderate BMI → diabetic
            [200, 36],  # High glucose, high BMI → diabetic
            [220, 40]   # Very high glucose, very high BMI → diabetic
],dtype=float)

y=np.array([[0],[0],[1],[1],[1],[1]])  # 0: not diabetic, 1: diabetic

x_mean=np.mean(x, axis=0)
x_std=np.std(x,axis=0)
X_scaled= (x-x_mean)/x_std

np.random.seed(42)
w=np.random.randn(2)
b=np.random.randn()

def sigmoid(z):
    return 1/(1+np.exp(z))

def predict(x_input):
    return x_input @ w+ b

def compute_loss(y_predict , y):
    eps = 1e-15
    y_predict = np.clip(y_predict ,eps,1-eps)
    return -np.mean(y.flatten() * np.log(y_predict) + (1 - y.flatten()) * np.log(1 - y_predict))

learning_rate = 0.1
epoches = 1000
losses = []

for i in range(epoches):
    y_predict = predict(X_scaled)
    loss = compute_loss(y, y_predict)
    losses.append(loss)

    error = y_predict - y.flatten()

    # Gradients
    dw = X_scaled.T @ error / len(y)  # shape: (2,)
    db = np.mean(error)  # scalar
    if i % 200 == 0:
        print(f"\nEpoch {i}")
        print("Prediction:", y_predict)
        print("Error:", error)
        print("dw:", dw)
        print("db:", db)
        print("Loss:", loss)


    # Update weights and bias
    w -= learning_rate * dw
    b -= learning_rate * db
def predict_class(X_input):
    return (predict(X_input) >= 0.5).astype(int)


print("\nFinal weights:", w)
print("Final bias:", b)
print("Final loss:", losses[-1])

print("\nPredicted classes:", predict_class(X_scaled))
print("Actual classes:   ", y.flatten())


def standardize_input(glucose, bmi):
    x = np.array([[glucose, bmi]])
    return (x - x_mean) / x_std

def predict_user(glucose, bmi):
    x_std = standardize_input(glucose, bmi)
    pred = predict_class(x_std)
    if pred[0] == 1:
        print(" diabetic")
    else:
        print("non-diabetic")

try:
    print("\n--- Predict Patient Diabetes Status ---")
    g = float(input("Enter glucose level (e.g., 120): "))
    b = float(input("Enter BMI (e.g., 28): "))
    predict_user(g, b)
except ValueError:
    print("Invalid input. Please enter numeric values.")

