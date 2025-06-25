import numpy as np
import matplotlib.pyplot as plt
x= np.array([150, 160, 170, 180, 190]).reshape(-1, 1)
y_actual =np.array([50,56,63,70,77]).reshape(-1,1)

np.random.seed(42)
w=np.random.randn()*0.01
b=np.random.randn()*0.01

def predict(x):
    return w * x + b

def compute_loss(y_actual,y_predict):
    return np.mean((y_actual -y_predict) **2)
learning_rate = 0.01
epochs = 100
losses = []

for i in range(epochs):
    y_predict=predict(x)
    loss= compute_loss(y_actual,y_predict)
    losses.append(loss)

    dw = -2 * np.mean(x * (y_actual - y_predict))
    db = -2 * np.mean(y_actual - y_predict)
    w -= learning_rate*dw
    b -= learning_rate*db
print(f"learned weight: w={w:.4f}, b={b:.4f}")
print(f"final loss: {losses[-1]:.4f}")

plt.figure(figsize=(8,5))
plt.scatter(x,y_actual,label='Actual Data',color='blue')
plt.plot(x,y_predict,label='Predicted Data',color='red')
plt.title('Height vs weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (Kg)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Loss over epochs')
plt.legend()
plt.grid(True)
plt.show()