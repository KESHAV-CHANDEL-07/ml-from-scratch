import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
y_actual= x**2

np.random.seed(42)

w=np.random.randn()*0.01
b=np.random.randn()*0.01
y_predict=w*x+b

def predict(x):
    return w*x+b

def compute_loss(y_actual, y_predict):
    return np.mean((y_actual - y_predict) ** 2)

learning_rate = 0.01
epochs = 1000
losses = []

for i in range(epochs):
    y_predict= predict(x)
    loss=compute_loss(y_actual,y_predict)
    losses.append(loss)

    dw = -2*np.mean(x*(y_actual - y_predict))
    db = -2*np.mean(y_actual - y_predict)
    w -= learning_rate*dw
    b -= learning_rate*db

print(f"leearned weight:, w= {w:.4f},b={b:.4f}")
print(f"Final loss: ={loss: .6f}")

plt.figure(figsize=(8,5))
plt.plot(x, y_actual, label='Actual', color='blue')
plt.plot(x, y_predict, label='Predicted', color='red',alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(losses, label='Loss', color='green')
plt.xlabel("epochs")
plt.ylabel('loss')
plt.grid(True)
plt.show()