import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter


df = pd.read_csv("diabetes_cleaned.csv")

columns_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in columns_with_zeros:
    median_val = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median_val)

X = df.drop("Outcome", axis=1).values
y = df["Outcome"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def knn_predict(X_train, y_train, x_test, k=5):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    
    # Sort by distance and get k nearest
    k_neighbors = sorted(distances, key=lambda x: x[0])[:k]
    
    # Get labels and do majority vote
    k_labels = [label for _, label in k_neighbors]
    most_common = Counter(k_labels).most_common(1)
    
    return most_common[0][0]


def predict_all(X_train, y_train, X_test, k=5):
    predictions = []
    for i in range(len(X_test)):
        pred = knn_predict(X_train, y_train, X_test[i], k)
        predictions.append(pred)
    return np.array(predictions)


y_pred = predict_all(X_train, y_train, X_test, k=5)
accuracy = np.mean(y_pred == y_test)*100

print(f"Handmade KNN Accuracy: {accuracy:.4f}")
