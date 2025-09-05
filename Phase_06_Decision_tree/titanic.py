import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- Load Data ----------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ---------------- Feature Engineering ----------------
# Extract Title from Name
train["Title"] = train["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
test["Title"] = test["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

# Simplify rare titles
title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Countess": "Rare", "Lady": "Rare", "Sir": "Rare", 
    "Jonkheer": "Rare", "Don": "Rare", "Dona": "Rare", "Rev": "Rare",
    "Col": "Rare", "Major": "Rare", "Capt": "Rare", "Dr": "Rare"
}
train["Title"] = train["Title"].replace(title_map)
test["Title"] = test["Title"].replace(title_map)

# Fill missing Age with median of Title
for df in [train, test]:
    df["Age"] = df.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

# Fill missing Fare with median
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# Fill missing Embarked with mode
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

# Convert categorical variables
for df in [train, test]:
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    df["Title"] = df["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4})

# ---------------- Select Features ----------------
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title"]
X = train[features]
y = train["Survived"]

# ---------------- Train Model ----------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# ---------------- Check Accuracy ----------------
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))

# ---------------- Predict on Test Data ----------------
test_X = test[features]
predictions = model.predict(test_X)

# ---------------- Save Submission ----------------
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
submission.to_csv("submission1.csv", index=False)
print("✅ submission.csv created for Kaggle upload!")
