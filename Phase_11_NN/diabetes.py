# =======================
# Professional Diabetes Prediction Pipeline
# =======================

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib

# Step 2: Load Dataset
df = pd.read_csv("diabetes.csv")

print("Dataset Shape:", df.shape)
print("Missing values per column:\n", df.isnull().sum())

# Step 3: Data Cleaning
# Replace impossible zeros with NaN for selected medical features
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Fill NaN with median (robust against outliers)
df.fillna(df.median(), inplace=True)

# Step 4: EDA (basic professional plots)
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

sns.countplot(x="Outcome", data=df, palette="Set2")
plt.title("Target Class Distribution (0=No Diabetes, 1=Diabetes)")
plt.show()

# Step 5: Features & Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 6: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Handle Class Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

print("Class balance after SMOTE:\n", pd.Series(y_res).value_counts())

# Step 8: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# Step 9: Define Models
log_model = LogisticRegression(max_iter=500)
rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Step 10: Hyperparameter Tuning (Random Forest example)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(rf_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
print("Best RF Params:", grid.best_params_)

# Step 11: Ensemble (Voting Classifier)
voting_model = VotingClassifier(
    estimators=[
        ("lr", log_model),
        ("rf", best_rf),
        ("xgb", xgb_model)
    ],
    voting="soft"
)

# Step 12: Training Models
models = {
    "Logistic Regression": log_model,
    "Random Forest (Tuned)": best_rf,
    "XGBoost": xgb_model,
    "Voting Ensemble": voting_model
}

# Step 13: Evaluation Function
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"\n=== {name} ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Confusion Matrix")
    plt.show()

# Step 14: Train, Evaluate & Compare
for name, model in models.items():
    model.fit(X_train, y_train)
    evaluate_model(name, model, X_test, y_test)

# Step 15: Save Best Model (Ensemble in this case)
joblib.dump(voting_model, "best_diabetes_model.pkl")
print("Best model saved as best_diabetes_model.pkl")
