import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns

# Column names for UCI Cleveland dataset
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Load and clean data
df = pd.read_csv("processed.cleveland.data", header=None, names=column_names, na_values="?")
df_clean = df.dropna()

# Convert multi-class target to binary: 0 = no disease, 1 = disease
df_clean['target'] = df_clean['target'].apply(lambda x: 1 if x > 0 else 0)

# Features and labels
X = df_clean.drop('target', axis=1)
y = df_clean['target']

# Visualize class distribution
sns.countplot(x=y)
plt.title("Binary Class Distribution (0 = No Disease, 1 = Disease)")
plt.show()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Train Decision Tree
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("🌳 Decision Tree")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt, zero_division=0))

# ✅ Visualize the tree
dot_data = export_graphviz(
    dt,
    out_file=None,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("heart_disease_tree_binary", format="png", cleanup=True)
graph.view()

# ✅ Train Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n🌲 Random Forest")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, zero_division=0))

# ✅ Feature Importance (Random Forest)
importances = rf.feature_importances_
plt.figure(figsize=(10,6))
plt.barh(X.columns, importances)
plt.title("Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
