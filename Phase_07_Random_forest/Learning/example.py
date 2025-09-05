column_names = [
    "age",         # age in years
    "sex",         # 1 = male; 0 = female
    "cp",          # chest pain type (4 values)
    "trestbps",    # resting blood pressure
    "chol",        # serum cholesterol
    "fbs",         # fasting blood sugar
    "restecg",     # resting electrocardiographic results
    "thalach",     # maximum heart rate achieved
    "exang",       # exercise induced angina
    "oldpeak",     # ST depression
    "slope",       # slope of peak exercise ST segment
    "ca",          # number of major vessels (0–3) colored by fluoroscopy
    "thal",        # 3 = normal; 6 = fixed defect; 7 = reversible defect
    "target",      # 0 = no disease, 1 = disease
]
import pandas as pd

df = pd.read_csv(
    "processed.cleveland.data",
    header=None,
    names=column_names,
    na_values="?"
)

print(df.head())
print(df.info())

df_clean = df.dropna()
print(df_clean.shape)
print(df_clean['target'].value_counts())
print(df_clean.isnull().sum())

X = df_clean.drop('target', axis=1)  # all columns except target
y = df_clean['target']               # the target column (values: 0,1,2,3,4)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # keeps class distribution balanced in both sets
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# create and train the model
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# make predictions
y_pred = clf.predict(X_test)

# evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.tree import export_graphviz
import graphviz

dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=X.columns,
    class_names=[str(c) for c in sorted(y.unique())],
    filled=True,
    rounded=True,
    special_characters=True
)
graph = graphviz.Source(dot_data)
graph.render("heart_disease_tree", format="png", cleanup=True)
graph.view()
