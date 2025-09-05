import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_distances

# ---------------- Load dataset ----------------
df = pd.read_csv("spam.csv", encoding='latin1')
df = df[['v1', 'v2']]  # keep only label & message
df = df.rename(columns={'v1':'label', 'v2':'message'})

# Drop NaN and duplicates, map labels
df = df.dropna(subset=['message', 'label']).drop_duplicates()
label_map = {'ham': 0, 'spam': 1}
df['y'] = df['label'].map(label_map)

# Split into train and test
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df['message'], df['y'],
    test_size=0.2, random_state=42, stratify=df['y']
)

# ---------------- TF-IDF Vectorization ----------------
vectorizer = TfidfVectorizer(lowercase=True, stop_words='english',
                             ngram_range=(1,2), min_df=2)
X_train = vectorizer.fit_transform(X_train_text).toarray()
X_test  = vectorizer.transform(X_test_text).toarray()

# ---------------- KNN Implementation with cosine similarity ----------------
class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        # Use cosine distance
        distances = cosine_distances(self.X_train, x.reshape(1, -1)).ravel()
        k_idx = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_idx]
        counts = Counter(k_labels)
        max_votes = max(counts.values())
        winners = [label for label, c in counts.items() if c == max_votes]
        return min(winners)  # tie-break: smaller label wins

# ---------------- Train & Test ----------------
knn = KNN(k=7)  # start with a small k
knn.fit(X_train, y_train)

# Predictions on test set
y_pred = knn.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"Test Accuracy: {acc:.3f}")

# Evaluation
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))

# ---------------- Predict a new message ----------------
msg = "Hey, are we still meeting today?"
X_new = vectorizer.transform([msg]).toarray()
pred = knn.predict(X_new)[0]
print("Prediction for new message:", 'spam' if pred==1 else 'ham')

# ---------------- Hyperparameter tuning ----------------
best_k, best_acc = None, -1
for k in [1,3,5,7,9,11,15,20,25,30]:
    model = KNN(k=k)
    model.fit(X_train, y_train)
    acc_k = (model.predict(X_test) == y_test).mean()
    print(f"k={k}: acc={acc_k:.3f}")
    if acc_k > best_acc:
        best_k, best_acc = k, acc_k

print(f"\nBest k={best_k} with test accuracy={best_acc:.3f}")
