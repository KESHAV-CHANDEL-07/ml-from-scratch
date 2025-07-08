# 🌳 Random Forest Classifier — From Scratch vs Scikit-learn

This project demonstrates a complete **Random Forest Classifier built from scratch using Python and NumPy**, alongside a comparison with the `scikit-learn` implementation, applied on two variations of the **Heart Disease Dataset**.

---

## 🛠️ What We Built

- ✅ A **Decision Tree Classifier** using the **Gini Index**
- ✅ A **Random Forest Classifier** using:
  - Bootstrap sampling
  - Majority voting
- ✅ Manual **stratified train-test split** (no `sklearn`)
- ✅ Custom evaluation metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- ✅ **Tree visualization** using `graphviz` and `scikit-learn` (for interpretability)

---

## 🧠 Skills Demonstrated

- Data preprocessing and cleaning
- Manual implementation of ensemble algorithms
- Mathematical understanding of:
  - Gini impurity
  - Tree splitting logic
  - Bias-variance tradeoff
- Handling class imbalance
- Visual debugging with decision tree visualization
- Building interpretable ML from scratch
- Metric-based model evaluation
- Python scripting using NumPy, Pandas, Graphviz

---

## 📈 Datasets Used

### 1. 🧪 UCI Heart Disease Dataset (`processed.cleveland.data`)
- Multiclass targets: `0, 1, 2, 3, 4` indicating increasing severity.
- Imbalanced, with higher severity classes underrepresented.

### 2. ❤️ Kaggle’s Heart Dataset (`heart.csv`)
- Balanced binary classification: `0` (No Disease), `1` (Disease)
- Used for final model comparisons and visualization

---

## 📚 Learning Journey

### ✅ Phase 1: From Scratch
- Built `build_tree`, `predict_tree`, `build_random_forest`, and `predict_forest` from scratch using NumPy.
- Implemented majority voting + bootstrap sampling.

### ✅ Phase 2: Faced Class Imbalance
- UCI dataset had too few samples for classes `2`, `3`, and `4`.
- Precision and recall for higher classes dropped to near **0%**.
- Accuracy stagnated around **48–55%**.

### ✅ Phase 3: Hyperparameter Tuning
- Increased `max_depth` and number of trees (`n_trees`).
- Slight improvements seen but imbalance persisted.

### ✅ Phase 4: Binary Classification
- Converted target into:
  - `0` → No Disease
  - `1` → Disease (combining classes 1–4)
- Performance boosted:
  - Accuracy reached **~93%**
  - High precision and recall for both classes

### ✅ Phase 5: Better Dataset
- Switched to **Kaggle’s balanced heart.csv**
- Improved training consistency and multiclass handling

### ✅ Phase 6: Scikit-learn Benchmark
- Used `DecisionTreeClassifier` and `RandomForestClassifier` from `sklearn`
- Confirmed that accuracy from scratch and sklearn were comparable
- Visualized trees using `graphviz`

---

## 🌳 Tree Visualization

We used **Graphviz** and `sklearn.tree.export_graphviz()` to generate decision tree diagrams. These visualizations helped interpret model decisions with:

- Gini impurity at each node
- Feature splits and thresholds
- Class distributions
- Leaf node predictions

The tree was saved as `tree.pdf`.

---

### 🧡 Built with heart by **Keshav Chandel**
