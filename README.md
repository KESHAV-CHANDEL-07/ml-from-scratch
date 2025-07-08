# 🤖 Machine Learning From Scratch — by Keshav Chandel

Welcome to my personal ML journey where I build everything from **scratch using only Python and NumPy**, focusing on **deep mathematical understanding**, **matrix operations**, and **step-by-step logic** — not just using prebuilt libraries.

---

## 📚 What I've Covered So Far

### ✅ Phase 1: Simple Linear Regression

* Implemented 1D linear regression from scratch.
* Learned cost function (MSE), gradient descent, and matrix dot products.
* Manually predicted new values and visualized loss reduction.

📁 Folder: `Phase_1_Simple_Regression/`

---

### ✅ Phase 2: Polynomial Regression

* Extended linear regression to fit **nonlinear data** using polynomial terms (x², x³, ...).
* Observed overfitting vs underfitting.
* Trained and predicted on curved data manually.

📁 Folder: `Phase_2_Polynomial_Regression/`

---

### ✅ Phase 3: Multiple Linear Regression

* Added support for multiple input features (2-feature and 4-feature models).
* Calculated cost, gradients, and updates using **matrix math**.
* Used real-world-like data (e.g., house prices) to predict with multiple variables.

📁 Folder: `Phase_3_Multiple_Linear_Regression/`

---

### ✅ Phase 4: Logistic Regression

* Implemented binary classification using logistic regression from scratch.
* Applied sigmoid activation, binary cross-entropy loss, and matrix-based gradients.
* Built a diabetes prediction model with user input.
* Evaluated with accuracy and confusion matrix.
* Added input validation to prevent wrong medical predictions.

📁 Folder: `Phase_4_Logistic_Regression/`

---

### ✅ Phase 5: Regularization (Ridge & Lasso)

* Implemented both Ridge (L2) and Lasso (L1) Regression from scratch.
* Applied Ridge on a used car price prediction dataset with 4 features.
* Applied Lasso on a student performance prediction dataset with academic & lifestyle inputs.
* Compared predictions across different λ values (0, 0.1, 1, 10).
* Observed how Ridge reduces overfitting by shrinking weights.
* Observed how Lasso performs feature selection by driving some weights to zero.

📁 Folder: `Phase_5_Regularization/`

---

### ✅ Phase 6: Decision Tree Classifier

* Implemented a decision tree classifier fully from scratch with Python & NumPy.
* Calculated **both entropy and Gini index** to measure uncertainty.
* Selected best features to split using a simple greedy algorithm.
* Handled categorical variables with binary splits.
* Visualized each calculation step with detailed printed output.
* Prepared for extending to random forests next.

📁 Folder: `Phase_6_Decision_Tree/`

---

### ✅ Phase 7: Random Forest Classifier

* Built a full **Random Forest Classifier from scratch** using NumPy.
* Used **bootstrap sampling + majority voting**.
* Implemented manual **stratified train-test split**.
* Evaluated with custom accuracy, precision, recall, and F1-score functions.
* Tried on **two versions** of heart disease datasets:
  - **Multiclass (0–4)** → struggled due to class imbalance
  - **Binary (0/1)** → high accuracy (~93%)
* Compared performance with `sklearn`'s `RandomForestClassifier` and `DecisionTreeClassifier`.
* Used `graphviz` to visualize the decision tree as a colorful PDF.
* Logged full training pipeline in a clean, reproducible script.

📁 Folder: `Phase_7_Random_Forest/`

---

## 🧠 Why I'm Doing This

I want to:

* Master the **core math and matrix logic** behind ML
* Build a **strong foundation for future advanced models**
* Apply this in **real-world hardware-based ML projects**
* Become internship/job-ready with a **hands-on + deep understanding** profile

---

## 🧰 Tools Used

* Python 3
* NumPy
* Pandas (for preprocessing only)
* Matplotlib (only for optional plots)
* Graphviz (for decision tree rendering)
* 💡 No scikit-learn or ML libraries for core logic — everything is done manually!

---

## 🚀 What's Next

* Phase 8: K-Nearest Neighbors (KNN) from scratch
* Phase 9: Naive Bayes Classifier
* Phase 10: Neural Network from scratch (using only NumPy)

---

## 🌐 About Me

**Name:** Keshav Chandel  
**Email:** [23bec053@nith.ac.in](mailto:23bec053@nith.ac.in)  
**Passion:** Combining Machine Learning with hardware (Raspberry Pi, Arduino, IoT)  
**Goal:** Internship-ready profile with projects that show real ML skill

---

Thanks for checking out my journey 🚀  
**Built with ❤️ by Keshav Chandel**
