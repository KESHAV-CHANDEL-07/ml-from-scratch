# 📘 Phase 3: Multiple Linear Regression (From Scratch)

In this phase, I implemented **multiple linear regression from scratch** using both **2-feature** and **4-feature** models. My goal was to fully understand how linear regression extends to multivariate cases using **matrix math**, and how features interact to affect predictions.

---

## 🧠 What I Learned

- How to work with input data shaped as a matrix (m × n)
- How to **add a bias column** (intercept term) manually
- **Matrix multiplication** for batch predictions: `y = X @ theta`
- How to compute cost (Mean Squared Error) with matrices
- How to calculate **gradients for all weights and bias at once**
- How to update weights using **gradient descent**
- How standardization affects training
- How multiple features together influence predictions (e.g. house size + bedrooms)

---

## 📂 What’s Included

| File | Description |
|------|-------------|
| `multiple_regression_2f.py` | Model using 2 features (e.g., house size, bedrooms) |
| `multiple_regression_4f.py` | Model using 4 features (e.g., size, rooms, age, distance) |
| `README.md` | Explanation of my learning, approach, and results |

---

## 📊 Techniques Used

- Feature matrix construction with `np.hstack()` and `np.c_[]`
- Manual standardization:  
  \[
  x_{scaled} = \frac{x - \mu}{\sigma}
  \]
- Cost function:  
  \[
  J(\theta) = \frac{1}{m} \sum (y - X\theta)^2
  \]
- Gradients:  
  \[
  \nabla_\theta = \frac{-2}{m} X^T (y - \hat{y})
  \]
- Batch gradient descent

---

## 🧪 Results

- Successfully trained and predicted outputs on small handcrafted datasets
- Observed how increasing one feature while keeping others fixed affects `y`
- Trained both 2-feature and 4-feature models and printed every matrix step-by-step

---

## 🧰 Tools Used

- Python 3
- NumPy
- Matplotlib (used in optional visualizations)

---

## 🚀 What’s Next

- 📌 Phase 4: Logistic Regression (classification)
- 🧮 Ridge/Lasso (Regularization)
- 🔍 Real dataset comparison with `scikit-learn`

---

> This work is part of my personal journey to learn **machine learning from scratch** — focusing on the math, not shortcuts. Every matrix, loss, gradient, and weight was printed and understood.
