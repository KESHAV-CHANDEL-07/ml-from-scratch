\## ✅ Phase 5: Regularization — Ridge & Lasso Regression

This folder contains my implementation of **Ridge (L2)** and **Lasso (L1)** Regression completely from scratch, using only Python and NumPy.

---

### 🔹 Ridge Regression

**Use Case:** Predicting the price of a used car  
**Features used:**
- Engine size (liters)
- Mileage (km driven)
- Number of seats
- Age (years)

**What I learned:**
- How Ridge adds L2 penalty: `λ * sum(w²)`
- How it reduces overfitting by shrinking weights
- How changing λ (0, 1, 100) affects predictions and model complexity
- How to scale features and target, and unscale predictions

---

### 🔸 Lasso Regression

**Use Case:** Predicting a student's final exam score  
**Features used:**
- Study hours
- Sleep hours
- School rank
- Mobile usage
- Attended extra classes (yes/no)
- Junk food intake

**What I learned:**
- How Lasso adds L1 penalty: `λ * sum(|w|)`
- How it helps with feature selection by making some weights exactly zero
- Observed impact of different λ values (0, 0.1, 1, 10)

---

### ✅ Common to Both

- Full manual gradient descent
- Feature & target standardization
- Predictions rescaled to real-world values
- Multiple λ values tested for comparison
- Console output formatted cleanly
- No external ML libraries used (no scikit-learn!)

---

### 🧠 Why This Matters

Regularization is essential when:
- The model overfits
- Some features may be redundant
- You want simpler, generalizable models

Now I understand how both Ridge and Lasso control complexity — and when to use each.

---

