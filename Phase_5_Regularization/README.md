## ✅ Lasso Regression — Student Performance Prediction

This project applies Lasso Regression (L1 Regularization) to predict a student's final exam score based on academic and lifestyle factors.

### 📊 Features Used

- **study_hours**: Average study hours per day
- **sleep_hours**: Average sleep hours per night
- **school_rank**: Rating of the school (1–10)
- **mobile_usage**: Daily mobile usage in hours
- **extra_classes**: Whether the student attended extra coaching (0 or 1)
- **junk_food**: Number of junk food meals per week

---

### 🎯 Target

- **Final exam score** (out of 100)

---

### 🔍 Why Lasso?

Lasso (L1) regularization helps:
- Automatically shrink irrelevant or weak features to **zero**
- Perform **feature selection** while training
- Create a **sparser**, more interpretable model

This is useful when some lifestyle habits may not actually impact performance.

---

### 🧠 Workflow

- Data was standardized (both features and target)
- Lasso loss = MSE + λ * sum(abs(weights))
- Trained models for λ = 0, 0.1, 1, 10
- Compared predictions for all students
- Predicted final score for a new student using all trained models

---

### 📈 Insights

- As λ increases, Lasso **shrinks less important weights**
- With higher λ values, the model focuses only on the **strongest features**
- Helps prevent overfitting in small, noisy datasets

---

### 🧪 Test Example

Predicted final exam scores for a student who:

- Studies 3 hours
- Sleeps 7 hours
- Goes to a high-ranked school
- Uses phone moderately
- Attends extra coaching
- Eats junk food occasionally

This scenario highlights Lasso's ability to adapt its prediction by choosing which features matter most under different λ values.

---

### 🚀 What You Learn

- How L1 regularization works in practice
- How it differs from Ridge (L2)
- How Lasso helps reduce overfitting **and** select features
