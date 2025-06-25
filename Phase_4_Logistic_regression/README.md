# Logistic Regression – Binary Classification (From Scratch)

This folder contains a collection of Python scripts that implement **logistic regression from scratch** using only NumPy. These examples build up the concepts from simple classification to full evaluation with accuracy and confusion matrix, and also include user interaction.

---

## 🔹 1. `binary classification.py`

This file demonstrates:

- How to classify binary outcomes using logistic regression.
- Training with manually defined features and labels.
- Calculation of sigmoid activation, binary cross-entropy loss, and manual gradient descent.

**Key Concepts:**
- Sigmoid function
- Binary cross-entropy loss
- Manual matrix-based gradient calculation
- Prediction thresholding (`>= 0.5`)

---

## 🔹 2. `patient is diabetic or not.py`

This file builds on the logistic regression foundation and:

- Predicts whether a patient is diabetic based on two input features: **glucose level** and **BMI**
- Accepts user input to make predictions
- Standardizes new user input using training data mean and standard deviation
- Warns if inputs seem unrealistic (e.g., glucose = 25)

**Key Features:**
- Realtime prediction using trained model
- Input validation (acceptable glucose: 40–300, BMI: 10–70)
- User interaction through command-line input

---

## 🔹 3. `Confusion and accuracy.py`

This file adds full model evaluation to your training workflow.

**Includes:**
- Accuracy calculation
- Manual confusion matrix computation:
  - TP (True Positives)
  - TN (True Negatives)
  - FP (False Positives)
  - FN (False Negatives)

**Why It's Important:**
These metrics help evaluate how well the model is performing beyond just the loss value. This is critical in real-world applications like healthcare or spam detection.

---

## 📁 Overall Workflow Across Files

1. Create or load dataset
2. Standardize features (mean/std normalization)
3. Initialize weights and bias
4. Perform matrix-based forward pass
5. Compute binary cross-entropy loss
6. Calculate gradients and update weights
7. Evaluate model using accuracy and confusion matrix
8. Accept and validate user input for predictions

---

## 🧰 Tools Used

- Python 3
- NumPy (for all numerical and matrix operations)
- No ML libraries (no scikit-learn, no TensorFlow)

---

## 🚀 Learning Outcome

Through these files, I built logistic regression completely from scratch — including the math for:
- Sigmoid
- BCE loss
- Gradients
- Prediction
- Model evaluation

This helps me deeply understand how classification models actually work under the hood.

