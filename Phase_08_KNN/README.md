# K-Nearest Neighbors (KNN) for SMS Spam Detection

## Overview
This project demonstrates the implementation of a **K-Nearest Neighbors (KNN) classifier from scratch** for detecting spam in SMS messages. The main goal was to understand **how KNN works**, handle real-world text data, and observe the effects of preprocessing and hyperparameter tuning on model performance.

## Dataset
- **Source:** Kaggle SMS Spam Collection Dataset
- **Total messages:** 5,574  
- **Spam proportion:** ~13%  
- **Columns:**
  - `v1` → Label (`ham` or `spam`)
  - `v2` → SMS content  

**Preprocessing steps:**
- Removed missing values and duplicate messages to prevent overfitting.  
- Mapped labels: `ham` → 0, `spam` → 1.  
- Split data: 80% train, 20% test (stratified by label).  

## Feature Engineering
- Used **TF-IDF vectorization** to convert SMS text into numerical features.  
- Tuned TF-IDF:
  - `lowercase=True`
  - `stop_words='english'`
  - `ngram_range=(1,2)` (unigrams + bigrams)
  - `min_df=2` (ignore rare words)  

## KNN Implementation
- Implemented KNN **from scratch** using only NumPy.
- **Distance metric:** Cosine distance (better for sparse, high-dimensional TF-IDF vectors).  
- **Prediction:**
  - Compute distance of a new message to all training messages.  
  - Identify **k nearest neighbors**.  
  - Assign the **majority label** (tie-break: smaller label wins).  

## Model Training & Testing
- Initial KNN trained with `k=7` neighbors.  
- Test accuracy: **97.2%**  

## Hyperparameter Tuning
- Explored `k` values: `[1,3,5,7,9,11,15,20,25,30]`  
- Optimal `k` found: `k=7` with **97.2% test accuracy**  

## Challenges & Insights
1. **Overfitting:**  
   - Many repeated messages in dataset caused Euclidean KNN to memorize the training set.  
   - Resolved by removing duplicates and using cosine distance.

2. **Distance metric:**  
   - Cosine distance worked better than Euclidean for high-dimensional TF-IDF vectors.

3. **Imbalanced dataset:**  
   - Only ~13% spam → careful tuning of `k` required to maintain recall for spam messages.

## Future Improvements
- Use **weighted KNN**: neighbors contribute proportionally to distance.  
- Implement **Naive Bayes or ensemble methods** for comparison.  
- Explore **text preprocessing enhancements**: stemming, lemmatization, or advanced embeddings.  

## Tools & Libraries
- Python 3  
- Libraries: `pandas`, `numpy`, `scikit-learn`  

## Conclusion
This project helped in understanding **the mechanics of KNN**, the importance of preprocessing for real-world text data, and how **hyperparameter tuning** affects model performance. With careful feature engineering and tuning, the KNN model achieved **97.2% accuracy** while correctly classifying new unseen messages.
"""