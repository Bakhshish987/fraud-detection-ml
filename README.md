# 💰 XGBoost-Based Classification Model for Credit Card Fraud Detection

This project demonstrates the application of machine learning techniques to detect fraudulent transactions in credit card data. The dataset is highly imbalanced, with only 0.17% of transactions labeled as fraud. We tackle this challenge using resampling techniques and train a high-performing classifier using XGBoost.

---

## 📁 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Rows:** 284,807 transactions
- **Features:** `V1` to `V28` (PCA-transformed), `Time`, `Amount`
- **Target:** `Class` — 0 (Legit), 1 (Fraud)

---

## 🧠 Objectives

- Handle extreme class imbalance (0.17% fraud)
- Build a classification model that maximizes fraud recall without sacrificing precision
- Compare baseline and advanced models (Logistic Regression vs XGBoost)
- Evaluate using precision, recall, F1-score, and ROC-AUC — not just accuracy

---

## 🔬 Key Steps

1. **Data Preprocessing**
   - Scaled `Amount` and `Time` using `StandardScaler`
   - Checked for nulls (none)
   - Verified extreme class imbalance

2. **Class Imbalance Handling**
   - Used **random oversampling** to duplicate fraud cases and balance the dataset

3. **Model Training**
   - Baseline: **Logistic Regression**
   - Advanced: **XGBoost Classifier**

4. **Evaluation**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1)
   - ROC-AUC Score
   - ROC Curve Visualization

---

## ⚙️ Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost`

---

## 📊 Results (Tested on Original Dataset)

**Logistic Regression:**
- Precision (Fraud): 0.07  
- Recall (Fraud): 0.92  
- F1 Score (Fraud): 0.13  
- ROC-AUC: ~0.95

**XGBoost Classifier:**
- Precision (Fraud): 0.99  
- Recall (Fraud): 1.00  
- F1 Score (Fraud): 0.99  
- ROC-AUC: ~1.00

---

## 📝 Conclusion

XGBoost significantly outperformed logistic regression on all metrics, demonstrating its ability to handle nonlinearity and class imbalance. The model was evaluated on the original, imbalanced test set to ensure reliability in real-world deployment.

---

## 📌 Project Status

✅ Completed — model trained, evaluated, and visualized  
📁 Ready for inclusion in ML portfolio and resume  
📊 Can be extended with SMOTE, time-based validation, or deployment options

---

## 👤 Author

Bakhshish Sethi  
_Data Science & Machine Learning Portfolio Project_

