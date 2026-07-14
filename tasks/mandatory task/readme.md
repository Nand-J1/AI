# ACM SIGAI – Mandatory Feature Engineering Challenge

## 1. Introduction
This project addresses the **Santander Customer Transaction Prediction** dataset (200,000 rows, ~10% positives).  
The challenge goal: **improve recall from baseline (~0.83) to ≥0.88 using Logistic Regression only, no external data.**

---

## 2. Methodology
Steps followed:
1. **Baseline Logistic Regression** – plain features, default threshold.  
2. **Feature Engineering** – added row‑wise statistical aggregates (mean, std, sum, min, max, median, skew, kurtosis, outlier count).  
3. **Feature Transforms** – applied PowerTransformer to fix skew.  
4. **Class Imbalance Handling** – used `class_weight="balanced"`.  
5. **Hyperparameter Tuning** – GridSearchCV over `C`, solver, penalty.  
6. **Threshold Tuning** – adjusted cutoff to push recall ≥0.88 while keeping F1/ROC‑AUC reasonable.

---

## 3. Results

### Experiment Log 

| Experiment                                | Recall | Precision | F1   | ROC‑AUC | Notes                                      |
|-------------------------------------------|--------|-----------|------|---------|--------------------------------------------|
| Baseline LogisticRegression (no tuning)   | 0.2590 | 0.6822    | 0.3754 | 0.8599 | Plain features, threshold=0.5              |
| LogReg + row‑wise statistical features    | 0.3062 | 0.6858    | 0.4234 | 0.8762 | Added aggregates                           |
| LogReg + PowerTransform + balanced weight | 0.7868| 0.3076    | 0.4423   | 0.8750    | Recall lever                               |
| GridSearchCV‑tuned LogReg (threshold=0.5) | 0.7878    | 0.3074       | 0.4422   | 0.8750    | Best params from grid search               |
| Final: tuned LogReg + custom threshold    | 0.8801  | 0.2273     | 0.3613  | 0.8750

---

### Confusion Matrices
-**=== Baseline LogisticRegression (no tuning) === **:
Recall=0.2590  Precision=0.6822  F1=0.3754  ROC-AUC=0.8599
[[35495   485]
[ 2979  1041]]
---
 - **Final tuned model**:
   New engineered features: ['row_mean', 'row_std', 'row_sum', 'row_min', 'row_max', 'row_median', 'row_skew', 'row_kurt', 'row_outlier_count']

=== LogReg + row-wise statistical features ===
Recall=0.3062  Precision=0.6858  F1=0.4234  ROC-AUC=0.8762
[[35416   564]
 [ 2789  1231]]

=== LogReg + PowerTransform + class_weight=balanced ===
Recall=0.7868  Precision=0.3076  F1=0.4423  ROC-AUC=0.8750
[[28861  7119]
 [  857  3163]]

Best params: {'clf__C': 0.01, 'clf__penalty': 'l2', 'clf__solver': 'liblinear'}
Best CV recall: 0.7873491214224367

Chosen threshold: 0.3434
Final -> Recall=0.8801  Precision=0.2273 F1=0.3613  ROC-AUC=0.8750
              precision    recall  f1-score   support

           0       0.98      0.67      0.79     35980
           1       0.23      0.88      0.36      4020

    accuracy                           0.69     40000
   macro avg       0.60      0.77      0.58     40000
weighted avg       0.90      0.69      0.75     40000
