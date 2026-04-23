# Machine Learning Midterm Assignment

## Breast Cancer Classification using Classical ML + PCA + LDA + SHAP

---

## Project Overview

This project implements a complete machine learning pipeline for the **Breast Cancer Wisconsin dataset**. The goal is to predict whether a tumor is **malignant (0)** or **benign (1)** based on 30 numerical features describing cell nuclei characteristics.

The pipeline includes:
- Data loading and quality checks (missing values, outliers)
- Exploratory Data Analysis (EDA)
- Data scaling and preprocessing
- Dimensionality reduction using **PCA** and **LDA**
- Training **15 models** (5 algorithms × 3 data representations)
- Validation and test evaluation with multiple metrics
- **SHAP** explainability analysis

---

## Dataset

| Property | Value |
|----------|-------|
| Source | `sklearn.datasets.load_breast_cancer()` |
| Samples | 569 |
| Features | 30 (numerical) |
| Target | 0 = Malignant, 1 = Benign |
| Class Distribution | Benign: 357, Malignant: 212 |

---

## Models Implemented

| Algorithm | Data Type |
|-----------|-----------|
| Logistic Regression | Raw, PCA, LDA |
| Decision Tree | Raw, PCA, LDA |
| Random Forest | Raw, PCA, LDA |
| XGBoost | Raw, PCA, LDA |
| Naive Bayes (GaussianNB) | Raw, PCA, LDA |

**Total models trained: 15**

---

## Key Results

### Validation Performance (Top 5 models)

| Model | Data | Accuracy | F1-score | ROC-AUC |
|-------|------|----------|----------|---------|
| Logistic Regression | Raw | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | LDA | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | LDA | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | Raw | 0.9825 | 0.9863 | 1.0000 |
| Random Forest | Raw | 0.9825 | 0.9859 | 0.9987 |

### Best Model on Test Set (Logistic Regression - Raw Data)

| Metric | Score |
|--------|-------|
| Accuracy | 0.9737 |
| Precision | 0.9722 |
| Recall | 0.9859 |
| F1-score | 0.9790 |
| ROC-AUC | 0.9967 |

---
