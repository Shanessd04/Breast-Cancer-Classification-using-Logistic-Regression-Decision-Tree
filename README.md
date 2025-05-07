# 🧠 Breast Cancer Classification using Logistic Regression & Decision Tree

This project explores binary classification of breast cancer data using supervised learning. The goal is to predict whether a tumor is malignant or benign based on clinical features.

---

## 📁 Dataset

- **Source:** [UCI Machine Learning Repository – Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer)
- **Imported using:** `ucimlrepo` Python package
- **Target Variable:** `diagnosis` (binary: benign or malignant)
- **Features:** 30 numeric medical attributes (e.g., radius, texture, smoothness, etc.)

---

## 🧹 Data Preprocessing

- Loaded dataset via `fetch_ucirepo(id=17)`
- Checked for missing values
- Encoded target labels (`LabelEncoder`)
- Split data: **80% train / 20% test**

---

## 🔍 Feature Selection

- Used **Chi-squared test** via `SelectKBest(chi2, k=10)`
- Selected top 10 features that contribute most to the classification

---

## 🧠 Models Trained

### ✅ Logistic Regression
- Tried both L1 (Lasso) and L2 (Ridge) penalties
- Key hyperparameters:
  - `solver`: `liblinear`, `lbfgs`
  - `penalty`: `'l1'` and `'l2'`
  - `C`: 1.0
- Evaluated performance using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report

### ✅ Decision Tree Classifier
- Used `criterion='entropy'` and `gini` for split quality
- Hyperparameters:
  - `max_depth`
  - `random_state`
- Performance evaluation similar to Logistic Regression

---

## 📊 Model Evaluation

Metrics used:
- **Accuracy Score**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**

Visualized confusion matrix and reviewed misclassification trends.

---

## 📦 Requirements

```bash
pip install pandas numpy scikit-learn ucimlrepo matplotlib seaborn
