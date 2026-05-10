# 📉 Customer Churn Prediction

A full end-to-end machine learning project that predicts whether a telecom customer will churn (leave the service) using classification models and explainable AI.

---

## 🎯 Project Overview

Customer churn is one of the most critical problems in the telecom industry. Losing a customer is far more expensive than retaining one. This project builds a machine learning pipeline that:

- Predicts which customers are likely to churn
- Handles real-world challenges like class imbalance
- Compares multiple ML models and selects the best one
- Explains **why** the model made each prediction using SHAP values

---

## 📂 Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target column:** `Churn` (Yes / No)
- **Features include:** tenure, contract type, monthly charges, internet service, payment method, and more

---

## 🔧 Tech Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Boosting | `xgboost` |
| Imbalance handling | `imbalanced-learn` (SMOTE) |
| Explainability | `shap` |

---

## 🚀 Project Pipeline

### Step 1 — Setup
Install all required libraries and download the dataset from Kaggle.

### Step 2 — Exploratory Data Analysis
- Load the dataset and inspect shape, column types, and statistics
- Visualize the churn distribution
- Identify class imbalance: ~73% stayed, ~27% churned

### Step 3 — Data Cleaning
- Convert `TotalCharges` from text to numeric
- Drop 11 rows with missing values
- Remove the `customerID` column (no predictive value)
- Encode the target column: `Yes → 1`, `No → 0`

### Step 4 — Feature Engineering
- Encode binary columns (`gender`, `Partner`, `PhoneService`, etc.) to 0/1
- Apply One-Hot Encoding to multi-value columns (`InternetService`, `Contract`, `PaymentMethod`)
- Scale numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`
- Split data into train (80%) and test (20%) sets

### Step 5 — Handle Class Imbalance with SMOTE
- Apply **SMOTE** (Synthetic Minority Oversampling Technique) on training data only
- Balance the churn class from ~1,500 to ~4,200 synthetic samples
- Test set remains untouched (real data only)

### Step 6 — Model Training & Evaluation
Train and compare 3 models:

| Model | F1-Score (Churn class) |
|-------|----------------------|
| Logistic Regression | ~0.63 |
| Random Forest | ~0.72 |
| **XGBoost** ✅ | **~0.87** |

Evaluation metrics used: Precision, Recall, F1-Score, Confusion Matrix

### Step 7 — Explainability with SHAP
- Generate SHAP values for the XGBoost model
- Plot global feature importance (bar + dot summary plots)
- Explain individual customer predictions with force plots

---

## 📊 Key Results

- **Best model:** XGBoost with **F1-Score of 0.87** on the churn class
- **Top churn drivers identified by SHAP:**
  - Short tenure → high churn risk
  - Month-to-month contract → high churn risk
  - High monthly charges → high churn risk
  - Long-term contracts → customers tend to stay

---

## 📁 Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
│
├── churn_prediction.ipynb                       # Main Jupyter Notebook
├── requirements.txt                             # All dependencies
└── README.md                                    # This file
```

---

## ⚙️ How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Go to [this Kaggle link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it inside the `data/` folder.

**4. Run the notebook**
```bash
jupyter notebook churn_prediction.ipynb
```

---

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
shap
jupyter
```

---

## 🙋‍♀️ Author

**Jinane Al Ammar**
Data Scientist | Machine Learning | Python
📧 jinaneelammar@gmail.com
📍 Beirut, Lebanon
