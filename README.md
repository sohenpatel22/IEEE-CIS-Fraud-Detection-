# IEEE-CIS Fraud Detection

This project is a structured machine learning pipeline built on the IEEE-CIS Fraud Detection dataset from Kaggle.  
The goal is to detect fraudulent transactions using real-world feature engineering and scalable ML models.

---

## Problem Statement

Financial fraud detection is a highly imbalanced classification problem where the objective is to identify fraudulent transactions from a large pool of legitimate transactions.

Challenges include:
- Extreme class imbalance (~3–4% fraud cases)
- High number of missing values
- Heterogeneous feature types (transaction + identity data)
- Noisy and partially incomplete identity information

---

## Dataset Description

The dataset consists of two main tables:

### 1. Transaction Data (`train_transaction.csv`)
- Contains core transaction-level information
- Includes amount, product type, card details, and target label (`isFraud`)

### 2. Identity Data (`train_identity.csv`)
- Optional identity-related attributes
- Sparse and partially missing
- Provides additional behavioral signals when available

Both datasets are merged using: TransactionID

## Project Structure

```text

IEEE-CIS-Fraud-Detection/

data/
    raw/                # original Kaggle dataset
    processed/          # cleaned + engineered data
    download_data.py    # script to fetch data

src/
    models.py
    process_data.py

eda/
    eda.ipynb

main.py
requirements.txt
README.md
```
---

## ML Pipeline Overview

The workflow is structured into the following stages:

### 1. Data Ingestion
- Downloaded using Kaggle API / KaggleHub
- Stored in `data/raw/`

### 2. Data Merging
- Transaction and identity datasets are merged using `TransactionID`
- Left join is used to preserve all transaction records

### 3. Exploratory Data Analysis (EDA)
Key insights:
- Strong class imbalance (~3–4% fraud cases)
- Fraudulent transactions show distinct patterns in:
  - Transaction amount
  - Missing values
  - Product categories
- Missingness itself is a predictive signal

### 4. Feature Engineering
Key engineered features include:
- Missing value count per row
- Log transformation of transaction amount
- Frequency encoding of categorical variables (card1, card2, etc.)
- Aggregation features (mean/std transaction amount per entity)
- Time-based features from TransactionDT

### 5. Model Training
Models implemented:
- Logistic Regression (baseline)
- Random Forest
- LightGBM (primary model)
- XGBoost
- CatBoost (optional)

Evaluation metric:
- ROC-AUC (primary metric due to class imbalance)

---

## Evaluation Strategy

Due to class imbalance, accuracy is not a meaningful metric.

Instead, the project uses:
- ROC-AUC score
- Stratified train-test split
- Class-weight balancing

---

## Key Learnings

- Missing values carry strong predictive signal in fraud detection
- Aggregation features significantly improve model performance
- Tree-based models outperform linear models for this dataset
- Feature engineering contributes more than model complexity

---
