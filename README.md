# Home Credit Default Risk Prediction

> Predicting whether a loan applicant will default using multi-table feature engineering and machine learning.

---

## Overview

This project tackles the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) problem — a real-world binary classification task where the goal is to predict if a client will repay their loan.

The dataset contains **307,511 loan applications** across **7 relational tables**, with a significant class imbalance (~92% non-default, ~8% default).

---

## Results

| Model | ROC-AUC (Test) | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** | **0.7754** | 0.1804 | 0.6900 | 0.2860 |
| Logistic Regression | 0.7610 | 0.1684 | 0.6886 | 0.2706 |
| Random Forest | 0.7360 | 0.1577 | 0.6554 | 0.2542 |
| Decision Tree | 0.7117 | 0.1585 | 0.5822 | 0.2491 |

> XGBoost achieved the best ROC-AUC of **0.7754** on the test set.

---

## Project Structure

```
home-credit-default-risk/
│
├── Home-Credit-Default-Risk-Notebook.ipynb              # Main notebook (full pipeline)
├── README.md
└── requirements.txt
```

---

## Pipeline

```
Data Loading → Feature Engineering → Dataset Construction
     → Preprocessing → Encoding → Train/Val/Test Split
          → Model Training → Evaluation → Error Analysis
```

### 1. Feature Engineering
Aggregated features from 5 auxiliary tables:

| Table | Features Created |
|---|---|
| `bureau` | Loan counts, overdue days, debt sums, credit status/type counts |
| `previous_application` | Application counts, status distribution, reject reasons, interest rates |
| `POS_CASH_balance` | DPD stats, late payment counts, installment progression |
| `credit_card_balance` | Receivable balance stats, delinquency indicators |
| `installments_payments` | Payment delays (DPD), payment ratios |

### 2. Preprocessing
- Fill zero for bureau request count columns
- Drop rows with critical missing values (very small count)
- Remove highly sparse columns (>60% missing)
- Mean imputation for numerical features + missing flags
- Mode imputation for categorical features + missing flags
- Special handling for `OWN_CAR_AGE` (zero-fill + flag)

### 3. Encoding & Splitting
- One-hot encoding for all categorical variables
- Stratified 70/10/20 train/dev/test split
- Sample weights computed to handle class imbalance

### 4. Models Trained
- Logistic Regression (with StandardScaler pipeline)
- Decision Tree
- Random Forest (400 estimators, depth-limited)
- XGBoost (2000 rounds, learning rate scheduler, early stopping)

---

## Dataset

The dataset is from the [Home Credit Default Risk Kaggle competition](https://www.kaggle.com/c/home-credit-default-risk/data).

Download and place the ZIP file at:
```
/content/drive/MyDrive/home-credit-default-risk.zip
```

---

## How to Run

1. Open the notebook in **Google Colab**
2. Mount your Google Drive
3. Place the dataset ZIP in your Drive as shown above
4. Run all cells in order

```python
# The notebook handles everything:
# - Extraction
# - Feature engineering
# - Preprocessing
# - Training
# - Evaluation
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Key Design Decisions

- **Missing value flags**: Created before imputation to preserve information about missingness.
- **No outlier removal**: Tree-based models are robust to outliers; many zero values come from intentional preprocessing.
- **Sample weights**: Used instead of oversampling to address class imbalance.
- **Threshold = 0.5**: Default threshold used for all models. In production, a lower threshold may improve recall on the default class.
- **Hyperparameter tuning**: All model hyperparameters were selected 
  through manual tuning based on validation performance.
---

## Author

**Ammar** — AI Student, Jordan University of Science and Technology 

