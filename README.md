# Credit Risk Analysis - Loan Predictor Model

## Project Overview

This case study focuses on risk analytics in banking, aiming to minimize financial risks associated with loan approvals. Loan companies face challenges in lending to individuals with little or no credit history, which may lead to loan defaults. The objective of this project is to predict loan eligibility based on various factors such as income, credit history, and more, ensuring capable applicants are not wrongly rejected while identifying risky ones.

### Business Problem

When deciding to approve a loan, the company encounters two primary risks:

1. **Losing Business:** Rejecting reliable applicants leads to missed opportunities.
2. **Losing Money:** Approving risky applicants may result in defaults.

This project employs Exploratory Data Analysis (EDA) and Machine Learning to develop a predictive model for loan eligibility.

---

## Dataset

**Data Source:** [Loan Prediction Dataset](https://raw.githubusercontent.com/Premalatha-success/Financial-Analytics-Loan-Approval-Prediction/main/loan_prediction.csv)

### Fields in the Dataset

- `Loan_ID`: Unique identifier for the loan
- `Gender`: Gender of the applicant
- `Married`: Marital status
- `Dependents`: Number of dependents
- `Education`: Educational qualification
- `Self_Employed`: Employment status
- `ApplicantIncome`: Applicant’s income
- `CoapplicantIncome`: Co-applicant’s income
- `LoanAmount`: Loan amount
- `Loan_Amount_Term`: Term of the loan
- `Credit_History`: Credit history of the applicant
- `Property_Area`: Area of the property
- `Loan_Status`: Loan approval status (Target variable)

Key libraries include:

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- 
## Model Training

The project implements and compares the following classifiers:

- **Logistic Regression:** A linear model for binary classification tasks.
- **K-Nearest Neighbors (KNN):** A non-parametric method for classification and regression.
- **Support Vector Classifier (SVC):** A supervised model that finds the optimal hyperplane for classification.
- **Decision Tree Classifier:** A tree-based model for decision-making tasks.
- **Bagging Classifier:** An ensemble method combining multiple classifiers to reduce variance.
- **Gradient Boosting Classifier:** A boosting algorithm that builds models sequentially.
- **AdaBoost Classifier:** A boosting technique focusing on correcting errors of previous models.

## Evaluation

The models were evaluated using the following metrics:

- **Accuracy:** Measures the percentage of correctly classified instances.
- **Precision:** Indicates the proportion of true positive predictions among all positive predictions.
- **Recall:** Represents the ability of the model to identify all relevant instances.
- **F1 Score:** Provides a harmonic mean of precision and recall.

