# E-Commerce Customer Churn Prediction v2

End-to-end machine learning project to predict and explain 
customer churn using XGBoost and SHAP.

## Results

| Metric | Logistic Regression (v1) | XGBoost (v2) |
|---|---|---|
| ROC-AUC | 0.745 | **0.929** |
| Accuracy | 67.4% | **91.0%** |
| Churn Recall | 70% | **86%** |

## Dataset
- 50,000 customer records, 25 features
- Source: Kaggle E-Commerce Customer Behavior Dataset
- Churn rate: 28.9%

## What this project does
1. Cleans and preprocesses raw customer behavioral data
2. Handles class imbalance with scale_pos_weight
3. Trains an XGBoost classifier with early stopping
4. Evaluates with ROC-AUC, precision, recall, F1
5. Explains predictions using SHAP values
6. Saves a deployable model artifact

## Key Findings (from SHAP)
- **Customer service calls** are the #1 churn predictor
- **High-value customers** are churning most — priority retention target
- **Cart abandonment** signals checkout friction
- **Email open rate** is a leading indicator of silent disengagement
- **30+ days inactivity** strongly predicts churn

## Business Recommendations
- Score all customers weekly, trigger campaigns at risk > 0.70
- Prioritize high Lifetime Value churners for VIP retention
- Monitor Customer_Service_Calls as an early warning system
- Fix checkout flow to reduce cart abandonment

## Tech Stack
Python, XGBoost, SHAP, Scikit-learn, Pandas, Matplotlib, Seaborn

## Project Structure
churn_xgboost_v2.ipynb  ← main notebook
churn_model_xgb.pkl     ← saved model
images/                 ← plots and visualizations

## Author
MERIEM AZNAG — Aspiring Data Scientist





