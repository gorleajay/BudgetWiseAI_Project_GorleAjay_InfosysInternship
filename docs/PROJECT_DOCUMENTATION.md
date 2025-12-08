# BudgetWise AI – Project Documentation

This folder contains documentation related to the BudgetWise AI system.

## Contents

### 1. Overview
BudgetWise AI is an intelligent expense forecasting and financial management tool developed as part of the Infosys Springboard Internship. It includes ML-based predictions, budgeting tools, data visualizations, and a Streamlit interface.

### 2. Features Covered
- AI Expense Forecasting
- Budget Tracking
- Recurring Expenses
- Expense History with Filters
- CSV Import/Export
- Secure Authentication
- Interactive Visualizations

### 3. ML Model Summary
- Models Used: CatBoost, XGBoost, LightGBM, Stacked Ensemble
- Final accuracy: 95–100%
- Final trained model: `best_finance_model.pkl`

### 4. Tech Stack
- Python, Streamlit
- SQLite Database
- Pandas, NumPy
- Plotly, Matplotlib, Seaborn
- Google Gemini Pro API

### 5. Execution Flow
1. Run `train_finance_model.py` to generate `best_finance_model.pkl`
2. Launch `app.py` using Streamlit
3. Database initializes automatically
4. User interacts with UI to add/manage expenses
5. ML model provides expense forecasting
