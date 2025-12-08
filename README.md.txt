#  BudgetWise AI – Expense Forecasting Tool  
### Developed as part of the **Infosys Springboard Internship (2025)**  
**Author:** Gorle Ajay  

---

##  Project Overview

**BudgetWise AI** is an intelligent financial management and expense forecasting system designed to help users track their spending, set budgets, analyze trends, and predict future expenses using advanced machine learning models.  
The system integrates a clean **Streamlit web interface**, secure **SQLite storage**, interactive **data visualizations**, and optimized **AI models** (CatBoost, XGBoost, LightGBM, Ensemble).

This project demonstrates real-world implementation of **AI/ML**, **model training**, **frontend dashboards**, **backend logic**, and **end-to-end deployment**.

---

##  Key Features

### 🔹 **AI & Machine Learning**
- Trained using CatBoost, XGBoost, LightGBM, and Stacked Ensemble
- Final accuracy: **95–100%** for financial goal prediction
- AI-powered expense forecasting by category

### 🔹 **Expense Management**
- Add, edit, delete, and track expenses
- Upload receipts (optional)
- View expense history with filtering by date/category

### 🔹 **Budgeting Tools**
- Set monthly budgets for different spending categories
- Dashboard showing:
  - Total spending
  - Transaction count
  - Average daily spending
  - Spending trends and charts

### 🔹 **Recurring Expenses**
- Add expenses like rent, EMIs, subscriptions
- Automatically added monthly

### 🔹 **Interactive Dashboard**
- Visual insights using:
  - Plotly
  - Matplotlib
  - Seaborn

### 🔹 **Secure Authentication**
- SHA256 password hashing
- User login + register system

### 🔹 **AI Advising (Gemini Pro API)**
- Personalized financial guidance
- Smart budgeting suggestions

### 🔹 **Data Portability**
- Export all data as CSV  
- Import CSV files into the system

---

##  Project Structure

```
 BudgetWiseAI_Project_GorleAjay_InfosysInternship
│
├── code/
│   ├── app.py                      # Main Streamlit app
│   └── train_finance_model.py      # ML training script
│
├── models/
│   └── best_finance_model.pkl      # Final trained model
│
├── database/                       # Future: SQLite DB file will appear here
│
├── images/                         # Future: screenshots, visual outputs
│
├── docs/                           # Future documentation files
│
├── personal_finance_tracker_dataset_inr.csv
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

###  Clone the repository

```bash
git clone https://github.com/gorleajay/BudgetWiseAI_Project_GorleAjay_InfosysInternship.git
cd BudgetWiseAI_Project_GorleAjay_InfosysInternship
```

###  Install dependencies

```bash
pip install -r requirements.txt
```

###  Run the Streamlit app

```bash
streamlit run code/app.py
```

###  (Optional) Retrain the ML model

```bash
python code/train_finance_model.py
```

---

##  Technologies Used

- **Python**
- **Streamlit**
- **CatBoost / XGBoost / LightGBM**
- **Scikit-learn**
- **Plotly / Matplotlib / Seaborn**
- **SQLite**
- **Google Gemini Pro API**
- **Pandas & NumPy**

---

##  Internship Contribution

This project was collaboratively developed as part of the Infosys Springboard Internship.  
All team members contributed equally throughout every stage of the project, including:

- Research and requirement analysis  
- Model training and evaluation  
- Streamlit application development  
- Database design and integration  
- Testing, debugging, and documentation  

The successful completion of BudgetWise AI reflects the combined effort, shared responsibilities, and teamwork of every member involved.

##  License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for details.

---

##  Author

**Gorle Ajay**  
Infosys Springboard Internship – AI/ML  
GitHub: https://github.com/gorleajay  
