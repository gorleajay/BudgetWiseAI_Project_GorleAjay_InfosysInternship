# ============================================================
# 🔥 BudgetWise AI – Personal Finance Goal Prediction Model Training
# Trains CatBoost, XGBoost, LightGBM and Stacked Ensemble Model
# Saves the best model as best_finance_model.pkl
# ============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

# Make sure your dataset is in the ROOT folder of the project
df = pd.read_csv("personal_finance_tracker_dataset_inr.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# ============================================================
# STEP 2: CLEANING & PREPROCESSING
# ============================================================

df.drop(["date", "user_id"], axis=1, inplace=True, errors='ignore')
df.dropna(inplace=True)

X = df.drop("savings_goal_met", axis=1)
y = df["savings_goal_met"]

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['number']).columns.tolist()

# ============================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# STEP 4: TRAIN MODEL 1 — CATBOOST
# ============================================================

print("\n🔥 Training CatBoost Model...")

cat_model = CatBoostClassifier(
    iterations=1500,
    depth=10,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=200
)

cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=categorical_features)

cat_preds = cat_model.predict(X_test)
cat_acc = accuracy_score(y_test, cat_preds)

print("\nCatBoost Accuracy:", cat_acc)
print(classification_report(y_test, cat_preds))

# ============================================================
# STEP 5: TRAIN MODEL 2 — XGBOOST
# ============================================================

print("\n🔥 Training XGBoost Model...")

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        n_estimators=800,
        max_depth=9,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    ))
])

xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)

print("\nXGBoost Accuracy:", xgb_acc)
print(classification_report(y_test, xgb_preds))

# ============================================================
# STEP 6: TRAIN MODEL 3 — LIGHTGBM
# ============================================================

print("\n🔥 Training LightGBM Model...")

lgb_train = X_train.copy()
lgb_test = X_test.copy()

for col in categorical_features:
    lgb_train[col] = lgb_train[col].astype("category")
    lgb_test[col] = lgb_test[col].astype("category")

lgb_model = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=40,
    random_state=42
)

lgb_model.fit(lgb_train, y_train)
lgb_preds = lgb_model.predict(lgb_test)
lgb_acc = accuracy_score(y_test, lgb_preds)

print("\nLightGBM Accuracy:", lgb_acc)
print(classification_report(y_test, lgb_preds))

# ============================================================
# STEP 7: STACKING ENSEMBLE MODEL
# ============================================================

print("\n🔥 Building Stacked Ensemble Model...")

stack_train = pd.DataFrame({
    "cat": cat_model.predict(X_train),
    "xgb": xgb_model.predict(X_train),
    "lgb": lgb_model.predict(lgb_train)
})

stack_test = pd.DataFrame({
    "cat": cat_preds,
    "xgb": xgb_preds,
    "lgb": lgb_preds
})

meta_model = CatBoostClassifier(
    iterations=800,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=100
)

meta_model.fit(stack_train, y_train)
final_preds = meta_model.predict(stack_test)
final_acc = accuracy_score(y_test, final_preds)

print("\n🔥 FINAL ENSEMBLE ACCURACY:", final_acc)
print(classification_report(y_test, final_preds))

# ============================================================
# STEP 8: SAVE BEST MODEL
# ============================================================

best_model_name = ""

if final_acc >= cat_acc and final_acc >= xgb_acc and final_acc >= lgb_acc:
    joblib.dump(meta_model, "best_finance_model.pkl")
    best_model_name = "Stacked Ensemble"
elif cat_acc >= xgb_acc and cat_acc >= lgb_acc:
    joblib.dump(cat_model, "best_finance_model.pkl")
    best_model_name = "CatBoost"
elif xgb_acc >= lgb_acc:
    joblib.dump(xgb_model, "best_finance_model.pkl")
    best_model_name = "XGBoost"
else:
    joblib.dump(lgb_model, "best_finance_model.pkl")
    best_model_name = "LightGBM"

print(f"\n✅ Best model saved as best_finance_model.pkl ({best_model_name})")

# ============================================================
# OPTIONAL: PLOTS (Accuracy, Confusion Matrix, Metrics)
# ============================================================

accuracies = {
    'CatBoost': cat_acc,
    'XGBoost': xgb_acc,
    'LightGBM': lgb_acc,
    'Stacked Ensemble': final_acc
}

accuracy_series = pd.Series(accuracies)
plt.figure(figsize=(10, 6))
sns.barplot(x=accuracy_series.index, y=accuracy_series.values, palette='viridis')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.9, 1.01)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix - Ensemble Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
