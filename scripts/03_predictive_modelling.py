"""
Script 03: Predictive Modelling (FINAL FIXED)
Healthcare Analysis & Disease Prediction Model
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, mean_squared_error, r2_score
)

import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH = r'C:\MicroProject\healthcare_project\data\healthcare_dataset.csv'
OUT_PATH  = r'C:\MicroProject\healthcare_project\outputs\\'
os.makedirs(OUT_PATH, exist_ok=True)

sns.set_theme(style='whitegrid')

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

print("="*60)
print("PREDICTIVE MODELLING")
print("="*60)

# ─────────────────────────────────────────────
# ENCODE CATEGORICALS
# ─────────────────────────────────────────────
le = LabelEncoder()
cat_cols = ['Gender','Smoking_Status','Alcohol_Use','Physical_Activity','Diet_Quality']

for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))

# ─────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────
drop_cols = ['PatientID','Diabetes','Heart_Disease','BMI_Category'] + cat_cols
features = [c for c in df.columns if c not in drop_cols + ['Disease']]

print(f"Features used ({len(features)}): {features}")

X = df[features]
y = df['Disease']

# ─────────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ─────────────────────────────────────────────
# PIPELINE FOR CLASSIFICATION
# ─────────────────────────────────────────────
clf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_train_sc = clf_pipeline.fit_transform(X_train)
X_test_sc  = clf_pipeline.transform(X_test)

# ═════════════════════════════════════════════
# 1. LINEAR REGRESSION (FINAL FIX)
# ═════════════════════════════════════════════
print("\n" + "─"*50)
print("MODEL 1: LINEAR REGRESSION")
print("─"*50)

lr_features = ['Age','BMI','Systolic_BP','Cholesterol','HbA1c',
               'Creatinine','LDL','HDL','Triglycerides','Hemoglobin']

# ✅ Drop NaN in target
lr_df = df[lr_features + ['Blood_Glucose']].copy()
lr_df = lr_df.dropna(subset=['Blood_Glucose'])

Xlr = lr_df[lr_features]
ylr = lr_df['Blood_Glucose']

Xlr_train, Xlr_test, ylr_train, ylr_test = train_test_split(
    Xlr, ylr, test_size=0.2, random_state=42)

# Pipeline
lr_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

lr_pipeline.fit(Xlr_train, ylr_train)
ylr_pred = lr_pipeline.predict(Xlr_test)

# Metrics
lr_mse  = mean_squared_error(ylr_test, ylr_pred)
lr_rmse = np.sqrt(lr_mse)
lr_r2   = r2_score(ylr_test, ylr_pred)

print(f"  R²  : {lr_r2:.4f}")
print(f"  RMSE: {lr_rmse:.4f}")

# ═════════════════════════════════════════════
# 2. LOGISTIC REGRESSION
# ═════════════════════════════════════════════
print("\n" + "─"*50)
print("MODEL 2: LOGISTIC REGRESSION")
print("─"*50)

log_reg = LogisticRegression(max_iter=1000)

log_reg.fit(X_train_sc, y_train)

y_pred_lr = log_reg.predict(X_test_sc)
y_prob_lr = log_reg.predict_proba(X_test_sc)[:,1]

lr_acc = accuracy_score(y_test, y_pred_lr)
lr_auc = roc_auc_score(y_test, y_prob_lr)

print(f"  Accuracy : {lr_acc:.4f}")
print(f"  ROC-AUC  : {lr_auc:.4f}")
print("\n", classification_report(y_test, y_pred_lr))

# ═════════════════════════════════════════════
# 3. DECISION TREE
# ═════════════════════════════════════════════
print("\n" + "─"*50)
print("MODEL 3: DECISION TREE")
print("─"*50)

dt = DecisionTreeClassifier(max_depth=6, random_state=42)

# Decision Tree handles NaN poorly → fill quickly
X_train_dt = X_train.fillna(X_train.median())
X_test_dt  = X_test.fillna(X_train.median())

dt.fit(X_train_dt, y_train)

y_pred_dt = dt.predict(X_test_dt)
y_prob_dt = dt.predict_proba(X_test_dt)[:,1]

dt_acc = accuracy_score(y_test, y_pred_dt)
dt_auc = roc_auc_score(y_test, y_prob_dt)

print(f"  Accuracy : {dt_acc:.4f}")
print(f"  ROC-AUC  : {dt_auc:.4f}")
print("\n", classification_report(y_test, y_pred_dt))

# ═════════════════════════════════════════════
# FINAL OUTPUT
# ═════════════════════════════════════════════
print("\n" + "="*60)
print("MODELLING COMPLETE (SUCCESS ✅)")
print("="*60)