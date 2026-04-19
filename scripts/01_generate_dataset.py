"""
Script 01: Dataset Generation
Healthcare Analysis & Disease Prediction Model
Generates a synthetic but realistic dataset of 10,500 patient records.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 10500

# --- Demographics ---
age = np.random.randint(18, 90, N)
gender = np.random.choice(['Male', 'Female'], N, p=[0.48, 0.52])
bmi = np.round(np.random.normal(27.5, 6.0, N).clip(15, 55), 1)

# --- Vitals ---
systolic_bp   = np.random.normal(120, 18, N).clip(80, 200).astype(int)
diastolic_bp  = np.random.normal(80, 12, N).clip(50, 130).astype(int)
heart_rate    = np.random.normal(75, 12, N).clip(45, 140).astype(int)
blood_glucose = np.random.normal(100, 30, N).clip(60, 400).astype(int)
cholesterol   = np.random.normal(200, 40, N).clip(100, 400).astype(int)
hemoglobin    = np.round(np.random.normal(13.5, 2.0, N).clip(7, 20), 1)

# --- Lifestyle ---
smoking_status   = np.random.choice(['Never', 'Former', 'Current'], N, p=[0.55, 0.25, 0.20])
alcohol_use      = np.random.choice(['None', 'Moderate', 'Heavy'],  N, p=[0.45, 0.40, 0.15])
physical_activity= np.random.choice(['Low', 'Moderate', 'High'],    N, p=[0.35, 0.40, 0.25])
diet_quality     = np.random.choice(['Poor', 'Fair', 'Good'],       N, p=[0.30, 0.40, 0.30])

# --- Medical history ---
family_history_diabetes = np.random.choice([0, 1], N, p=[0.65, 0.35])
family_history_heart    = np.random.choice([0, 1], N, p=[0.60, 0.40])
previous_stroke         = np.random.choice([0, 1], N, p=[0.93, 0.07])
chronic_kidney_disease  = np.random.choice([0, 1], N, p=[0.88, 0.12])

# --- Symptoms ---
chest_pain       = np.random.choice([0, 1], N, p=[0.80, 0.20])
shortness_breath = np.random.choice([0, 1], N, p=[0.78, 0.22])
fatigue          = np.random.choice([0, 1], N, p=[0.65, 0.35])
frequent_urination = np.random.choice([0, 1], N, p=[0.75, 0.25])

# --- Lab results ---
creatinine  = np.round(np.random.normal(1.0, 0.3, N).clip(0.4, 5.0), 2)
hba1c       = np.round(np.random.normal(5.7, 1.2, N).clip(4.0, 14.0), 1)
ldl         = np.random.normal(130, 35, N).clip(50, 300).astype(int)
hdl         = np.random.normal(55, 15, N).clip(20, 100).astype(int)
triglycerides = np.random.normal(150, 60, N).clip(50, 600).astype(int)

# --- Disease probability model ---
# Diabetes risk
diabetes_score = (
    0.03 * age
    + 0.04 * bmi
    + 0.005 * blood_glucose
    + 0.8  * family_history_diabetes
    + 0.6  * (smoking_status == 'Current')
    + 0.5  * (physical_activity == 'Low')
    + 0.7  * (diet_quality == 'Poor')
    + 0.8  * (hba1c > 6.5)
    + 0.5  * frequent_urination
    - 0.5  * (physical_activity == 'High')
)
diabetes_prob = 1 / (1 + np.exp(-(diabetes_score - 5)))
diabetes = (np.random.rand(N) < diabetes_prob).astype(int)

# Heart disease risk
heart_score = (
    0.03 * age
    + 0.02 * bmi
    + 0.005 * systolic_bp
    + 0.004 * cholesterol
    + 0.8  * family_history_heart
    + 0.9  * (smoking_status == 'Current')
    + 0.7  * chest_pain
    + 0.5  * shortness_breath
    + 0.4  * (alcohol_use == 'Heavy')
    + 0.6  * (physical_activity == 'Low')
    + 0.5  * previous_stroke
    - 0.4  * (physical_activity == 'High')
)
heart_prob = 1 / (1 + np.exp(-(heart_score - 5)))
heart_disease = (np.random.rand(N) < heart_prob).astype(int)

# Overall disease prediction target (1 = has at least one major disease)
disease = np.where((diabetes == 1) | (heart_disease == 1), 1, 0)

# --- Assemble DataFrame ---
df = pd.DataFrame({
    'PatientID':            [f'P{str(i).zfill(5)}' for i in range(1, N+1)],
    'Age':                  age,
    'Gender':               gender,
    'BMI':                  bmi,
    'Systolic_BP':          systolic_bp,
    'Diastolic_BP':         diastolic_bp,
    'Heart_Rate':           heart_rate,
    'Blood_Glucose':        blood_glucose,
    'Cholesterol':          cholesterol,
    'Hemoglobin':           hemoglobin,
    'LDL':                  ldl,
    'HDL':                  hdl,
    'Triglycerides':        triglycerides,
    'HbA1c':                hba1c,
    'Creatinine':           creatinine,
    'Smoking_Status':       smoking_status,
    'Alcohol_Use':          alcohol_use,
    'Physical_Activity':    physical_activity,
    'Diet_Quality':         diet_quality,
    'Family_History_Diabetes': family_history_diabetes,
    'Family_History_Heart': family_history_heart,
    'Previous_Stroke':      previous_stroke,
    'Chronic_Kidney_Disease': chronic_kidney_disease,
    'Chest_Pain':           chest_pain,
    'Shortness_of_Breath':  shortness_breath,
    'Fatigue':              fatigue,
    'Frequent_Urination':   frequent_urination,
    'Diabetes':             diabetes,
    'Heart_Disease':        heart_disease,
    'Disease':              disease,   # Primary target variable
})

# Introduce ~2% missing values for realism
for col in ['BMI', 'Cholesterol', 'Blood_Glucose', 'HbA1c', 'Creatinine']:
    mask = np.random.rand(N) < 0.02
    df.loc[mask, col] = np.nan

df.to_csv(r'C:\MicroProject\healthcare_project\data\healthcare_dataset.csv', index=False)
print(f"Dataset saved: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Disease prevalence: {df['Disease'].mean():.2%}")
print(f"Diabetes prevalence: {df['Diabetes'].mean():.2%}")
print(f"Heart Disease prevalence: {df['Heart_Disease'].mean():.2%}")
print(df.dtypes)
