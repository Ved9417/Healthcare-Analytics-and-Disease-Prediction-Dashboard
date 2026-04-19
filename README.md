# Healthcare Analysis & Disease Prediction Model
## End-to-End Data Analytics Pipeline

**Micro Project | SY AIML AIDS | School of Engineering and Technology**

---

## Project Overview

This project develops a complete end-to-end data analytics pipeline for healthcare data, covering disease prediction using statistical analysis and machine learning.

| Detail | Info |
|--------|------|
| **Domain** | Healthcare |
| **Dataset** | 10,500 Patient Records |
| **Target** | Disease Prediction (Binary Classification) |
| **Models** | Linear Regression, Logistic Regression, Decision Tree |

---

## Project Structure

```
healthcare_project/
├── data/
│   └── healthcare_dataset.csv          # Raw dataset (10,500 records)
├── notebooks/
│   └── Healthcare_Disease_Prediction_Complete.ipynb  # Full pipeline notebook
├── scripts/
│   ├── 01_generate_dataset.py          # Dataset generation
│   ├── 02_eda_statistical_analysis.py  # EDA & hypothesis testing
│   ├── 03_predictive_modelling.py      # ML models
│   └── 04_create_notebook.py           # Notebook generator
├── outputs/
│   ├── healthcare_clean.csv            # Cleaned dataset
│   ├── descriptive_statistics.csv      # Statistical summary
│   ├── hypothesis_testing_results.csv  # Hypothesis test results
│   ├── model_comparison.csv            # Model metrics
│   ├── decision_tree_rules.txt         # DT rules
│   ├── fig01_distributions.png         # Variable distributions
│   ├── fig02_disease_prevalence.png    # Disease pie charts
│   ├── fig03_boxplots_by_disease.png   # Boxplots
│   ├── fig04_correlation_heatmap.png   # Correlation matrix
│   ├── fig05_lifestyle_analysis.png    # Lifestyle factors
│   ├── fig06_age_analysis.png          # Age analysis
│   ├── fig07_linear_regression.png     # LR results
│   ├── fig08_logistic_regression.png   # Log Reg results
│   ├── fig09_decision_tree.png         # DT results
│   ├── fig10_decision_tree_viz.png     # DT visualization
│   ├── fig11_model_comparison.png      # Model comparison
│   └── fig12_combined_roc.png          # ROC curves
├── dashboard.html                      # Interactive web dashboard
├── requirements.txt
└── README.md
```

---

## Dataset Description

The dataset contains **10,500 patient records** with **30 features**:

### Demographics
- `Age`, `Gender`, `BMI`

### Vitals
- `Systolic_BP`, `Diastolic_BP`, `Heart_Rate`

### Lab Results
- `Blood_Glucose`, `Cholesterol`, `Hemoglobin`, `LDL`, `HDL`, `Triglycerides`, `HbA1c`, `Creatinine`

### Lifestyle
- `Smoking_Status`, `Alcohol_Use`, `Physical_Activity`, `Diet_Quality`

### Medical History
- `Family_History_Diabetes`, `Family_History_Heart`, `Previous_Stroke`, `Chronic_Kidney_Disease`

### Symptoms
- `Chest_Pain`, `Shortness_of_Breath`, `Fatigue`, `Frequent_Urination`

### Targets
- `Diabetes` (binary), `Heart_Disease` (binary), `Disease` (primary target)

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline (Sequential)
```bash
python scripts/01_generate_dataset.py
python scripts/02_eda_statistical_analysis.py
python scripts/03_predictive_modelling.py
```

### 3. Open the Notebook
```bash
jupyter notebook notebooks/Healthcare_Disease_Prediction_Complete.ipynb
```

### 4. View Dashboard
Open `dashboard.html` in any web browser.

---

## Key Results

### Statistical Analysis
| Feature | Test | p-value | Conclusion |
|---------|------|---------|------------|
| Age | t-test | < 0.001 | REJECT H₀ |
| BMI | t-test | < 0.001 | REJECT H₀ |
| Blood Glucose | t-test | 0.000087 | REJECT H₀ |
| Smoking Status | Chi-Square | < 0.001 | REJECT H₀ |
| Physical Activity | Chi-Square | < 0.001 | REJECT H₀ |
| Gender | Chi-Square | 0.378 | Fail to Reject H₀ |

### Model Performance
| Model | Accuracy | ROC-AUC | CV Accuracy |
|-------|----------|---------|-------------|
| Logistic Regression | **67.5%** | **0.724** | 67.0% |
| Decision Tree | 64.5% | 0.675 | 64.8% |

---

## Tools & Technologies
- **Python 3.x**
- **Pandas** – Data manipulation
- **NumPy** – Numerical computing
- **Matplotlib / Seaborn** – Visualization
- **Scikit-learn** – Machine learning
- **SciPy** – Statistical tests
- **HTML/CSS/JS** – Dashboard deployment

---

## Pipeline Stages
1. **Problem Definition** – Healthcare disease prediction
2. **Data Collection** – Synthetic clinically-modelled dataset
3. **Data Cleaning** – Missing value imputation, duplicate removal
4. **EDA** – Distributions, boxplots, correlation analysis
5. **Statistical Analysis** – Mean, Median, Std Dev, Variance, t-tests, Chi-Square
6. **Modelling** – Linear Regression, Logistic Regression, Decision Tree
7. **Evaluation** – Accuracy, ROC-AUC, Cross-validation
8. **Deployment** – Interactive HTML dashboard

---

*Submitted by: SY AIML AIDS | Deadline: 30th March 2026*
