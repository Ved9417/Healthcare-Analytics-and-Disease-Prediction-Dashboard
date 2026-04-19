"""
Script 02: Exploratory Data Analysis & Statistical Analysis
Healthcare Analysis & Disease Prediction Model
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_PATH = r'C:\MicroProject\healthcare_project\data\healthcare_dataset.csv'  # Update this path as needed
OUT_PATH  = r'C:\MicroProject\healthcare_project\outputs'

sns.set_theme(style='whitegrid', palette='muted')
COLORS = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print("=" * 60)
print("HEALTHCARE DATASET - EDA & STATISTICAL ANALYSIS")
print("=" * 60)
print(f"\nShape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ── 1. MISSING VALUES ─────────────────────────────────────────────────────────
print("\n--- Missing Values ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing)

# Impute numeric with median, categorical with mode
for col in missing.index:
    if df[col].dtype in ['float64','int64']:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
print("Missing values imputed.")

# ── 2. BASIC STATISTICS ───────────────────────────────────────────────────────
print("\n--- Descriptive Statistics ---")
num_cols = ['Age','BMI','Systolic_BP','Diastolic_BP','Heart_Rate',
            'Blood_Glucose','Cholesterol','Hemoglobin','LDL','HDL',
            'Triglycerides','HbA1c','Creatinine']
desc = df[num_cols].describe().T
desc['variance'] = df[num_cols].var()
desc['skewness'] = df[num_cols].skew()
print(desc[['mean','50%','std','variance','min','max','skewness']].round(2))
desc.to_csv(OUT_PATH + 'descriptive_statistics.csv')

# ── FIGURE 1: Distributions ───────────────────────────────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
fig.suptitle('Distribution of Key Clinical Variables', fontsize=16, fontweight='bold')
cols_plot = ['Age','BMI','Systolic_BP','Diastolic_BP',
             'Blood_Glucose','Cholesterol','HbA1c','LDL',
             'HDL','Triglycerides','Heart_Rate','Hemoglobin']
for ax, col in zip(axes.flatten(), cols_plot):
    ax.hist(df[col], bins=40, color='#2196F3', edgecolor='white', alpha=0.8)
    ax.axvline(df[col].mean(),   color='red',    linestyle='--', linewidth=1.2, label=f'Mean={df[col].mean():.1f}')
    ax.axvline(df[col].median(), color='orange', linestyle=':',  linewidth=1.2, label=f'Median={df[col].median():.1f}')
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(OUT_PATH + 'fig01_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig01_distributions.png")

# ── FIGURE 2: Disease Prevalence ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Disease Prevalence in Dataset', fontsize=14, fontweight='bold')

for ax, col, title in zip(axes,
    ['Disease','Diabetes','Heart_Disease'],
    ['Overall Disease','Diabetes','Heart Disease']):
    counts = df[col].value_counts()
    labels = ['No Disease','Disease'] if col == 'Disease' else ['No','Yes']
    ax.pie(counts, labels=labels, autopct='%1.1f%%',
           colors=['#4CAF50','#F44336'], startangle=90,
           wedgeprops={'edgecolor':'white','linewidth':2})
    ax.set_title(title, fontsize=12)

plt.tight_layout()
plt.savefig(OUT_PATH + 'fig02_disease_prevalence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig02_disease_prevalence.png")

# ── FIGURE 3: Boxplots by Disease ────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle('Clinical Variables by Disease Status', fontsize=14, fontweight='bold')
key_vars = ['Age','BMI','Blood_Glucose','Systolic_BP',
            'Cholesterol','HbA1c','LDL','Triglycerides']
for ax, col in zip(axes.flatten(), key_vars):
    df.boxplot(column=col, by='Disease', ax=ax,
               patch_artist=True,
               boxprops=dict(facecolor='#90CAF9'),
               medianprops=dict(color='red', linewidth=2))
    ax.set_title(col, fontsize=10)
    ax.set_xlabel('Disease (0=No, 1=Yes)')
    plt.sca(ax)
    plt.title(col)
plt.suptitle('Clinical Variables by Disease Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_PATH + 'fig03_boxplots_by_disease.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig03_boxplots_by_disease.png")

# ── FIGURE 4: Correlation Heatmap ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
corr = df[num_cols + ['Disease','Diabetes','Heart_Disease']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, ax=ax, linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Correlation Matrix – Clinical Features & Disease Outcomes',
             fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(OUT_PATH + 'fig04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig04_correlation_heatmap.png")

# ── FIGURE 5: Gender & Lifestyle Analysis ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Lifestyle Factors & Disease', fontsize=14, fontweight='bold')

# Gender vs Disease
g_data = df.groupby(['Gender','Disease']).size().unstack()
g_data.plot(kind='bar', ax=axes[0,0], color=['#4CAF50','#F44336'], rot=0, edgecolor='white')
axes[0,0].set_title('Disease by Gender'); axes[0,0].set_ylabel('Count')
axes[0,0].legend(['No Disease','Disease'])

# Smoking vs Disease
s_data = df.groupby(['Smoking_Status','Disease']).size().unstack()
s_data.plot(kind='bar', ax=axes[0,1], color=['#4CAF50','#F44336'], rot=0, edgecolor='white')
axes[0,1].set_title('Disease by Smoking Status'); axes[0,1].set_ylabel('Count')
axes[0,1].legend(['No Disease','Disease'])

# Physical Activity vs Disease
p_data = df.groupby(['Physical_Activity','Disease']).size().unstack()
p_data.plot(kind='bar', ax=axes[1,0], color=['#4CAF50','#F44336'], rot=0, edgecolor='white')
axes[1,0].set_title('Disease by Physical Activity'); axes[1,0].set_ylabel('Count')
axes[1,0].legend(['No Disease','Disease'])

# BMI category
df['BMI_Category'] = pd.cut(df['BMI'],
    bins=[0,18.5,25,30,100],
    labels=['Underweight','Normal','Overweight','Obese'])
bmi_data = df.groupby(['BMI_Category','Disease']).size().unstack()
bmi_data.plot(kind='bar', ax=axes[1,1], color=['#4CAF50','#F44336'], rot=15, edgecolor='white')
axes[1,1].set_title('Disease by BMI Category'); axes[1,1].set_ylabel('Count')
axes[1,1].legend(['No Disease','Disease'])

plt.tight_layout()
plt.savefig(OUT_PATH + 'fig05_lifestyle_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig05_lifestyle_analysis.png")

# ── FIGURE 6: Age Distribution by Disease ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Age Profile by Disease Status', fontsize=14, fontweight='bold')

for grp, color, label in [(0,'#4CAF50','No Disease'),(1,'#F44336','Disease')]:
    axes[0].hist(df[df['Disease']==grp]['Age'], bins=35, alpha=0.6,
                 color=color, label=label, edgecolor='white')
axes[0].set_xlabel('Age'); axes[0].set_ylabel('Count')
axes[0].set_title('Age Distribution'); axes[0].legend()

age_bins = pd.cut(df['Age'], bins=[18,30,40,50,60,70,90])
age_prev = df.groupby(age_bins, observed=True)['Disease'].mean() * 100
age_prev.plot(kind='bar', ax=axes[1], color='#2196F3', edgecolor='white', rot=30)
axes[1].set_title('Disease Prevalence by Age Group')
axes[1].set_ylabel('Disease Prevalence (%)')
axes[1].set_xlabel('Age Group')

plt.tight_layout()
plt.savefig(OUT_PATH + 'fig06_age_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig06_age_analysis.png")

# ── 3. HYPOTHESIS TESTING ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("HYPOTHESIS TESTING")
print("="*60)

results = []

def run_ttest(col):
    g0 = df[df['Disease']==0][col]
    g1 = df[df['Disease']==1][col]
    t, p = stats.ttest_ind(g0, g1)
    sig = "REJECT H0 (Significant)" if p < 0.05 else "FAIL TO REJECT H0"
    results.append({'Feature': col, 'Test': 'Independent t-test',
                    'Statistic': round(t, 4), 'p-value': round(p, 6),
                    'Mean (No Disease)': round(g0.mean(), 2),
                    'Mean (Disease)': round(g1.mean(), 2),
                    'Conclusion': sig})
    print(f"\nH0: No significant difference in {col} between diseased/non-diseased groups")
    print(f"H1: Significant difference exists")
    print(f"  t-statistic = {t:.4f}, p-value = {p:.6f} → {sig}")

for c in ['Age','BMI','Blood_Glucose','Systolic_BP','Cholesterol','HbA1c','LDL','Triglycerides']:
    run_ttest(c)

# Chi-Square tests for categorical
print("\n--- Chi-Square Tests ---")
for cat in ['Smoking_Status','Physical_Activity','Diet_Quality','Gender']:
    ct = pd.crosstab(df[cat], df['Disease'])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    sig = "REJECT H0 (Significant)" if p < 0.05 else "FAIL TO REJECT H0"
    results.append({'Feature': cat, 'Test': 'Chi-Square',
                    'Statistic': round(chi2, 4), 'p-value': round(p, 6),
                    'Mean (No Disease)': '-', 'Mean (Disease)': '-',
                    'Conclusion': sig})
    print(f"\nH0: {cat} is independent of Disease")
    print(f"H1: {cat} is associated with Disease")
    print(f"  Chi2 = {chi2:.4f}, p-value = {p:.6f} → {sig}")

pd.DataFrame(results).to_csv(OUT_PATH + 'hypothesis_testing_results.csv', index=False)
print("\nHypothesis testing results saved.")

# Save clean dataset for modelling
df.to_csv(OUT_PATH + 'healthcare_clean.csv', index=False)
print("\nClean dataset saved for modelling.")
print("\nEDA Complete!")
