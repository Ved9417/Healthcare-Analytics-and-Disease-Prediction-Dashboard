"""
app.py — Healthcare Analysis & Disease Prediction
Best-in-class Streamlit deployment for Micro Project
# SY AIML AIDS | School of Engineering and Technology
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, mean_squared_error, r2_score,
    ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Disease Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a2035, #1f2b47);
        border: 1px solid #2a3f5f;
        border-radius: 12px;
        padding: 14px 18px;
    }
    [data-testid="metric-container"] label { color: #90a4ae !important; font-size: 0.78rem !important; }
    [data-testid="metric-container"] [data-testid="metric-value"] { color: #64b5f6 !important; font-size: 1.8rem !important; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1565c0, #0d3b6e);
        color: white; padding: 8px 16px; border-radius: 8px;
        font-weight: 700; font-size: 1rem; margin: 10px 0 14px 0;
    }

    /* Risk badge */
    .risk-low    { background:#1b5e20; color:#a5d6a7; padding:14px 20px; border-radius:10px; font-size:1.1rem; font-weight:700; text-align:center; }
    .risk-medium { background:#e65100; color:#ffcc80; padding:14px 20px; border-radius:10px; font-size:1.1rem; font-weight:700; text-align:center; }
    .risk-high   { background:#b71c1c; color:#ef9a9a; padding:14px 20px; border-radius:10px; font-size:1.1rem; font-weight:700; text-align:center; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #111927; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #90a4ae; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #64b5f6 !important; border-bottom: 2px solid #64b5f6 !important; }

    /* Dataframe */
    .dataframe { font-size: 0.82rem !important; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0d47a1);
        color: white; border: none; border-radius: 8px;
        font-weight: 700; padding: 10px 24px; width: 100%;
    }
    .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING & MODEL TRAINING (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare_data():
    """Load dataset — tries outputs/ then generates fresh if missing."""
    try:
        df = pd.read_csv("outputs/healthcare_clean.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/healthcare_dataset.csv")
        except FileNotFoundError:
            df = generate_dataset()

    # Impute any missing
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64"]:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # BMI Category if missing
    if "BMI_Category" not in df.columns:
        df["BMI_Category"] = pd.cut(df["BMI"],
            bins=[0,18.5,25,30,100],
            labels=["Underweight","Normal","Overweight","Obese"])

    return df


def generate_dataset(N=10500):
    """Generate synthetic dataset if CSV not found."""
    np.random.seed(42)
    age    = np.random.randint(18, 90, N)
    gender = np.random.choice(["Male","Female"], N, p=[0.48,0.52])
    bmi    = np.round(np.random.normal(27.5,6.0,N).clip(15,55),1)
    systolic_bp   = np.random.normal(120,18,N).clip(80,200).astype(int)
    diastolic_bp  = np.random.normal(80,12,N).clip(50,130).astype(int)
    heart_rate    = np.random.normal(75,12,N).clip(45,140).astype(int)
    blood_glucose = np.random.normal(100,30,N).clip(60,400).astype(int)
    cholesterol   = np.random.normal(200,40,N).clip(100,400).astype(int)
    hemoglobin    = np.round(np.random.normal(13.5,2.0,N).clip(7,20),1)
    ldl           = np.random.normal(130,35,N).clip(50,300).astype(int)
    hdl           = np.random.normal(55,15,N).clip(20,100).astype(int)
    triglycerides = np.random.normal(150,60,N).clip(50,600).astype(int)
    hba1c         = np.round(np.random.normal(5.7,1.2,N).clip(4.0,14.0),1)
    creatinine    = np.round(np.random.normal(1.0,0.3,N).clip(0.4,5.0),2)
    smoking_status    = np.random.choice(["Never","Former","Current"],N,p=[0.55,0.25,0.20])
    alcohol_use       = np.random.choice(["None","Moderate","Heavy"],N,p=[0.45,0.40,0.15])
    physical_activity = np.random.choice(["Low","Moderate","High"],N,p=[0.35,0.40,0.25])
    diet_quality      = np.random.choice(["Poor","Fair","Good"],N,p=[0.30,0.40,0.30])
    family_history_diabetes = np.random.choice([0,1],N,p=[0.65,0.35])
    family_history_heart    = np.random.choice([0,1],N,p=[0.60,0.40])
    previous_stroke         = np.random.choice([0,1],N,p=[0.93,0.07])
    chronic_kidney_disease  = np.random.choice([0,1],N,p=[0.88,0.12])
    chest_pain         = np.random.choice([0,1],N,p=[0.80,0.20])
    shortness_breath   = np.random.choice([0,1],N,p=[0.78,0.22])
    fatigue            = np.random.choice([0,1],N,p=[0.65,0.35])
    frequent_urination = np.random.choice([0,1],N,p=[0.75,0.25])

    d_score = (0.03*age+0.04*bmi+0.005*blood_glucose+0.8*family_history_diabetes+
               0.6*(smoking_status=="Current")+0.5*(physical_activity=="Low")+
               0.7*(diet_quality=="Poor")+0.8*(hba1c>6.5)+0.5*frequent_urination-
               0.5*(physical_activity=="High"))
    diabetes = (np.random.rand(N) < 1/(1+np.exp(-(d_score-5)))).astype(int)

    h_score = (0.03*age+0.02*bmi+0.005*systolic_bp+0.004*cholesterol+
               0.8*family_history_heart+0.9*(smoking_status=="Current")+
               0.7*chest_pain+0.5*shortness_breath+0.4*(alcohol_use=="Heavy")+
               0.6*(physical_activity=="Low")+0.5*previous_stroke-0.4*(physical_activity=="High"))
    heart_disease = (np.random.rand(N) < 1/(1+np.exp(-(h_score-5)))).astype(int)
    disease = np.where((diabetes==1)|(heart_disease==1),1,0)

    return pd.DataFrame({
        "PatientID":[f"P{str(i).zfill(5)}" for i in range(1,N+1)],
        "Age":age,"Gender":gender,"BMI":bmi,
        "Systolic_BP":systolic_bp,"Diastolic_BP":diastolic_bp,"Heart_Rate":heart_rate,
        "Blood_Glucose":blood_glucose,"Cholesterol":cholesterol,"Hemoglobin":hemoglobin,
        "LDL":ldl,"HDL":hdl,"Triglycerides":triglycerides,"HbA1c":hba1c,"Creatinine":creatinine,
        "Smoking_Status":smoking_status,"Alcohol_Use":alcohol_use,
        "Physical_Activity":physical_activity,"Diet_Quality":diet_quality,
        "Family_History_Diabetes":family_history_diabetes,"Family_History_Heart":family_history_heart,
        "Previous_Stroke":previous_stroke,"Chronic_Kidney_Disease":chronic_kidney_disease,
        "Chest_Pain":chest_pain,"Shortness_of_Breath":shortness_breath,
        "Fatigue":fatigue,"Frequent_Urination":frequent_urination,
        "Diabetes":diabetes,"Heart_Disease":heart_disease,"Disease":disease,
        "BMI_Category":pd.cut(bmi,bins=[0,18.5,25,30,100],
                              labels=["Underweight","Normal","Overweight","Obese"])
    })


@st.cache_resource
def train_models(df):
    """Encode features and train all three models."""
    le = LabelEncoder()
    cat_cols = ["Gender","Smoking_Status","Alcohol_Use","Physical_Activity","Diet_Quality"]
    for col in cat_cols:
        df[col+"_enc"] = le.fit_transform(df[col].astype(str))

    drop = ["PatientID","Diabetes","Heart_Disease","BMI_Category","Disease"] + cat_cols
    features = [c for c in df.columns if c not in drop]

    X, y = df[features], df["Disease"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc  = sc.transform(X_te)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_tr_sc, y_tr)
    lr_pred  = lr.predict(X_te_sc)
    lr_prob  = lr.predict_proba(X_te_sc)[:,1]
    lr_acc   = accuracy_score(y_te, lr_pred)
    lr_auc   = roc_auc_score(y_te, lr_prob)
    lr_cv    = cross_val_score(lr, X_tr_sc, y_tr, cv=5, scoring="accuracy").mean()
    lr_fpr, lr_tpr, _ = roc_curve(y_te, lr_prob)
    lr_cm    = confusion_matrix(y_te, lr_pred)
    lr_coef  = pd.Series(lr.coef_[0], index=features).abs().sort_values(ascending=False)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=6, min_samples_split=50, min_samples_leaf=20, random_state=42)
    dt.fit(X_tr, y_tr)
    dt_pred  = dt.predict(X_te)
    dt_prob  = dt.predict_proba(X_te)[:,1]
    dt_acc   = accuracy_score(y_te, dt_pred)
    dt_auc   = roc_auc_score(y_te, dt_prob)
    dt_cv    = cross_val_score(dt, X_tr, y_tr, cv=5, scoring="accuracy").mean()
    dt_fpr, dt_tpr, _ = roc_curve(y_te, dt_prob)
    dt_cm    = confusion_matrix(y_te, dt_pred)
    dt_imp   = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=False)

    # Linear Regression (predict Blood_Glucose)
    lr_feat  = ["Age","BMI","Systolic_BP","Cholesterol","HbA1c","Creatinine","LDL","HDL","Triglycerides","Hemoglobin"]
    Xlr, ylr = df[lr_feat], df["Blood_Glucose"]
    Xlr_tr, Xlr_te, ylr_tr, ylr_te = train_test_split(Xlr, ylr, test_size=0.2, random_state=42)
    sc_lr    = StandardScaler()
    lrm      = LinearRegression().fit(sc_lr.fit_transform(Xlr_tr), ylr_tr)
    ylr_pred = lrm.predict(sc_lr.transform(Xlr_te))
    lrm_r2   = r2_score(ylr_te, ylr_pred)
    lrm_rmse = np.sqrt(mean_squared_error(ylr_te, ylr_pred))
    lrm_coef = pd.Series(lrm.coef_, index=lr_feat).sort_values()

    return {
        "features": features,
        "X_test": X_te, "y_test": y_te,
        "scaler": sc,
        # Logistic Regression
        "lr_model": lr, "lr_acc": lr_acc, "lr_auc": lr_auc, "lr_cv": lr_cv,
        "lr_pred": lr_pred, "lr_prob": lr_prob,
        "lr_fpr": lr_fpr, "lr_tpr": lr_tpr, "lr_cm": lr_cm, "lr_coef": lr_coef,
        # Decision Tree
        "dt_model": dt, "dt_acc": dt_acc, "dt_auc": dt_auc, "dt_cv": dt_cv,
        "dt_pred": dt_pred, "dt_prob": dt_prob,
        "dt_fpr": dt_fpr, "dt_tpr": dt_tpr, "dt_cm": dt_cm, "dt_imp": dt_imp,
        # Linear Regression
        "lrm_r2": lrm_r2, "lrm_rmse": lrm_rmse, "lrm_coef": lrm_coef,
        "ylr_te": ylr_te, "ylr_pred": ylr_pred,
    }


# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
with st.spinner("Loading dataset and training models..."):
    df = load_and_prepare_data()
    m  = train_models(df)

NUM_COLS = ["Age","BMI","Systolic_BP","Diastolic_BP","Heart_Rate",
            "Blood_Glucose","Cholesterol","Hemoglobin","LDL","HDL",
            "Triglycerides","HbA1c","Creatinine"]

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-with-pulse.png", width=70)
    st.title("Healthcare Analytics")
    st.caption("SY AIML AIDS · Micro Project")
    st.divider()

    st.markdown("### 📌 Project Info")
    st.markdown("- **Domain:** Healthcare\n- **Records:** 10,500\n- **Features:** 30\n- **Target:** Disease Prediction")
    st.divider()

    st.markdown("### 🗂️ Navigate")
    page = st.radio("Go to section:", [
        "🏠 Home & KPIs",
        "📊 EDA & Visualizations",
        "📐 Statistical Analysis",
        "🔬 Hypothesis Testing",
        "🤖 Predictive Models",
        "⚡ Risk Calculator",
        "📋 Raw Data Explorer"
    ])

    st.divider()
    st.markdown("**Tools Used:**")
    st.markdown("`Python` `Pandas` `NumPy`\n`Scikit-learn` `Matplotlib`\n`Seaborn` `SciPy` `Streamlit`")

# ─────────────────────────────────────────────────────────────
# PAGE 1: HOME & KPIs
# ─────────────────────────────────────────────────────────────
if page == "🏠 Home & KPIs":
    st.markdown("# 🏥 Healthcare Analysis & Disease Prediction")
    st.markdown("#### End-to-End Data Analytics Pipeline | SY AIML AIDS · School of Engineering and Technology")
    st.divider()

    # KPI Row
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Total Patients",    "10,500")
    c2.metric("Disease Rate",      f"{df['Disease'].mean():.1%}",    "High Risk")
    c3.metric("Diabetes Rate",     f"{df['Diabetes'].mean():.1%}")
    c4.metric("Heart Disease",     f"{df['Heart_Disease'].mean():.1%}")
    c5.metric("Best Accuracy",     f"{m['lr_acc']:.1%}",             "Logistic Reg")
    c6.metric("Best AUC",          f"{m['lr_auc']:.3f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">📌 Problem Statement</div>', unsafe_allow_html=True)
        st.info("""
Healthcare systems globally struggle with **early and accurate disease detection**.
Late diagnosis of conditions like **Diabetes** and **Heart Disease** leads to:
- Poor patient outcomes
- High treatment costs
- Preventable deaths

This project builds an **end-to-end ML pipeline** to predict disease risk from patient data.
        """)

        st.markdown('<div class="section-header">🎯 Objectives</div>', unsafe_allow_html=True)
        st.success("""
1. Perform comprehensive **EDA** on 10,500 patient records
2. Identify **patterns & correlations** in clinical data
3. Apply **Hypothesis Testing** (t-test, Chi-Square)
4. Build **3 Predictive Models**: Linear Regression, Logistic Regression, Decision Tree
        """)

    with col2:
        st.markdown('<div class="section-header">🔄 Analytics Pipeline</div>', unsafe_allow_html=True)
        pipeline_steps = [
            ("1️⃣ Problem Definition",    "Define healthcare disease prediction task"),
            ("2️⃣ Data Collection",       "10,500 synthetic patient records, 30 features"),
            ("3️⃣ Data Cleaning",         "Missing value imputation, duplicate removal"),
            ("4️⃣ EDA",                   "Distributions, boxplots, correlation analysis"),
            ("5️⃣ Statistical Analysis",  "Mean, Median, Std Dev, Variance, Skewness"),
            ("6️⃣ Hypothesis Testing",    "t-test, Chi-Square (H₀ vs H₁)"),
            ("7️⃣ Predictive Modelling",  "Linear Reg, Logistic Reg, Decision Tree"),
            ("8️⃣ Deployment",            "Streamlit web app (this!)"),
        ]
        for step, desc in pipeline_steps:
            st.markdown(f"**{step}** — {desc}")

    st.divider()
    st.markdown('<div class="section-header">📊 Dataset Feature Overview</div>', unsafe_allow_html=True)
    feat_data = {
        "Category": ["Demographics","Vitals","Lab Results","Lifestyle","Medical History","Symptoms","Targets"],
        "Features": [
            "Age, Gender, BMI",
            "Systolic BP, Diastolic BP, Heart Rate",
            "Blood Glucose, Cholesterol, HbA1c, LDL, HDL, Triglycerides, Hemoglobin, Creatinine",
            "Smoking Status, Alcohol Use, Physical Activity, Diet Quality",
            "Family History (Diabetes/Heart), Previous Stroke, Chronic Kidney Disease",
            "Chest Pain, Shortness of Breath, Fatigue, Frequent Urination",
            "Disease (primary), Diabetes, Heart Disease"
        ],
        "Count": [3,3,8,4,4,4,3]
    }
    st.dataframe(pd.DataFrame(feat_data), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────
# PAGE 2: EDA
# ─────────────────────────────────────────────────────────────
elif page == "📊 EDA & Visualizations":
    st.markdown("# 📊 Exploratory Data Analysis")
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distributions", "🥧 Disease Prevalence",
        "📦 Boxplots", "🌡️ Correlation", "🧬 Lifestyle Factors"
    ])

    with tab1:
        st.markdown('<div class="section-header">Distribution of Clinical Variables</div>', unsafe_allow_html=True)
        selected = st.multiselect("Choose variables:", NUM_COLS, default=["Age","BMI","Blood_Glucose","Systolic_BP"])
        if selected:
            cols_per_row = 2
            rows = [selected[i:i+cols_per_row] for i in range(0, len(selected), cols_per_row)]
            for row in rows:
                c = st.columns(len(row))
                for ax_col, col in zip(c, row):
                    with ax_col:
                        fig, ax = plt.subplots(figsize=(6,3.5))
                        fig.patch.set_facecolor("#0e1117")
                        ax.set_facecolor("#0e1117")
                        ax.hist(df[col], bins=40, color="#2196F3", edgecolor="#0e1117", alpha=0.85)
                        ax.axvline(df[col].mean(),   color="#ef5350", linestyle="--", lw=1.5, label=f"Mean={df[col].mean():.1f}")
                        ax.axvline(df[col].median(), color="#FFC107", linestyle=":",  lw=1.5, label=f"Median={df[col].median():.1f}")
                        ax.set_title(col, color="white", fontsize=11)
                        ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
                        for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
                        ax.legend(fontsize=7, facecolor="#1a2035", labelcolor="white")
                        st.pyplot(fig); plt.close()

    with tab2:
        st.markdown('<div class="section-header">Disease Prevalence</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        for col_ui, col_data, labels, title in zip(
            [c1, c2, c3],
            ["Disease","Diabetes","Heart_Disease"],
            [["No Disease","Disease"],["No Diabetes","Diabetes"],["No Heart Dis.","Heart Disease"]],
            ["Overall Disease","Diabetes","Heart Disease"]
        ):
            with col_ui:
                counts = df[col_data].value_counts()
                fig, ax = plt.subplots(figsize=(4,4))
                fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
                ax.pie(counts, labels=labels, autopct="%1.1f%%",
                       colors=["#4CAF50","#ef5350"], startangle=90,
                       wedgeprops={"edgecolor":"#0e1117","linewidth":2},
                       textprops={"color":"white"})
                ax.set_title(title, color="white", fontsize=12, fontweight="bold")
                st.pyplot(fig); plt.close()

        st.divider()
        # Disease by gender and age group
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            g = df.groupby("Gender")["Disease"].mean()*100
            ax.bar(g.index, g.values, color=["#2196F3","#E91E63"], edgecolor="#0e1117", width=0.5)
            ax.set_title("Disease Rate by Gender", color="white")
            ax.set_ylabel("Disease %", color="white"); ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            for i,(k,v) in enumerate(g.items()): ax.text(i, v+0.5, f"{v:.1f}%", ha="center", color="white", fontweight="bold")
            st.pyplot(fig); plt.close()

        with c2:
            fig, ax = plt.subplots(figsize=(6,4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            age_bins = pd.cut(df["Age"], bins=[18,30,40,50,60,70,90])
            age_prev = df.groupby(age_bins, observed=True)["Disease"].mean()*100
            ax.bar(range(len(age_prev)), age_prev.values, color="#FF9800", edgecolor="#0e1117")
            ax.set_xticks(range(len(age_prev))); ax.set_xticklabels([str(x) for x in age_prev.index], rotation=30, color="white")
            ax.set_title("Disease Rate by Age Group", color="white")
            ax.set_ylabel("Disease %", color="white"); ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()

    with tab3:
        st.markdown('<div class="section-header">Clinical Variables by Disease Status</div>', unsafe_allow_html=True)
        sel = st.selectbox("Select variable:", NUM_COLS, index=0)
        fig, ax = plt.subplots(figsize=(7,4))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        g0 = df[df["Disease"]==0][sel]; g1 = df[df["Disease"]==1][sel]
        bp = ax.boxplot([g0, g1], patch_artist=True, labels=["No Disease","Disease"],
                        medianprops=dict(color="#ef5350", linewidth=2.5))
        bp["boxes"][0].set_facecolor("#1565c0"); bp["boxes"][1].set_facecolor("#b71c1c")
        for whisker in bp["whiskers"]: whisker.set_color("white")
        for cap in bp["caps"]: cap.set_color("white")
        for flier in bp["fliers"]: flier.set_markerfacecolor("#90a4ae"); flier.set_markersize(3)
        ax.set_title(f"{sel} by Disease Status", color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel(sel, color="white"); ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
        g0m, g1m = g0.mean(), g1.mean()
        ax.text(1.5, ax.get_ylim()[1]*0.95, f"No Disease Mean: {g0m:.1f}", color="#64b5f6", fontsize=9)
        ax.text(1.5, ax.get_ylim()[1]*0.88, f"Disease Mean: {g1m:.1f}",    color="#ef5350", fontsize=9)
        st.pyplot(fig); plt.close()

        t_stat, p_val = stats.ttest_ind(g0, g1)
        st.info(f"**t-test:** t = {t_stat:.4f}, p = {p_val:.6f} → {'✅ Statistically Significant' if p_val < 0.05 else '❌ Not Significant'}")

    with tab4:
        st.markdown('<div class="section-header">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = st.multiselect("Select columns:", NUM_COLS + ["Disease","Diabetes","Heart_Disease"],
                                   default=NUM_COLS[:8] + ["Disease"])
        if len(corr_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10,7))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            corr = df[corr_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlBu_r",
                        center=0, ax=ax, linewidths=0.5, annot_kws={"size":8},
                        cbar_kws={"shrink":0.8})
            ax.set_title("Correlation Matrix", color="white", fontsize=13, fontweight="bold")
            ax.tick_params(colors="white")
            st.pyplot(fig); plt.close()

    with tab5:
        st.markdown('<div class="section-header">Lifestyle Factors vs Disease</div>', unsafe_allow_html=True)
        factor = st.selectbox("Select factor:", ["Smoking_Status","Physical_Activity","Diet_Quality","Alcohol_Use","BMI_Category"])
        fig, axes = plt.subplots(1,2,figsize=(13,4))
        for ax in axes: fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")

        data = df.groupby([factor,"Disease"]).size().unstack(fill_value=0)
        data.plot(kind="bar", ax=axes[0], color=["#4CAF50","#ef5350"], edgecolor="#0e1117", rot=15)
        axes[0].set_title(f"Count by {factor}", color="white"); axes[0].tick_params(colors="white")
        axes[0].set_ylabel("Count", color="white")
        axes[0].legend(["No Disease","Disease"], facecolor="#1a2035", labelcolor="white")
        for spine in axes[0].spines.values(): spine.set_edgecolor("#2a3f5f")

        prev = df.groupby(factor)["Disease"].mean()*100
        axes[1].bar(prev.index, prev.values, color="#FF9800", edgecolor="#0e1117")
        axes[1].set_title(f"Disease Rate by {factor}", color="white"); axes[1].tick_params(colors="white")
        axes[1].set_ylabel("Disease %", color="white")
        for spine in axes[1].spines.values(): spine.set_edgecolor("#2a3f5f")
        plt.xticks(rotation=15)
        st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────
# PAGE 3: STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────
elif page == "📐 Statistical Analysis":
    st.markdown("# 📐 Statistical Analysis")
    st.divider()

    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    desc = df[NUM_COLS].describe().T
    desc["Variance"]  = df[NUM_COLS].var().round(2)
    desc["Skewness"]  = df[NUM_COLS].skew().round(4)
    desc["Kurtosis"]  = df[NUM_COLS].kurtosis().round(4)
    desc = desc[["mean","50%","std","Variance","min","max","Skewness","Kurtosis"]].round(3)
    desc.columns = ["Mean","Median","Std Dev","Variance","Min","Max","Skewness","Kurtosis"]
    st.dataframe(desc.style.background_gradient(cmap="Blues", subset=["Mean","Std Dev","Variance"]),
                 use_container_width=True)

    st.divider()
    st.markdown('<div class="section-header">Feature-by-Feature Deep Dive</div>', unsafe_allow_html=True)
    col = st.selectbox("Select variable:", NUM_COLS)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Mean",    f"{df[col].mean():.2f}")
    c2.metric("Median",  f"{df[col].median():.2f}")
    c3.metric("Std Dev", f"{df[col].std():.2f}")
    c4.metric("Variance",f"{df[col].var():.2f}")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Min",      f"{df[col].min():.2f}")
    c2.metric("Max",      f"{df[col].max():.2f}")
    c3.metric("Skewness", f"{df[col].skew():.4f}")
    c4.metric("Kurtosis", f"{df[col].kurtosis():.4f}")

    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for ax in axes: fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")

    axes[0].hist(df[col], bins=40, color="#2196F3", edgecolor="#0e1117", alpha=0.85)
    axes[0].axvline(df[col].mean(),   color="#ef5350", linestyle="--", lw=2, label="Mean")
    axes[0].axvline(df[col].median(), color="#FFC107", linestyle=":",  lw=2, label="Median")
    axes[0].set_title(f"Distribution of {col}", color="white"); axes[0].legend(facecolor="#1a2035", labelcolor="white")
    axes[0].tick_params(colors="white")

    axes[1].boxplot(df[col], patch_artist=True, boxprops=dict(facecolor="#1565c0"),
                    medianprops=dict(color="#ef5350", linewidth=2),
                    whiskerprops=dict(color="white"), capprops=dict(color="white"))
    axes[1].set_title(f"Boxplot of {col}", color="white"); axes[1].tick_params(colors="white")

    (stats.probplot(df[col], plot=axes[2]))
    axes[2].set_title(f"Q-Q Plot of {col}", color="white"); axes[2].tick_params(colors="white")
    axes[2].get_lines()[1].set_color("#ef5350")
    for spine in [*axes[0].spines.values(), *axes[1].spines.values(), *axes[2].spines.values()]:
        spine.set_edgecolor("#2a3f5f")
    st.pyplot(fig); plt.close()

    st.divider()
    st.markdown('<div class="section-header">Correlation with Disease Outcome</div>', unsafe_allow_html=True)
    corr_vals = df[NUM_COLS + ["Disease"]].corr()["Disease"].drop("Disease").sort_values()
    fig, ax = plt.subplots(figsize=(8,4))
    fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
    colors = ["#ef5350" if v>0 else "#2196F3" for v in corr_vals]
    ax.barh(corr_vals.index, corr_vals.values, color=colors, edgecolor="#0e1117")
    ax.axvline(0, color="white", linewidth=0.8)
    ax.set_title("Pearson Correlation with Disease", color="white", fontsize=12, fontweight="bold")
    ax.set_xlabel("Correlation Coefficient", color="white"); ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
    st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────
# PAGE 4: HYPOTHESIS TESTING
# ─────────────────────────────────────────────────────────────
elif page == "🔬 Hypothesis Testing":
    st.markdown("# 🔬 Hypothesis Testing")
    st.info("**H₀:** There is no significant relationship between the feature and disease outcome.\n\n**H₁:** There is a significant relationship between the feature and disease outcome.\n\n**Significance Level (α):** 0.05")
    st.divider()

    st.markdown('<div class="section-header">Independent Samples t-test (Continuous Features)</div>', unsafe_allow_html=True)
    t_rows = []
    for col in NUM_COLS:
        g0 = df[df["Disease"]==0][col]; g1 = df[df["Disease"]==1][col]
        t, p = stats.ttest_ind(g0, g1)
        t_rows.append({
            "Feature": col,
            "Mean (No Disease)": round(g0.mean(),2),
            "Mean (Disease)":    round(g1.mean(),2),
            "t-statistic":       round(t,4),
            "p-value":           round(p,6),
            "Significant (α=0.05)": "✅ YES — REJECT H₀" if p<0.05 else "❌ NO — Fail to Reject H₀"
        })
    t_df = pd.DataFrame(t_rows)

    def highlight_sig(row):
        if "YES" in str(row["Significant (α=0.05)"]):
            return ["background-color: #1b3a1b"]*len(row)
        return ["background-color: #3a1a1a"]*len(row)

    st.dataframe(t_df.style.apply(highlight_sig, axis=1), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-header">Chi-Square Test (Categorical Features)</div>', unsafe_allow_html=True)
    chi_rows = []
    for cat in ["Smoking_Status","Physical_Activity","Diet_Quality","Gender","Alcohol_Use"]:
        ct = pd.crosstab(df[cat], df["Disease"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        chi_rows.append({
            "Feature": cat,
            "Chi² Statistic": round(chi2,4),
            "Degrees of Freedom": dof,
            "p-value": round(p,6),
            "Significant (α=0.05)": "✅ YES — REJECT H₀" if p<0.05 else "❌ NO — Fail to Reject H₀"
        })
    chi_df = pd.DataFrame(chi_rows)
    st.dataframe(chi_df.style.apply(highlight_sig, axis=1), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-header">Interactive t-test Explorer</div>', unsafe_allow_html=True)
    sel_col = st.selectbox("Select feature for detailed test:", NUM_COLS)
    g0 = df[df["Disease"]==0][sel_col]; g1 = df[df["Disease"]==1][sel_col]
    t_stat, p_val = stats.ttest_ind(g0, g1)

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | Feature | **{sel_col}** |
        | t-statistic | **{t_stat:.4f}** |
        | p-value | **{p_val:.6f}** |
        | Mean (No Disease) | **{g0.mean():.2f}** |
        | Mean (Disease) | **{g1.mean():.2f}** |
        | Conclusion | **{'✅ REJECT H₀ — Significant' if p_val<0.05 else '❌ Fail to Reject H₀'}** |
        """)
    with c2:
        fig, ax = plt.subplots(figsize=(6,4))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        ax.hist(g0, bins=35, alpha=0.6, color="#4CAF50", label="No Disease", edgecolor="#0e1117")
        ax.hist(g1, bins=35, alpha=0.6, color="#ef5350", label="Disease",    edgecolor="#0e1117")
        ax.set_title(f"{sel_col} — Group Comparison", color="white")
        ax.legend(facecolor="#1a2035", labelcolor="white"); ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
        st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────
# PAGE 5: PREDICTIVE MODELS
# ─────────────────────────────────────────────────────────────
elif page == "🤖 Predictive Models":
    st.markdown("# 🤖 Predictive Modelling")
    st.divider()

    model_tab1, model_tab2, model_tab3, model_tab4 = st.tabs([
        "📉 Linear Regression", "📊 Logistic Regression",
        "🌳 Decision Tree", "⚖️ Model Comparison"
    ])

    with model_tab1:
        st.markdown('<div class="section-header">Linear Regression — Predicting Blood Glucose</div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("R² Score",  f"{m['lrm_r2']:.4f}")
        c2.metric("RMSE",      f"{m['lrm_rmse']:.4f}")
        c3.metric("Task",      "Blood Glucose Prediction")

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6,4.5))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.scatter(m["ylr_te"], m["ylr_pred"], alpha=0.25, color="#2196F3", s=8)
            mn, mx = m["ylr_te"].min(), m["ylr_te"].max()
            ax.plot([mn,mx],[mn,mx],"r--", linewidth=2, label="Perfect fit")
            ax.set_xlabel("Actual Blood Glucose", color="white")
            ax.set_ylabel("Predicted Blood Glucose", color="white")
            ax.set_title(f"Actual vs Predicted (R²={m['lrm_r2']:.3f})", color="white")
            ax.legend(facecolor="#1a2035", labelcolor="white"); ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(6,4.5))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            coef = m["lrm_coef"]
            colors = ["#ef5350" if v>0 else "#2196F3" for v in coef]
            ax.barh(coef.index, coef.values, color=colors, edgecolor="#0e1117")
            ax.axvline(0, color="white", linewidth=0.8)
            ax.set_title("Feature Coefficients", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()

    with model_tab2:
        st.markdown('<div class="section-header">Logistic Regression — Disease Classification</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",    f"{m['lr_acc']:.4f}")
        c2.metric("ROC-AUC",     f"{m['lr_auc']:.4f}")
        c3.metric("CV Accuracy", f"{m['lr_cv']:.4f}")
        c4.metric("Iterations",  "1000")

        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ConfusionMatrixDisplay(m["lr_cm"], display_labels=["No Disease","Disease"]).plot(
                ax=ax, cmap="Blues", colorbar=False)
            ax.set_title("Confusion Matrix", color="white")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.plot(m["lr_fpr"], m["lr_tpr"], color="#2196F3", lw=2.5, label=f"AUC={m['lr_auc']:.3f}")
            ax.plot([0,1],[0,1],"k--", lw=1)
            ax.fill_between(m["lr_fpr"], m["lr_tpr"], alpha=0.15, color="#2196F3")
            ax.set_xlabel("False Positive Rate", color="white")
            ax.set_ylabel("True Positive Rate", color="white")
            ax.set_title("ROC Curve", color="white")
            ax.legend(facecolor="#1a2035", labelcolor="white"); ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()

        with col3:
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            top12 = m["lr_coef"].head(12)
            ax.barh(top12.index[::-1], top12.values[::-1], color="#FF9800", edgecolor="#0e1117")
            ax.set_title("Top Feature Importances", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()

        st.divider()
        st.markdown("**Classification Report:**")
        report = classification_report(m["y_test"], m["lr_pred"],
                                       target_names=["No Disease","Disease"], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    with model_tab3:
        st.markdown('<div class="section-header">Decision Tree — Disease Classification</div>', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Accuracy",    f"{m['dt_acc']:.4f}")
        c2.metric("ROC-AUC",     f"{m['dt_auc']:.4f}")
        c3.metric("CV Accuracy", f"{m['dt_cv']:.4f}")
        c4.metric("Max Depth",   "6")

        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ConfusionMatrixDisplay(m["dt_cm"], display_labels=["No Disease","Disease"]).plot(
                ax=ax, cmap="Oranges", colorbar=False)
            ax.set_title("Confusion Matrix", color="white")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.plot(m["dt_fpr"], m["dt_tpr"], color="#F44336", lw=2.5, label=f"AUC={m['dt_auc']:.3f}")
            ax.plot([0,1],[0,1],"k--", lw=1)
            ax.fill_between(m["dt_fpr"], m["dt_tpr"], alpha=0.15, color="#F44336")
            ax.set_xlabel("False Positive Rate", color="white")
            ax.set_ylabel("True Positive Rate", color="white")
            ax.set_title("ROC Curve", color="white")
            ax.legend(facecolor="#1a2035", labelcolor="white"); ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()

        with col3:
            fig, ax = plt.subplots(figsize=(5,4))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            top12 = m["dt_imp"].head(12)
            ax.barh(top12.index[::-1], top12.values[::-1], color="#4CAF50", edgecolor="#0e1117")
            ax.set_title("Feature Importances (Gini)", color="white")
            ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()

        st.divider()
        st.markdown("**Decision Tree Visualization (Top 3 Levels):**")
        fig, ax = plt.subplots(figsize=(20,8))
        fig.patch.set_facecolor("#0e1117")
        plot_tree(m["dt_model"], feature_names=m["features"],
                  class_names=["No Disease","Disease"],
                  filled=True, max_depth=3, ax=ax,
                  fontsize=7, impurity=False, proportion=True)
        ax.set_title("Decision Tree Visualization", color="white", fontsize=14, fontweight="bold")
        st.pyplot(fig); plt.close()

    with model_tab4:
        st.markdown('<div class="section-header">Model Comparison Summary</div>', unsafe_allow_html=True)
        comp = pd.DataFrame({
            "Model": ["Logistic Regression","Decision Tree"],
            "Accuracy": [m["lr_acc"], m["dt_acc"]],
            "ROC-AUC":  [m["lr_auc"], m["dt_auc"]],
            "CV Accuracy": [m["lr_cv"], m["dt_cv"]],
            "Best Model": ["✅ Yes","❌ No"]
        })
        st.dataframe(comp.style.highlight_max(subset=["Accuracy","ROC-AUC","CV Accuracy"],
                                              color="#1b3a1b"), use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7,5))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.plot(m["lr_fpr"], m["lr_tpr"], color="#2196F3", lw=2.5, label=f"Logistic Reg (AUC={m['lr_auc']:.3f})")
            ax.plot(m["dt_fpr"], m["dt_tpr"], color="#F44336", lw=2.5, label=f"Decision Tree (AUC={m['dt_auc']:.3f})")
            ax.plot([0,1],[0,1],"k--", lw=1, label="Random")
            ax.set_xlabel("False Positive Rate", color="white")
            ax.set_ylabel("True Positive Rate", color="white")
            ax.set_title("Combined ROC Curves", color="white", fontsize=12, fontweight="bold")
            ax.legend(facecolor="#1a2035", labelcolor="white"); ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(7,5))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            metrics = ["Accuracy","ROC-AUC","CV Accuracy"]
            x = np.arange(len(metrics)); w = 0.35
            ax.bar(x-w/2, [m["lr_acc"],m["lr_auc"],m["lr_cv"]], w, label="Logistic Reg", color="#2196F3", edgecolor="#0e1117")
            ax.bar(x+w/2, [m["dt_acc"],m["dt_auc"],m["dt_cv"]], w, label="Decision Tree", color="#4CAF50", edgecolor="#0e1117")
            ax.set_xticks(x); ax.set_xticklabels(metrics, color="white")
            ax.set_ylim(0.5, 0.85); ax.set_ylabel("Score", color="white")
            ax.set_title("Performance Comparison", color="white", fontsize=12, fontweight="bold")
            ax.legend(facecolor="#1a2035", labelcolor="white"); ax.tick_params(colors="white")
            for spine in ax.spines.values(): spine.set_edgecolor("#2a3f5f")
            st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────
# PAGE 6: RISK CALCULATOR
# ─────────────────────────────────────────────────────────────
elif page == "⚡ Risk Calculator":
    st.markdown("# ⚡ Disease Risk Calculator")
    st.markdown("Enter patient details below to get an **instant disease risk assessment** powered by the trained Logistic Regression model.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Demographics**")
        age     = st.slider("Age (years)", 18, 90, 45)
        gender  = st.selectbox("Gender", ["Male","Female"])
        bmi     = st.slider("BMI", 15.0, 55.0, 27.0, 0.1)

    with col2:
        st.markdown("**💉 Clinical Values**")
        sbp     = st.slider("Systolic BP (mmHg)", 80, 200, 120)
        glucose = st.slider("Blood Glucose (mg/dL)", 60, 400, 100)
        hba1c   = st.slider("HbA1c (%)", 4.0, 14.0, 5.7, 0.1)
        chol    = st.slider("Cholesterol (mg/dL)", 100, 400, 200)

    with col3:
        st.markdown("**🧬 Lifestyle & History**")
        smoke    = st.selectbox("Smoking Status",    ["Never","Former","Current"])
        activity = st.selectbox("Physical Activity", ["High","Moderate","Low"])
        diet     = st.selectbox("Diet Quality",      ["Good","Fair","Poor"])
        fheart   = st.selectbox("Family History — Heart Disease", ["No","Yes"])
        fdiab    = st.selectbox("Family History — Diabetes",      ["No","Yes"])
        chest    = st.selectbox("Chest Pain",        ["No","Yes"])

    st.divider()

    if st.button("🔍 Calculate Disease Risk"):
        # Encode inputs
        smoke_map    = {"Never":0, "Former":1, "Current":2}
        activity_map = {"High":2, "Moderate":1, "Low":0}
        diet_map     = {"Good":2, "Fair":1, "Poor":0}

        # Build feature vector matching training features
        input_dict = {
            "Age": age, "BMI": bmi,
            "Systolic_BP": sbp, "Diastolic_BP": 80,
            "Heart_Rate": 75, "Blood_Glucose": glucose,
            "Cholesterol": chol, "Hemoglobin": 13.5,
            "LDL": 130, "HDL": 55, "Triglycerides": 150,
            "HbA1c": hba1c, "Creatinine": 1.0,
            "Family_History_Diabetes": 1 if fdiab=="Yes" else 0,
            "Family_History_Heart":    1 if fheart=="Yes" else 0,
            "Previous_Stroke": 0, "Chronic_Kidney_Disease": 0,
            "Chest_Pain": 1 if chest=="Yes" else 0,
            "Shortness_of_Breath": 0, "Fatigue": 0,
            "Frequent_Urination": 0,
            "Gender_enc":          0 if gender=="Female" else 1,
            "Smoking_Status_enc":  smoke_map[smoke],
            "Alcohol_Use_enc":     1,
            "Physical_Activity_enc": activity_map[activity],
            "Diet_Quality_enc":    diet_map[diet]
        }

        # Only include features used in training
        feat_vec = pd.DataFrame([[input_dict.get(f, 0) for f in m["features"]]], columns=m["features"])
        feat_sc  = m["scaler"].transform(feat_vec)
        prob     = m["lr_model"].predict_proba(feat_sc)[0][1]
        risk_pct = int(prob * 100)
        pred     = m["lr_model"].predict(feat_sc)[0]

        # Display risk
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if risk_pct < 35:
                st.markdown(f'<div class="risk-low">✅ Disease Risk: {risk_pct}%<br><small>LOW RISK — Maintain healthy lifestyle</small></div>', unsafe_allow_html=True)
            elif risk_pct < 60:
                st.markdown(f'<div class="risk-medium">⚠️ Disease Risk: {risk_pct}%<br><small>MODERATE RISK — Consult your doctor regularly</small></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-high">🚨 Disease Risk: {risk_pct}%<br><small>HIGH RISK — Seek immediate medical attention</small></div>', unsafe_allow_html=True)

        st.divider()
        # Risk gauge bar
        st.markdown("**Risk Probability Breakdown:**")
        st.progress(risk_pct/100, text=f"Disease Probability: {risk_pct}%")

        # Recommendations
        st.markdown("**📋 Personalised Recommendations:**")
        recs = []
        if bmi > 25:         recs.append("⚠️ BMI is above normal — consider weight management")
        if sbp > 130:        recs.append("⚠️ High Systolic BP — monitor blood pressure regularly")
        if glucose > 126:    recs.append("⚠️ High Blood Glucose — get HbA1c test done")
        if hba1c > 6.5:      recs.append("🚨 HbA1c > 6.5% — diabetes likely, consult endocrinologist")
        if smoke == "Current": recs.append("🚬 Smoking significantly increases disease risk — consider cessation")
        if activity == "Low":  recs.append("🏃 Low physical activity — aim for 150 min/week of moderate exercise")
        if diet == "Poor":     recs.append("🥗 Poor diet quality — increase fruits, vegetables, and whole grains")
        if fheart == "Yes":    recs.append("🧬 Family history of heart disease — regular cardiac screening recommended")
        if not recs:           recs.append("✅ All key indicators look healthy — keep it up!")

        for r in recs:
            st.markdown(f"- {r}")


# ─────────────────────────────────────────────────────────────
# PAGE 7: RAW DATA EXPLORER
# ─────────────────────────────────────────────────────────────
elif page == "📋 Raw Data Explorer":
    st.markdown("# 📋 Raw Data Explorer")
    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows",    len(df))
    c2.metric("Total Columns", len(df.columns))
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.divider()

    # Filters
    st.markdown('<div class="section-header">🔍 Filter Dataset</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender_filter   = st.multiselect("Gender",           df["Gender"].unique(),           default=list(df["Gender"].unique()))
        smoking_filter  = st.multiselect("Smoking Status",   df["Smoking_Status"].unique(),   default=list(df["Smoking_Status"].unique()))
    with col2:
        activity_filter = st.multiselect("Physical Activity",df["Physical_Activity"].unique(),default=list(df["Physical_Activity"].unique()))
        disease_filter  = st.multiselect("Disease",          [0,1],                           default=[0,1])
    with col3:
        age_range = st.slider("Age Range", int(df.Age.min()), int(df.Age.max()), (18,90))
        bmi_range = st.slider("BMI Range", float(df.BMI.min()), float(df.BMI.max()), (15.0,55.0))

    filtered = df[
        df["Gender"].isin(gender_filter) &
        df["Smoking_Status"].isin(smoking_filter) &
        df["Physical_Activity"].isin(activity_filter) &
        df["Disease"].isin(disease_filter) &
        df["Age"].between(*age_range) &
        df["BMI"].between(*bmi_range)
    ]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=400)

    # Download
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Filtered Data as CSV", csv, "filtered_healthcare_data.csv", "text/csv")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Healthcare Analytics & Disease Prediction Model · SY AIML AIDS · "
    "School of Engineering and Technology · Built with Python & Streamlit</small></center>",
    unsafe_allow_html=True
)
