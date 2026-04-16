# Heart Disease Risk Predictor — Streamlit UI
# Integrates Expert System (rule-based) + Decision Tree (ML model)

import os
import sys
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ── Path Setup ────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rule_based_system'))
from ExprtSystem import run_expert_system

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",

    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
html, body, * { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.main { background: #f8f5ff; }
.header {
    background: linear-gradient(135deg, #1e0a3c, #7b2ff7);
    padding: 36px 40px; border-radius: 20px; color: white;
    margin-bottom: 24px; text-align: center;
}
.header h1 { margin: 0; font-size: 26px; font-weight: 800; }
.header p  { margin: 6px 0 0; opacity: 0.75; font-size: 14px; }
.sec {
    font-size: 15px; font-weight: 700; color: #1e0a3c;
    margin: 22px 0 12px; padding-bottom: 8px;
    border-bottom: 2px solid #c4b5fd;
}
.rcard {
    padding: 20px; border-radius: 14px;
    text-align: center; font-size: 19px; font-weight: 700; margin: 6px 0;
}
.rhigh   { background:#fff0f0; color:#9b1c1c; border:2px solid #e02424; }
.rmedium { background:#fffbeb; color:#78350f; border:2px solid #f59e0b; }
.rlow    { background:#f0fdf4; color:#14532d; border:2px solid #22c55e; }
.sbox {
    background: white; border-radius: 12px; padding: 16px 10px;
    text-align: center; border: 1px solid #ede9fe;
}
.snum { font-size: 26px; font-weight: 800; }
.slbl { font-size: 11px; text-transform: uppercase; letter-spacing: 0.6px; margin-top: 2px; font-weight: 600; }
div.stButton > button {
    background: linear-gradient(135deg, #4a1a8f, #7b2ff7);
    color: white; border: none; border-radius: 12px;
    height: 50px; width: 100%; font-size: 16px; font-weight: 700;
}
div.stButton > button:hover { opacity: 0.85; }
.disc {
    margin-top: 24px; padding: 12px 16px; background: #f9fafb;
    border-radius: 10px; font-size: 12px; color: #6b7280; border: 1px solid #e5e7eb;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model & Scaler ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), '..', 'ml_model')
    mdl = joblib.load(os.path.join(base, 'model.pkl'))
    scl = joblib.load(os.path.join(base, 'scaler.pkl'))
    return mdl, scl

model, scaler = load_artifacts()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1> Heart Disease Risk Predictor</h1>
    <p>Expert System rules × Decision Tree ML — side-by-side risk assessment</p>
</div>
""", unsafe_allow_html=True)

# ── Inputs ────────────────────────────────────────────────────
st.markdown('<div class="sec"> Patient Health Data</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("** Vitals**")
    age      = st.slider("Age",                           29,  77,  50)
    trestbps = st.slider("Resting Blood Pressure (mmHg)", 94, 200, 130)
    chol     = st.slider("Cholesterol (mg/dL)",          126, 564, 250)

with col2:
    st.markdown("** Heart Metrics**")
    thalach = st.slider("Max Heart Rate Achieved",  71, 202, 150)
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)

with col3:
    st.markdown("** Symptoms**")
    exang = st.radio(
        "Exercise Induced Angina", ["No", "Yes"],
        help="Chest pain during exercise?"
    )

exang_1 = 1 if exang == "Yes" else 0
exang_0 = 1 - exang_1

st.markdown("")
predict_btn = st.button(" Predict Risk Now")

# ── Prediction 
if predict_btn:

   
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    raw_df   = pd.DataFrame([[age, trestbps, chol, thalach, oldpeak]], columns=num_cols)
    scaled   = scaler.transform(raw_df)[0]
    age_n, trestbps_n, chol_n, thalach_n, oldpeak_n = scaled

    # Expert System
    es_result = run_expert_system(
        age=age_n, trestbps=trestbps_n, chol=chol_n,
        thalach=thalach_n, oldpeak=oldpeak_n, exang_1=bool(exang_1)
    )

    # Decision Tree
    input_row = pd.DataFrame([{
    'age': age_n,
    'sex': 1 if exang_1 else 0,  
    'cp': 1,
    'trestbps': trestbps_n,
    'chol': chol,
    'fbs': 0,
    'restecg': 0,
    'thalach': thalach_n,
    'exang': exang_1,
    'oldpeak': oldpeak_n,
    'slope': 1,
    'ca': 0,
    'thal': 2
}])

    dt_pred  = model.predict(input_row)[0]
    dt_prob  = model.predict_proba(input_row)[0][1]
    dt_pct   = round(dt_prob * 100, 1)
    dt_label = "High" if dt_pred == 1 else "Low"
    es_risk  = es_result["final_risk"]
    counts   = es_result["counts"]

    # ── Results 
    st.markdown('<div class="sec"> Risk Assessment Results</div>', unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
   
     st.markdown(" Expert System")
     css_es  = "rhigh" if es_risk == "High" else ("rmedium" if es_risk == "Medium" else "rlow")
     st.markdown(f'<div class="rcard {css_es}">{es_risk} Risk</div>', unsafe_allow_html=True)

    # Decision Tree
     st.markdown(" Decision Tree")
     css_dt  = "rhigh" if dt_label == "High" else "rlow"
     st.markdown(f'<div class="rcard {css_dt}">{dt_label} Risk</div>', unsafe_allow_html=True)
     