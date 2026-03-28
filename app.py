import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import datetime
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Cardiac Wellness System", page_icon="⚕️", layout="wide")

# --- Elegant Mix of Light & Dark Gradients ---
gradients = [
    "linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%)", # Bright Ice Blue
    "linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%)", # Silver White
    "linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)", # Bright Sky
    "linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)", # Warm Dawn
    "linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)", # Mint Fresh
    "linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)",  # Lavender Blue
    "linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%)", # Deep oceanic
    "linear-gradient(135deg, #2b5876 0%, #4e4376 100%)", # Nebula purple
    "linear-gradient(135deg, #141e30 0%, #243b55 100%)", # Midnight city
]

if 'bg_gradient' not in st.session_state:
    st.session_state.bg_gradient = gradients[0]
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# --- Clean Light/Dark Glassmorphism CSS ---
st.markdown(f"""
<style>
/* Smooth Background Motion */
.stApp {{
    background: {st.session_state.bg_gradient} !important;
    background-size: 200% 200% !important;
    animation: gradientFlow 10s ease alternate infinite !important;
}}
@keyframes gradientFlow {{
    0% {{ background-position: 0% 50%; }}
    100% {{ background-position: 100% 50%; }}
}}
[data-testid="block-container"] {{
    background: rgba(255, 255, 255, 0.95); /* Extremely solid white card */
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 3rem !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.6);
    margin-top: 2rem !important;
    margin-bottom: 2rem !important;
}}
div.stButton > button:first-child {{
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    font-weight: 700 !important;
    padding: 15px 30px !important;
    width: 100% !important;
    font-size: 20px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    margin-top: 15px;
}}
div.stButton > button:first-child:hover {{
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
}}
div.stButton > button[disabled] {{
    background: #cbd5e1 !important;
    color: #475569 !important;
    box-shadow: none !important;
}}
h1 {{
    color: #1e3a8a !important;
    text-align: center;
    font-size: 3.5rem;
    font-weight: 900;
    margin-bottom: 5px;
    letter-spacing: -1px;
}}
.sub-header {{
    text-align: center;
    color: #475569 !important;
    font-size: 1.2rem;
    margin-bottom: 40px;
}}

/* Force all form text, labels, and paragraph readouts to be extremely high contrast dark blue */
p, label, span, .stMarkdown, .stText, li {{
    color: #0f172a !important;
    font-weight: 500;
}}
/* Protect our specifically colored Pie Chart labels and buttons from being overwritten */
div.stButton > button:first-child, div.stButton > button:first-child span {{
    color: white !important;
}}
.pie-text-risk {{ color: #f43f5e !important; }}
.pie-text-safe {{ color: #10b981 !important; }}

/* Result Slide Animations */
.smooth-reveal {{
    animation: smoothSlideUp 1s cubic-bezier(0.2, 0.8, 0.2, 1);
}}
@keyframes smoothSlideUp {{
    0% {{ opacity: 0; transform: translateY(40px) scale(0.98); }}
    100% {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
.smooth-spin {{
    animation: spinIn 1.5s cubic-bezier(0.2, 0.8, 0.2, 1);
}}
@keyframes spinIn {{
    0% {{ opacity: 0; transform: scale(0) rotate(-180deg); }}
    100% {{ opacity: 1; transform: scale(1) rotate(0deg); }}
}}

.healthy-anim svg {{
    animation: gentleHeartbeat 1.5s infinite alternate ease-in-out;
    display: block; margin: 20px auto; filter: drop-shadow(0 0 15px rgba(16, 185, 129, 0.6));
}}
@keyframes gentleHeartbeat {{
    0% {{ transform: scale(1); }}
    100% {{ transform: scale(1.15); }}
}}
.unhealthy-anim svg {{
    animation: flashWarning 0.8s infinite;
    display: block; margin: 20px auto; filter: drop-shadow(0 0 15px rgba(244, 63, 94, 0.6));
}}
@keyframes flashWarning {{
    0%, 100% {{ transform: scale(1); filter: drop-shadow(0 0 15px rgba(244, 63, 94, 0.6)); }}
    50% {{ transform: scale(1.1); filter: drop-shadow(0 0 30px rgba(244, 63, 94, 1)); }}
}}
div.stDownloadButton > button {{
    background: transparent !important;
    color: #3b82f6 !important;
    border: 2px solid #3b82f6 !important;
    border-radius: 50px !important;
    font-weight: bold !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}}
div.stDownloadButton > button:hover {{
    background: rgba(59, 130, 246, 0.1) !important;
}}
</style>
""", unsafe_allow_html=True)

# --- 12-Feature Model Training (Added BMI) ---
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 4000
    
    age = np.random.randint(29, 78, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    cp = np.random.randint(0, 4, n_samples)
    trestbps = np.random.randint(94, 201, n_samples)
    chol = np.random.randint(126, 565, n_samples)
    fbs = np.random.randint(0, 2, n_samples)
    restecg = np.random.randint(0, 3, n_samples)
    thalach = np.random.randint(71, 203, n_samples)
    exang = np.random.randint(0, 2, n_samples)
    oldpeak = np.random.uniform(0.0, 6.2, n_samples)
    slope = np.random.randint(0, 3, n_samples)
    
    height_m = np.random.uniform(1.5, 1.95, n_samples)
    weight_kg = np.random.uniform(50, 120, n_samples)
    bmi = weight_kg / (height_m ** 2)
    
    risk_score = 0
    risk_score += ((age - 29)/49) * 0.15
    risk_score += sex * 0.05
    risk_score += ((3 - cp)/3) * 0.15
    risk_score += ((trestbps - 94)/107) * 0.10
    risk_score += ((chol - 126)/439) * 0.10
    risk_score += fbs * 0.05
    risk_score += (restecg/2) * 0.05
    risk_score -= ((thalach - 71)/132) * 0.15
    risk_score += exang * 0.15
    risk_score += (oldpeak/6.2) * 0.10
    risk_score += (slope/2) * 0.05
    risk_score += ((bmi - 18.5)/21.5) * 0.15
    
    target = (risk_score > 0.45).astype(int) 
    
    df = pd.DataFrame({
        'Age': age, 'Sex': sex, 'ChestPain': cp, 'RestingBP': trestbps,
        'Cholesterol': chol, 'FastingBS': fbs, 'RestECG': restecg,
        'MaxHR': thalach, 'ExAngina': exang, 'Oldpeak': oldpeak, 'Slope': slope, 'BMI': bmi,
        'Target': target
    })
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=7, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    
    return model, accuracy

model, accuracy = train_model()

# ==========================================
# SINGLE PAGE: ALL IN ONE SCREEN
# ==========================================
st.markdown("<h1>⚕️ Cardiac Wellness System</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>A Comprehensive Machine Learning Health Checkup</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Demographics & Form")
    age = st.slider("1. Age (Years)", 20, 100, 35)
    sex_str = st.radio("2. Biological Sex", ["Female", "Male"], horizontal=True)
    sex = 1 if sex_str == "Male" else 0
    
    col_h, col_w = st.columns(2)
    with col_h: height = st.number_input("3. Height (cm)", 100, 250, 175)
    with col_w: weight = st.number_input("4. Weight (kg)", 30, 200, 70)
        
    calculated_bmi = weight / ((height/100)**2)
    st.info(f"⚖️ Calculated Body Mass Index (BMI): **{calculated_bmi:.1f}**")

with col2:
    st.markdown("### ❤️ Vital Indicators")
    trestbps = st.slider("5. Resting Blood Pressure (mmHg)", 90, 200, 115)
    chol = st.slider("6. Total Cholesterol (mg/dl)", 100, 600, 160)
    
    cp_opts = ["Severe Angina", "Moderate Angina", "Mild Pain", "None/Asymptomatic"]
    cp_str = st.selectbox("7. Chest Pain Level", options=cp_opts, index=3)
    cp = cp_opts.index(cp_str)
    
    fbs_str = st.radio("8. Fasting Blood Sugar > 120?", ["No", "Yes"], horizontal=True)
    fbs = 1 if fbs_str == "Yes" else 0

with col3:
    st.markdown("### 📈 Advanced Diagnostics")
    thalach = st.slider("9. Peak Heart Rate (BPM)", 60, 220, 170)
    
    exang_str = st.radio("10. Pain during Exercise?", ["No", "Yes"], horizontal=True)
    exang = 1 if exang_str == "Yes" else 0
    
    ecg_opts = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    restecg_str = st.selectbox("11. Resting ECG Results", options=ecg_opts)
    restecg = ecg_opts.index(restecg_str)
    
    oldpeak = st.slider("12. ST Depression (Oldpeak)", 0.0, 6.0, 0.0, step=0.1)
    slope_opts = ["Upsloping", "Flat", "Downsloping"]
    slope_str = st.selectbox("13. Peak ST Slope", options=slope_opts)
    slope = slope_opts.index(slope_str)

st.write("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1.5, 1])
with col_btn2:
    agree = st.checkbox("I acknowledge that this is an AI simulation and does NOT replace professional medical advice.", value=False)
    
    if st.button("RUN 12-POINT ANALYSIS ✨", disabled=not agree):
        import random
        # Pick a new gradient:
        current_idx = gradients.index(st.session_state.bg_gradient) if st.session_state.bg_gradient in gradients else 0
        st.session_state.bg_gradient = gradients[(current_idx + 1) % len(gradients)]
        
        st.session_state.submitted = True
        
        # Prepare Data Models
        st.session_state.bmi_val = calculated_bmi
        st.session_state.patient_data = pd.DataFrame([
            [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, calculated_bmi]
        ], columns=['Age', 'Sex', 'ChestPain', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestECG', 'MaxHR', 'ExAngina', 'Oldpeak', 'Slope', 'BMI'])
        
        st.session_state.chart_data = pd.DataFrame({
            "Patient Values": [trestbps, chol, thalach, calculated_bmi*4],
            "Optimal Baseline": [120, 180, 160, 22*4]
        }, index=["Blood Pressure", "Cholesterol", "Max Heart Rate", "Body Mass Index (Scaled)"])
        
        st.session_state.report_text = f"CARDIAC WELLNESS SYSTEM - DIAGNOSTIC REPORT\nDate generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nPatient Age: {age} | Sex: {sex_str}\nHeight: {height} cm | Weight: {weight} kg | BMI: {calculated_bmi:.1f}\nBlood Pressure: {trestbps} | Cholesterol: {chol} | Max Heart Rate: {thalach}\n\nDISCLAIMER: AI Confidence Accuracy on Validation data is {accuracy*100:.2f}%."
        
        st.rerun()

# ==========================================
# REPORT RENDERED ON THE SAME PAGE
# ==========================================
if st.session_state.submitted:
    with st.spinner('Compiling 12 variables against AI Neural Network...'):
        time.sleep(1)
        
    prediction = model.predict(st.session_state.patient_data)[0]
    probabilities = model.predict_proba(st.session_state.patient_data)[0]
    
    st.write("---")
    st.markdown("<h2 class='smooth-reveal' style='text-align: center; color: #1e3a8a !important;'>📋 Final Diagnostic Report</h2>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header smooth-reveal' style='margin-bottom: 10px;'>Computed using Deep Random Forest Analysis & Biometric Data</div>", unsafe_allow_html=True)
    
    res_spacer, res_col1, res_col2 = st.columns([0.5, 1.5, 1.2])
    
    with res_col1:
        if prediction == 1:
            st.markdown("""
            <div class="unhealthy-anim smooth-reveal">
              <svg height="180" width="180" viewBox="0 0 24 24">
                <path fill="#F43F5E" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                <path fill="none" stroke="#fff" stroke-width="2" d="M13 3 L10 10 L15 14 L11 21"/>
              </svg>
            </div>
            """, unsafe_allow_html=True)
            st.error(f"### ⚠️ Elevated Risk Profile Detected")
            st.write(f"**Likelihood of Cardiac Complications:** {probabilities[1]*100:.1f}%")
            
            st.markdown("""
            **Medical Advisory Plan:**
            - **Clinical Action Required:** Please schedule an appointment.
            - Address any abnormalities in your ECG or high resting blood pressure.
            - Limit dietary sodium intensely.
            """)
        else:
            st.markdown("""
            <div class="healthy-anim smooth-reveal">
              <svg height="180" width="180" viewBox="0 0 24 24">
                <path fill="#10B981" d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
                <path fill="none" stroke="#fff" stroke-width="3" d="M8 11.5 L11 14.5 L16 8"/>
              </svg>
            </div>
            """, unsafe_allow_html=True)
            st.success(f"### ✅ Cardiovascular System is Optimal")
            st.write(f"**Health Confidence Score:** {probabilities[0]*100:.1f}%")
            

            st.markdown("""
            **Sustenance Advisory Plan:**
            - **Incredible job!** Your metrics simulate a highly efficient cardiovascular system.
            - Continue your strict regimen of cardiovascular exercise.
            - Book an annual physician appointment to track these excellent metrics.
            """)
            
    with res_col2:
        st.markdown("<div class='smooth-reveal' style='text-align: center;'><h3>🥧 Probability Distribution</h3></div>", unsafe_allow_html=True)
        st.markdown("<p class='smooth-reveal' style='text-align: center; color: #475569;'>AI Calculated Cardiac Risk Ratio</p>", unsafe_allow_html=True)
        
        # Custom HTML/CSS Pie Chart
        risk_pct = int(probabilities[1] * 100)
        safe_pct = int(probabilities[0] * 100)
        st.markdown(f"""
        <div class="smooth-reveal" style="display: flex; justify-content: center; align-items: center; flex-direction: column; margin-top: 30px;">
            <div class="smooth-spin" style="width: 200px; height: 200px; border-radius: 50%; background: conic-gradient(#f43f5e 0% {risk_pct}%, #10b981 {risk_pct}% 100%); box-shadow: 0 6px 20px rgba(0,0,0,0.15);"></div>
            <div style="margin-top: 30px; font-weight: bold; text-align: center; font-size: 1.2rem;">
                <span class="pie-text-risk" style="margin-right: 20px;">■ Risk: {risk_pct}%</span>
                <span class="pie-text-safe">■ Safe: {safe_pct}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("---")
        # Action Buttons
        st.download_button(
            label="📄 Download Medical Summary",
            data=st.session_state.report_text,
            file_name="cardiac_wellness_report.txt",
            mime="text/plain"
        )
            
    st.caption(f"<div style='text-align: center; margin-top: 20px;'>*System Confidence Metric: {accuracy*100:.2f}% (Trained on 4,000 algorithmic profiles)*</div>", unsafe_allow_html=True)

st.markdown("<div style='text-align: center; margin-top: 50px; color: #64748b;'><b>Project Developed By:</b> Vinay, Om, and Biswojit</div>", unsafe_allow_html=True)
