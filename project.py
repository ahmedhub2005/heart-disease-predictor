# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF
import streamlit.components.v1 as components

# --- Streamlit Config ---
st.set_page_config(layout="wide", page_title="Heart Disease Predictor")

# --- Custom UI ---
st.markdown("""
    <style>
    body {
        background-color: #f5f8ff;
    }
    .app-title {
        font-size: 36px;
        font-weight: 700;
        color: #1a3d7c;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-title {
        font-size: 18px;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
    }
    .card {
        background: #fff;
        padding: 20px 25px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 15px 0;
        transition: transform 0.2s ease;
    }
    .card:hover {
        transform: translateY(-3px);
    }
    .recommendation {
        font-size: 15px;
        color: #333;
        padding: 10px 14px;
        border-left: 5px solid #0073e6;
        margin-bottom: 8px;
        background: #eef6ff;
        border-radius: 6px;
        line-height: 1.5;
    }
    .btn {
        background: linear-gradient(135deg, #0073e6, #00c6ff);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 10px;
        cursor: pointer;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .btn:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #005bb5, #0099cc);
    }
    </style>
""", unsafe_allow_html=True)

# --- Title + Subtitle ---
st.markdown('<div class="app-title">🩺 Heart Disease Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Professional clinical analysis, evaluation & recommendations</div>', unsafe_allow_html=True)

# --- Example Card ---
st.markdown("""
    <div class="card">
        <h4 style="color:#1a3d7c; margin-bottom:12px;">Doctor Recommendations</h4>
        <div class="recommendation">✔ Maintain a balanced diet and exercise regularly.</div>
        <div class="recommendation">✔ Monitor your blood pressure every week.</div>
        <div class="recommendation">✔ Avoid smoking and alcohol consumption.</div>
    </div>
""", unsafe_allow_html=True)

# --- Interactive Button ---
components.html("""
    <button class="btn" id="notifyBtn">💡 Show Health Tip</button>
    <script>
        const tips = [
            "Drink at least 2 liters of water daily 💧",
            "30 minutes of walking can improve heart health 🚶",
            "Reduce salt intake to lower blood pressure 🧂",
            "Regular sleep (7-8 hrs) keeps the heart healthy 😴"
        ];
        document.getElementById("notifyBtn").onclick = function() {
            const randomTip = tips[Math.floor(Math.random()*tips.length)];
            alert("Health Tip: " + randomTip);
        }
    </script>
""", height=120)

# --- Load model + scaler ---
MODEL_FILES = ["heart_disease_model.pkl", "heart_disease_V2.sav", "heart_disease_model.sav"]
model = None
for f in MODEL_FILES:
    if os.path.exists(f):
        with open(f, "rb") as fh:
            model = pickle.load(fh)
        break

if model is None:
    st.error("❌ Model file not found. Please run training script first.")
    st.stop()

scaler, feature_names = None, None
if os.path.exists("scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
if os.path.exists("feature_names.pkl"):
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

# --- Sidebar Inputs ---
st.sidebar.header("🧍 Patient Data Input")
age = st.sidebar.number_input("Age", 1, 120, 45)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.sidebar.number_input("Resting BP (trestbps)", 60, 260, 120)
chol = st.sidebar.number_input("Cholesterol (chol)", 50, 600, 200)
fbs = st.sidebar.selectbox("Fasting BS > 120 (fbs)", [0,1])
restecg = st.sidebar.selectbox("Resting ECG", [0,1,2])
thalach = st.sidebar.number_input("Max HR (thalach)", 40, 220, 150)
exang = st.sidebar.selectbox("Exercise Angina (exang)", [0,1])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 8.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope", [0,1,2])
ca = st.sidebar.selectbox("Major Vessels (ca)", [0,1,2,3,4])
thal = st.sidebar.selectbox("Thalassemia (thal)", [0,1,2,3])

# --- Indicators ---
def age_group(age): return 0 if age<40 else 1 if age<60 else 2
def bp_category(bp): return 0 if bp<120 else 1 if bp<130 else 2 if bp<140 else 3
def chol_risk(chol): return 0 if chol<200 else 1 if chol<240 else 2
def oldpeak_risk(op): return 0 if op<1 else 1 if op<2 else 2

ag = age_group(age)
bpc = bp_category(trestbps)
cr = chol_risk(chol)
HRR = (220 - age) - thalach
opr = oldpeak_risk(oldpeak)
rscore = (age/100) + (trestbps/200) + (chol/300) + (oldpeak/5)

# --- DataFrame ---
input_dict = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    "age_group": ag, "bp_category": bpc, "chol_risk": cr, "HRR": HRR,
    "oldpeak_risk": opr, "risk_score": rscore
}
if feature_names:
    row = [input_dict.get(name, 0) for name in feature_names]
    X = pd.DataFrame([row], columns=feature_names)
else:
    X = pd.DataFrame([input_dict])
X_for_model = scaler.transform(X) if scaler is not None else X.values

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Indicators", "📄 Report"])

# --- Tab1: Prediction ---
with tab1:
    st.subheader("Prediction Result")
    if st.button("Run Prediction"):
        pred = int(model.predict(X_for_model)[0])
        prob = float(model.predict_proba(X_for_model)[0][1]) if hasattr(model, "predict_proba") else None
        st.session_state.update({"pred": pred, "prob": prob, "input_dict": input_dict})

        if pred == 1:
            st.error("❤️ Prediction: HEART DISEASE DETECTED")
        else:
            st.success("✅ Prediction: No Heart Disease Detected")
        if prob is not None:
            st.info(f"Predicted probability (positive): {prob:.2%}")

        recs = []
        recs.append("High BP: consider lifestyle changes or medication." if bpc>=2 else "BP acceptable.")
        recs.append("High cholesterol: diet/medication advised." if cr==2 else "Cholesterol acceptable.")
        recs.append("Low HRR: exercise ECG recommended." if HRR<50 else "HRR within range.")
        recs.append("Significant ST depression — cardiology evaluation." if opr==2 else "No significant ST depression.")
        recs.append("Follow-up needed." if pred==1 else "Continue preventive measures.")
        st.session_state["recs"] = recs

        for r in recs:
            st.write("- " + r)

# --- Tab2: Indicators & Visualizations ---
with tab2:
    st.markdown("### 🧪 Clinical Indicators & Visualizations")
    if "pred" not in st.session_state:
        st.info("⚠️ Run prediction first in Tab1.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="card" style="color:#f5f5f5; background:#1c1c1c;">
                <h4>👤 Patient Info</h4>
                <p><b>Age:</b> {age}</p>
                <p><b>Sex:</b> {"Male" if sex==1 else "Female"}</p>
                <p><b>Resting BP:</b> {trestbps} mmHg</p>
                <p><b>Cholesterol:</b> {chol} mg/dl</p>
                <p><b>Max HR:</b> {thalach}</p>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="card" style="color:#f5f5f5; background:#1c1c1c;">
                <h4>📊 Risk Analysis</h4>
                <p><b>Age Group:</b> {["Young","Middle","Old"][ag]}</p>
                <p><b>Heart Rate Reserve:</b> {HRR}</p>
                <p><b>Oldpeak Risk:</b> {["Low","Moderate","High"][opr]}</p>
                <p><b>Composite Risk Score:</b> {rscore:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Visual Diagnostics")
        colA, colB = st.columns([2,1])
        with colA:
            t = np.linspace(0, 1, 500)
            ecg = np.sin(5*2*np.pi*t) + 0.4*np.sin(15*2*np.pi*t) + 0.05*np.random.randn(len(t))
            fig, ax = plt.subplots(figsize=(9,3))
            ax.plot(t, ecg, color="#00ffcc", linewidth=1.5)
            ax.set_facecolor("#000000")
            ax.set_yticks([])
            ax.set_title("🫀 Simulated ECG", fontsize=12, color="#00ffcc")
            st.pyplot(fig)
            st.session_state["fig_ecg"] = fig

        with colB:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=thalach,
                gauge={
                    'axis': {'range': [40,220], 'tickcolor': "white"},
                    'bar': {'color': "#ff3366"}, 'bgcolor': "#1c1c1c",
                    'steps': [
                        {'range': [40,100], 'color': "#333333"},
                        {'range': [100,160], 'color': "#0055aa"},
                        {'range': [160,220], 'color': "#aa0033"} ]},
                number={'font': {'color': "white"}},
                title={'text': "Max HR (bpm)", 'font': {'size': 16, 'color': "white"}}
            ))
            fig_g.update_layout(paper_bgcolor="#000000", font={'color':"white"})
            st.plotly_chart(fig_g, use_container_width=True)
            st.session_state["fig_gauge"] = fig_g

        categories = ["Age","BP","Chol","Oldpeak","RiskScore"]
        vals = [ag/2, bpc/3, cr/2, opr/2, min(rscore/5, 1.0)] + [ag/2]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals, theta=categories+ [categories[0]], fill='toself', line_color="#00ccff"
        ))
        fig_radar.update_layout(
            title="🌀 Risk Profile Radar",
            polar=dict(bgcolor="#000000", radialaxis=dict(range=[0,1], tickfont=dict(color="white"), gridcolor="#444")),
            paper_bgcolor="#000000", font=dict(color="white"), showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.session_state["fig_radar"] = fig_radar

# --- Tab3: Report (PDF) ---
with tab3:
    st.subheader("📄 Export Professional Clinical Report (PDF)")
    if "pred" not in st.session_state:
        st.info("⚠️ Run prediction first in Tab1.")
    else:
        if st.button("Export PDF Report"):
            from datetime import datetime
            import uuid
            uid = str(uuid.uuid4())[:8]
            ecg_path, radar_path, gauge_path = f"ecg_{uid}.png", f"radar_{uid}.png", f"gauge_{uid}.png"

            st.session_state["fig_ecg"].savefig(ecg_path, bbox_inches="tight")
            with open(radar_path, "wb") as f: f.write(st.session_state["fig_radar"].to_image(format="png", scale=2))
            st.session_state["fig_gauge"].write_image(gauge_path, scale=2)

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 20)
            pdf.cell(0, 10, "Heart Disease Clinical Report", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Date: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align="C")

            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Prediction & Evaluation", ln=True)
            pdf.set_font("Arial", size=12)
            pred = st.session_state["pred"]
            prob = st.session_state["prob"]
            pdf.cell(0, 10, f"Prediction: {'HEART DISEASE DETECTED' if pred==1 else 'No Heart Disease'}", ln=True)
            if prob is not None:
                pdf.cell(0, 10, f"Probability: {prob:.2%}", ln=True)

            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Doctor Recommendations", ln=True)
            pdf.set_font("Arial", size=12)
            for r in st.session_state["recs"]:
                pdf.multi_cell(0, 8, "- " + r)

            pdf.add_page()
            pdf.image(ecg_path, x=15, w=180)
            pdf.image(radar_path, x=15, w=180)
            pdf.image(gauge_path, x=60, w=80)

            out_file = f"heart_report_{uid}.pdf"
            pdf.output(out_file)

            for p in [ecg_path, radar_path, gauge_path]:
                if os.path.exists(p): os.remove(p)

            with open(out_file, "rb") as f:
                st.download_button("⬇️ Download PDF Report", f, file_name=out_file, mime="application/pdf")
            st.success("✅ Professional report generated successfully!")






    




