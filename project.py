# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import uuid
from fpdf import FPDF
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

# Title + subtitle
st.markdown('<div class="app-title">🩺 Heart Disease Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Professional clinical analysis, evaluation & recommendations</div>', unsafe_allow_html=True)

# Example card with recommendations
st.markdown("""
    <div class="card">
        <h4 style="color:#1a3d7c; margin-bottom:12px;">Doctor Recommendations</h4>
        <div class="recommendation">✔ Maintain a balanced diet and exercise regularly.</div>
        <div class="recommendation">✔ Monitor your blood pressure every week.</div>
        <div class="recommendation">✔ Avoid smoking and alcohol consumption.</div>
    </div>
""", unsafe_allow_html=True)

# Interactive Button with JavaScript
import streamlit.components.v1 as components
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


# --- Load model + scaler if exist (tries both names) ---
MODEL_FILES = ["heart_disease_model.pkl", "scaler.pkl", "feature_names.pkl"]
model = None
for f in MODEL_FILES:
    if os.path.exists(f):
        with open(f, "rb") as fh:
            model = pickle.load(fh)
        break
if model is None:
    st.error("Model file not found. Run training script first (train_save_model.py).")
    st.stop()

scaler = None
if os.path.exists("scaler.pkl"):
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

feature_names = None
if os.path.exists("feature_names.pkl"):
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

# --- App UI ---
st.set_page_config(layout="wide")
st.title("🩺 Easy Heart Disease Prediction App (Advanced)")

# Sidebar Inputs
st.sidebar.header("Enter Patient Data")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=45)
sex = st.sidebar.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0,1,2,3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", 60, 260, 120)
chol = st.sidebar.number_input("Serum Cholesterol (chol)", 50, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0,1])
restecg = st.sidebar.selectbox("Resting ECG (restecg)", [0,1,2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved (thalach)", 40, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0,1])
oldpeak = st.sidebar.number_input("Oldpeak", 0.0, 8.0, 1.0, step=0.1)
slope = st.sidebar.selectbox("Slope", [0,1,2])
ca = st.sidebar.selectbox("Number of Major Vessels (ca)", [0,1,2,3,4])
thal = st.sidebar.selectbox("Thalassemia (thal)", [0,1,2,3])

# indicator functions
def age_group(age):
    if age < 40: return 0
    elif age < 60: return 1
    else: return 2

def bp_category(bp):
    if bp < 120: return 0
    elif bp < 130: return 1
    elif bp < 140: return 2
    else: return 3

def chol_risk(chol):
    if chol < 200: return 0
    elif chol < 240: return 1
    else: return 2

def oldpeak_risk(op):
    if op < 1: return 0
    elif op < 2: return 1
    else: return 2

# compute extras
ag = age_group(age)
bpc = bp_category(trestbps)
cr = chol_risk(chol)
HRR = (220 - age) - thalach
opr = oldpeak_risk(oldpeak)
rscore = (age/100) + (trestbps/200) + (chol/300) + (oldpeak/5)

# Prepare DataFrame in same order as training if available
input_dict = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
    "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
    "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
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

with tab1:
    st.subheader("Prediction Result")
    if st.button("Run Prediction"):
        pred = int(model.predict(X_for_model)[0])
        prob = float(model.predict_proba(X_for_model)[0][1]) if hasattr(model, "predict_proba") else None

        # Save to session_state for other tabs
        st.session_state["pred"] = pred
        st.session_state["prob"] = prob
        st.session_state["input_dict"] = input_dict

        # result
        if pred == 1:
            st.error(" Prediction: HEART DISEASE DETECTED")
        else:
            st.success("Prediction: No Heart Disease Detected")
        if prob is not None:
            st.write(f"Predicted probability (positive): {prob:.2%}")

        # Recommendations
        recs = []
        if bpc >= 2:
            recs.append("High BP: recommend antihypertensive evaluation, lifestyle changes.")
        elif bpc == 1:
            recs.append("Elevated BP: lifestyle modification, monitor BP regularly.")
        else:
            recs.append("BP within normal range.")

        if cr == 2:
            recs.append("High cholesterol: consider statin therapy, diet modification.")
        elif cr == 1:
            recs.append("Borderline cholesterol: recheck in 3 months.")
        else:
            recs.append("Cholesterol acceptable.")

        if HRR < 50:
            recs.append("Low HRR: consider exercise ECG or referral.")
        else:
            recs.append("HRR within acceptable limits.")

        if opr == 2:
            recs.append("Significant ST depression — urgent cardiology evaluation.")
        elif opr == 1:
            recs.append("Mild ST changes — consider noninvasive testing.")
        else:
            recs.append("No significant ST depression detected.")

        if pred == 1:
            recs.append("Model predicts heart disease: arrange cardiology follow-up.")
        else:
            recs.append("No model-predicted disease: continue preventive measures.")

        st.session_state["recs"] = recs
        for r in recs:
            st.write("- " + r)

with tab2:
    st.markdown("### 🧪 Clinical Indicators & Visualizations")

    if "pred" not in st.session_state:
        st.info("⚠️ Run prediction first in Tab1.")
    else:
        # --- Patient Info Cards ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="card" style="color:#f5f5f5; background:#1c1c1c; padding:10px; border-radius:10px;">
                    <h4>👤 Patient Info</h4>
                    <p><b>Age:</b> {age}</p>
                    <p><b>Sex:</b> {"Male" if sex==1 else "Female"}</p>
                    <p><b>Resting BP:</b> {trestbps} mmHg</p>
                    <p><b>Cholesterol:</b> {chol} mg/dl</p>
                    <p><b>Fasting BS > 120:</b> {"Yes" if fbs==1 else "No"}</p>
                    <p><b>Max HR:</b> {thalach}</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
                <div class="card" style="color:#f5f5f5; background:#1c1c1c; padding:10px; border-radius:10px;">
                    <h4>📊 Risk Analysis</h4>
                    <p><b>Age Group:</b> {["Young","Middle","Old"][ag]}</p>
                    <p><b>Heart Rate Reserve:</b> {HRR}</p>
                    <p><b>Oldpeak Risk:</b> {["Low","Moderate","High"][opr]}</p>
                    <p><b>Composite Risk Score:</b> {rscore:.2f}</p>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📈 Visual Diagnostics")
        colA, colB = st.columns([2,1])

        # --- High-Resolution ECG ---
        with colA:
            t = np.linspace(0, 1, 2000)
            ecg = (0.5*np.sin(2*np.pi*5*t) + 
                   0.3*np.sin(2*np.pi*15*t) + 
                   0.2*np.sin(2*np.pi*25*t) + 
                   0.05*np.random.randn(len(t)))
            fig, ax = plt.subplots(figsize=(12,3))
            ax.plot(t, ecg, color="#00ffcc", linewidth=1.5)
            ax.set_facecolor("#000000")
            ax.set_yticks([])
            ax.set_title("🫀 Simulated ECG (High-Res)", fontsize=12, color="#00ffcc")
            st.pyplot(fig)
            st.session_state["fig_ecg"] = fig

        # --- Gauge for Max HR ---
        with colB:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=thalach,
                delta={'reference': 150, 'increasing': {'color': "#ffdd00"}},
                gauge={
                    'axis': {'range': [40,220], 'tickcolor': "white"},
                    'bar': {'color': "#ff3366"},
                    'bgcolor': "#1c1c1c",
                    'steps': [
                        {'range': [40,100], 'color': "#333333"},
                        {'range': [100,160], 'color': "#0055aa"},
                        {'range': [160,220], 'color': "#aa0033"}
                    ],
                    'threshold': {
                        'line': {'color': "yellow", 'width': 4},
                        'thickness': 0.75,
                        'value': 160
                    }
                },
                number={'font': {'color': "white"}},
                title={'text': "Max HR (bpm)", 'font': {'size': 16, 'color': "white"}}
            ))
            fig_g.update_layout(paper_bgcolor="#000000", font={'color':"white"})
            st.plotly_chart(fig_g, use_container_width=True)
            st.session_state["fig_gauge"] = fig_g

        # --- Radar Chart for Risk Profile ---
        categories = ["Age","BP","Chol","Oldpeak","RiskScore"]
        vals = [ag/2, bpc/3, cr/2, opr/2, min(rscore/5, 1.0)]
        vals = vals + [vals[0]]
        fig_radar = go.Figure(go.Scatterpolar(
            r=vals,
            theta=categories + [categories[0]],
            fill='toself',
            line_color="cyan",
            opacity=0.7
        ))
        fig_radar.update_layout(
            title="🌀 Risk Profile Radar",
            polar=dict(
                bgcolor="#000000",
                radialaxis=dict(range=[0,1], showticklabels=True, tickfont=dict(color="white"), gridcolor="#444")
            ),
            paper_bgcolor="#000000",
            font=dict(color="white"),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.session_state["fig_radar"] = fig_radar


with tab3:
    st.subheader("Export Professional Clinical Report (PDF)")

    if "pred" not in st.session_state:
        st.info("Run prediction first in Tab1.")
    else:
        if st.button("Export PDF Report"):
            from datetime import datetime
            import uuid
            import os
            from fpdf import FPDF
            import plotly.io as pio

            # --- Helper function to remove non-ASCII chars ---
            def safe_ascii(text):
                return ''.join([c if ord(c) < 128 else ' ' for c in str(text)])

            # --- Temp files for images ---
            uid = str(uuid.uuid4())[:8]
            ecg_path = f"temp_ecg_{uid}.png"
            gauge_path = f"temp_gauge_{uid}.png"

            st.session_state["fig_ecg"].savefig(ecg_path, bbox_inches="tight")
            try:
                pio.write_image(st.session_state["fig_gauge"], gauge_path, format="png", scale=2)
                gauge_available = True
            except Exception:
                gauge_available = False

            # --- Custom PDF class with footer ---
            class PDF(FPDF):
                def footer(self):
                    self.set_y(-15)
                    self.set_font('Helvetica', 'I', 8)
                    self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

            pdf = PDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            # === 1. Cover Page ===
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 24)
            pdf.cell(0, 20, "Heart Disease Risk Assessment Report", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Helvetica", "", 14)
            patient_name = st.session_state['input_dict'].get('PatientName','N/A')
            pdf.cell(0, 10, f"Patient Name: {safe_ascii(patient_name)}", ln=True, align="C")
            pdf.cell(0, 10, f"Date of Report: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align="C")
            pdf.ln(10)
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Generated By: Heart Disease Predictor System", ln=True, align="C")
            pdf.ln(20)

            # === 2. Executive Summary ===
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 18)
            pdf.cell(0, 10, "Executive Summary", ln=True)
            pdf.ln(5)
            pdf.set_font("Helvetica", "", 12)
            pdf.multi_cell(0, 8, "This report provides an assessment of your risk of developing heart disease based on clinical indicators and diagnostic data.")
            pdf.ln(5)
            pred = st.session_state["pred"]
            prob = st.session_state.get("prob")
            risk_level = "Low"
            if prob is not None:
                if prob < 0.3:
                    risk_level = "Low"
                elif prob < 0.7:
                    risk_level = "Moderate"
                else:
                    risk_level = "High"
            pdf.multi_cell(0, 8, f"Overall Risk: {risk_level}")

            # === 3. Patient Information Table ===
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Patient Information", ln=True)
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(60, 8, "Attribute", border=1)
            pdf.cell(0, 8, "Value", border=1, ln=True)
            pdf.set_font("Helvetica", "", 12)
            for k, v in st.session_state["input_dict"].items():
                pdf.cell(60, 8, safe_ascii(str(k)), border=1)
                pdf.cell(0, 8, safe_ascii(str(v)), border=1, ln=True)

            # === 4. Clinical Indicators ===
            if "tests_dict" in st.session_state:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, "Clinical Indicators", ln=True)
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(50, 8, "Indicator", border=1)
                pdf.cell(35, 8, "Value", border=1)
                pdf.cell(35, 8, "Normal Range", border=1)
                pdf.cell(0, 8, "Notes", border=1, ln=True)
                pdf.set_font("Helvetica", "", 12)
                for test, result in st.session_state["tests_dict"].items():
                    pdf.cell(50, 8, safe_ascii(test), border=1)
                    pdf.cell(35, 8, safe_ascii(str(result['value'])), border=1)
                    pdf.cell(35, 8, safe_ascii(str(result.get('normal','N/A'))), border=1)
                    pdf.cell(0, 8, safe_ascii(result.get('notes','')), border=1, ln=True)

            # === 5. Risk Analysis ===
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Risk Analysis", ln=True)
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 12)
            pdf.multi_cell(0, 8, f"Predicted Risk: {prob:.2%}" if prob is not None else "Predicted Risk: N/A")
            pdf.multi_cell(0, 8, f"Interpretation: {risk_level}")
            if gauge_available:
                pdf.image(gauge_path, x=pdf.w/2 - 40, w=80)
                pdf.multi_cell(0, 8, "Risk Gauge", align="C")

            # === 6. Recommendations ===
            if "recs" in st.session_state:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, "Recommendations", ln=True)
                pdf.ln(2)
                pdf.set_font("Helvetica", "", 12)
                for r in st.session_state["recs"]:
                    pdf.multi_cell(0, 8, f"- {safe_ascii(r)}")

            # === 7. Graphical Summary ===
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Graphical Summary", ln=True)
            pdf.ln(2)
            pdf.set_font("Helvetica", "", 12)
            pdf.image(ecg_path, x=pdf.w/2 - 90, w=180)
            pdf.multi_cell(0, 8, "Electrocardiogram (ECG)", align="C")

            # === 8. Conclusion & Disclaimer ===
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Conclusion & Notes", ln=True)
            pdf.ln(3)
            pdf.set_font("Helvetica", "", 12)
            summary_lines = [
                f"Patient ID: {st.session_state['input_dict'].get('PatientID','N/A')}",
                f"Age: {st.session_state['input_dict'].get('age','N/A')}",
                f"Prediction: {'HEART DISEASE DETECTED' if pred==1 else 'No Heart Disease'}",
            ]
            if prob is not None:
                summary_lines.append(f"Predicted Probability: {prob:.2%}")
            if "recs" in st.session_state:
                summary_lines.append("")
                summary_lines.append("Key Recommendations:")
                for r in st.session_state["recs"]:
                    summary_lines.append(f"- {safe_ascii(r)}")
            summary_lines.append("")
            summary_lines.append("Disclaimer: This report is generated based on predictive analytics and should not replace medical consultation.")
            pdf.multi_cell(0, 8, "\n".join(summary_lines))

            # --- Save PDF ---
            out_file = f"heart_report_{uid}.pdf"
            pdf.output(out_file)

            # Cleanup temp images
            for p in [ecg_path, gauge_path]:
                if os.path.exists(p):
                    os.remove(p)

            # Download button
            with open(out_file, "rb") as f:
                st.download_button("Download PDF Report", f, file_name=out_file, mime="application/pdf")

            st.success("Professional clinical report generated successfully!")












    




















