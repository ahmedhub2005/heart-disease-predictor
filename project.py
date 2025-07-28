import pickle
import pandas as pd
import streamlit as st

# تحميل النموذج
data = pickle.load(open(r'D:\python for data science\notebooks\heart_disease.sav','rb'))

st.title('Heart Disease Prediction')
st.info('Easy App for Heart Prediction Disease')


# إدخالات المستخدم
st.sidebar.header("Feature Selection")
age = st.text_input('age')
sex = st.text_input('sex')
cp = st.text_input('cp')
trestbps = st.text_input('trestbps')
chol = st.text_input('chol')
fbs = st.text_input('fbs')
restecg = st.text_input('restecg')
thalach = st.text_input('thalach')
exang = st.text_input('exang')
oldpeak = st.text_input('oldpeak')
slope = st.text_input('slope')
ca = st.text_input('ca')
thal = st.text_input('thal')

# زر التأكيد
con = st.sidebar.button("Confirm")

if con:
    # التحقق من إدخال جميع القيم
    if '' in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]:
        st.sidebar.warning("Please fill in all the fields before confirming.")
    else:
        try:
            # تحويل البيانات إلى DataFrame بعد التأكد
            df = pd.DataFrame({
                'age': [age],
                'sex': [sex],
                "cp": [cp],
                "trestbps": [trestbps],
                "chol": [chol],
                "fbs": [fbs],
                "restecg": [restecg],
                "thalach": [thalach],
                "exang": [exang],
                "oldpeak": [oldpeak],
                'slope': [slope],
                "ca": [ca],
                "thal": [thal]
            }, index=[0]).astype(float)

            # التنبؤ بالنتيجة
            result = data.predict(df)
            prediction_label = "Heart Disease Detected" if result[0] == 1 else "No Heart Disease"
            st.sidebar.success(f"Prediction: {prediction_label}")


        except Exception as e:
            st.sidebar.error(f"An error occurred during prediction: {e}")



    