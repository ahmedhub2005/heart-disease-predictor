import pickle
import pandas as pd
import streamlit as st

data = pickle.load(open(r'heart_disease_V2.sav','rb'))

st.title('Easy Heart Disease Prediction App')
st.info('A simple and intelligent tool to help predict the risk of heart disease using machine learning.')


st.sidebar.header("Feature Selection")
age = st.text_input('Age')
sex = st.text_input('Sex')
cp = st.text_input('Cp')
trestbps = st.text_input('Trestbps')
chol = st.text_input('Chol')
fbs = st.text_input('Fbs')
restecg = st.text_input('Restecg')
thalach = st.text_input('Thalach')
exang = st.text_input('Exang')
oldpeak = st.text_input('Oldpeak')
slope = st.text_input('Slope')
ca = st.text_input('Ca')
thal = st.text_input('Thal')


con = st.sidebar.button("Confirm")

if con:
    
    if '' in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]:
        st.sidebar.warning("Please fill in all the fields before confirming.")
    else:
        try:
        
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

     
            result = data.predict(df)
            prediction_label = "Heart Disease Detected" if result[0] == 1 else "No Heart Disease"
            st.sidebar.success(f"Prediction: {prediction_label}")


        except Exception as e:
            st.sidebar.error(f"An error occurred during prediction: {e}")




    

