import pickle

import joblib
import pandas as pd
import streamlit as st

# Loading the model and scaler
model = joblib.load("rf_model.joblib")
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Page header
st.header("Heart Disease Prediction Application")

# Options
fasting_bs_opts = {"Yes": 1, "No": 0}
sex_opts = {"Male": "M", "Female": "F"}
chest_pain_type_opts = {
    "Typical Angina": "TA",
    "Atypical Angina": "ATA",
    "Non-Anginal Pain": "NAP",
    "Asymptomatic": "ASY",
}
resting_ecg_opts = {
    "Normal": "Normal",
    "Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)": "ST",
    "Showing probable or definite left ventricular hypertrophy by Estes' criteria": "LVH",
}
exercise_angina_opts = {"Yes": "Y", "No": "N"}
st_slope_opts = {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}

# User inputs for prediction
col1, col2, col3 = st.columns([0.4, 0.2, 0.4])
with col1:
    age = st.slider("Age of the patient [years]", min_value=20, max_value=100)
    resting_bp = st.slider(
        "Resting Blood Pressure [mm Hg]", min_value=70, max_value=200
    )
    cholestrol = st.slider("Serum Cholesterol [mm/dl]", min_value=0, max_value=450)
    max_hr = st.slider("Maximum Heart Rate achieved", min_value=60, max_value=200)
    old_peak = st.slider(
        "ST [Numeric value measured in depression]", min_value=-2.0, max_value=6.0
    )

with col3:
    fasting_bs = st.selectbox(
        "Is Fasting Blood Sugar > 120 mg/dl ?", fasting_bs_opts.keys()
    )
    sex = st.selectbox("Sex of the patient", sex_opts.keys())
    chest_pain_type = st.selectbox("Chest pain type", chest_pain_type_opts.keys())
    resting_ecg = st.selectbox(
        "Resting Electrocardiogram results", resting_ecg_opts.keys()
    )
    exercise_angina = st.selectbox(
        "Exercise Induced Angina", exercise_angina_opts.keys()
    )
    st_slope = st.selectbox(
        "The slope of the peak exercise ST segment", st_slope_opts.keys()
    )

# Input dataframe
data = pd.DataFrame(
    {
        "Age": [age],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholestrol],
        "FastingBS": [fasting_bs_opts.get(fasting_bs)],
        "MaxHR": [max_hr],
        "Oldpeak": [old_peak],
        "Sex": [sex_opts.get(sex)],
        "ChestPainType": [chest_pain_type_opts.get(chest_pain_type)],
        "RestingECG": [resting_ecg_opts.get(resting_ecg)],
        "ExerciseAngina": [exercise_angina_opts.get(exercise_angina)],
        "ST_Slope": [st_slope_opts.get(st_slope)],
    }
)

if st.button("Predict", key="user_submit"):
    categorical_cols = data.select_dtypes(include=["object"]).columns
    data = pd.get_dummies(data, columns=categorical_cols)

    training_cols = [
        "Age",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "MaxHR",
        "Oldpeak",
        "Sex_F",
        "Sex_M",
        "ChestPainType_ASY",
        "ChestPainType_ATA",
        "ChestPainType_NAP",
        "ChestPainType_TA",
        "RestingECG_LVH",
        "RestingECG_Normal",
        "RestingECG_ST",
        "ExerciseAngina_N",
        "ExerciseAngina_Y",
        "ST_Slope_Down",
        "ST_Slope_Flat",
        "ST_Slope_Up",
    ]

    for col in training_cols:
        if col not in data.columns:
            data[col] = 0

    data = data[training_cols]
    scaled_data = scaler.transform(data)

    prediction = model.predict(scaled_data)[0]

    if prediction:
        st.write("Heart Disease is detected.")
    else:
        st.write("Heart Disease is not detected.")