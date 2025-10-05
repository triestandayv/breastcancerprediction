import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Breast Cancer Prediction using Logistic Regression Algorithm")
st.markdown("""
**Data Mining Final Project**  
**Developed by:** Triestan Dave Talamán
""")
st.write("Adjust the sliders for each feature below and click **Predict** to see if the tumor is malignant or benign.")

feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

feature_ranges = {
    'radius_mean': (6.0, 30.0),
    'texture_mean': (9.0, 40.0),
    'perimeter_mean': (40.0, 200.0),
    'area_mean': (140.0, 2500.0),
    'smoothness_mean': (0.05, 0.2),
    'compactness_mean': (0.02, 0.35),
    'concavity_mean': (0.0, 0.45),
    'concave points_mean': (0.0, 0.2),
    'symmetry_mean': (0.1, 0.3),
    'fractal_dimension_mean': (0.04, 0.1),
    'radius_se': (0.1, 3.0),
    'texture_se': (0.3, 5.0),
    'perimeter_se': (1.0, 25.0),
    'area_se': (5.0, 550.0),
    'smoothness_se': (0.001, 0.03),
    'compactness_se': (0.002, 0.1),
    'concavity_se': (0.0, 0.3),
    'concave points_se': (0.0, 0.05),
    'symmetry_se': (0.005, 0.08),
    'fractal_dimension_se': (0.001, 0.03),
    'radius_worst': (7.0, 40.0),
    'texture_worst': (10.0, 50.0),
    'perimeter_worst': (50.0, 300.0),
    'area_worst': (200.0, 4000.0),
    'smoothness_worst': (0.07, 0.25),
    'compactness_worst': (0.02, 1.0),
    'concavity_worst': (0.0, 1.3),
    'concave points_worst': (0.0, 0.3),
    'symmetry_worst': (0.1, 0.5),
    'fractal_dimension_worst': (0.05, 0.3)
}

st.subheader("Mean Features")
cols = st.columns(2)
inputs = []
for i, feature in enumerate(feature_names[:10]):
    with cols[i % 2]:
        val = st.slider(feature, *feature_ranges[feature])
        inputs.append(val)

st.subheader("Standard Error Features")
cols = st.columns(2)
for i, feature in enumerate(feature_names[10:20]):
    with cols[i % 2]:
        val = st.slider(feature, *feature_ranges[feature])
        inputs.append(val)

st.subheader("Worst Features")
cols = st.columns(2)
for i, feature in enumerate(feature_names[20:]):
    with cols[i % 2]:
        val = st.slider(feature, *feature_ranges[feature])
        inputs.append(val)

if st.button("Predict"):
    input_df = pd.DataFrame([inputs], columns=feature_names)
    scaled = scaler.transform(input_df)
    prediction = model.predict(scaled)[0]

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.error("Malignant (Cancerous Tumor)")
    else:
        st.success("Benign (Non-Cancerous Tumor)")
