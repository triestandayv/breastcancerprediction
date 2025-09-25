import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data.csv")
df = df.drop(columns=["id", "Unnamed: 32"])

encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])

X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.sidebar.title("🔬 Breast Cancer KNN App")
mode = st.sidebar.radio("Choose Mode:", ["Model Evaluation", "Patient Prediction"])

if mode == "Model Evaluation":
    st.title("📊 Model Evaluation")

    k = st.sidebar.slider("Select number of neighbors (k)", 1, 20, 5)

    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)

    st.subheader("Model Performance")
    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

    st.write("**Confusion Matrix:**")
    st.write(confusion_matrix(y_test, y_pred))

    st.write("**Classification Report:**")
    st.text(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

elif mode == "Patient Prediction":
    st.title("🔮 Patient Prediction")

    st.write("**Manually enter patient measurements**")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    st.subheader("Manual Input")
     tabs = st.tabs(["Mean Values", "Standard Error", "Worst Values"])

    inputs = {}

    with tabs[0]:
        st.write("👉 Enter **Mean Features**")
        for col in [c for c in X.columns if "mean" in c]:
            inputs[col] = st.number_input(f"{col}", min_value=0.0, format="%.4f")

    with tabs[1]:
        st.write("👉 Enter **Standard Error Features**")
        for col in [c for c in X.columns if "se" in c]:
            inputs[col] = st.number_input(f"{col}", min_value=0.0, format="%.4f")

    with tabs[2]:
        st.write("👉 Enter **Worst Features**")
        for col in [c for c in X.columns if "worst" in c]:
            inputs[col] = st.number_input(f"{col}", min_value=0.0, format="%.4f")

    if st.button("Predict (Manual Input)"):
        new_data = pd.DataFrame([inputs])
        new_data_scaled = scaler.transform(new_data.reindex(columns=X.columns, fill_value=0))
        prediction = knn.predict(new_data_scaled)
        result = "Malignant" if prediction[0] == 1 else "Benign"
        st.success(f"Prediction: **{result}**")