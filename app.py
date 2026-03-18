
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load Models
# -----------------------------
clinical_model = joblib.load("clinical_rf_model.pkl")
ecg_model = load_model("ecg_cnn_model.h5")

st.title("Multimodal Heart Disease Prediction System")

# -----------------------------
# Clinical Data Input
# -----------------------------
st.header("Enter Clinical Patient Data")

age = st.number_input("Age", min_value=0.0, max_value=120.0, value=55.0)
cholesterol = st.number_input("Cholesterol", min_value=0.0, max_value=600.0, value=150.0)
bp = st.number_input("Resting Blood Pressure", min_value=0.0, max_value=300.0, value=120.0)

clinical_prediction = None

if st.button("Predict Clinical Risk"):

    clinical_input = np.zeros((1,21))

    clinical_input[0][0] = age
    clinical_input[0][1] = cholesterol
    clinical_input[0][2] = bp

    clinical_pred = clinical_model.predict(clinical_input)
    clinical_prob = clinical_model.predict_proba(clinical_input)

    # Risk Probability Graph
    st.subheader("Clinical Risk Probability")

    prob_df = pd.DataFrame({
        "Risk": ["Low Risk", "High Risk"],
        "Probability": clinical_prob[0]
    })

    st.bar_chart(prob_df.set_index("Risk"))

    if clinical_pred[0] == 1:
        st.error("Clinical Model Prediction: High Heart Disease Risk")
        clinical_prediction = 1
    else:
        st.success("Clinical Model Prediction: Low Heart Disease Risk")
        clinical_prediction = 0


# -----------------------------
# ECG Upload
# -----------------------------
st.header("Upload ECG Signal")

uploaded_file = st.file_uploader(
    "Upload ECG signal file",
    type=["csv","txt","npy"]
)

ecg_prediction = None

if uploaded_file is not None:

    try:

        # Read ECG CSV
        ecg_df = pd.read_csv(uploaded_file, header=None)

        ecg_data = ecg_df.values.flatten()

        # ECG Graph
        st.subheader("ECG Signal Visualization")

        signal_df = pd.DataFrame(ecg_data[:187], columns=["ECG Signal"])
        st.line_chart(signal_df)

        # Prepare ECG data for CNN
        ecg_data = ecg_data[:187]
        ecg_data = ecg_data.reshape(1,187,1)

        ecg_pred = ecg_model.predict(ecg_data)

        ecg_result = np.argmax(ecg_pred)

        if ecg_result == 0:
            st.success("ECG Model Prediction: Normal ECG")
            ecg_prediction = 0
        else:
            st.error("ECG Model Prediction: Abnormal ECG")
            ecg_prediction = 1

    except:
        st.warning("Please upload a valid ECG signal file.")


# -----------------------------
# Final Multimodal Prediction
# -----------------------------
st.header("Final Multimodal Prediction")

if clinical_prediction is not None and ecg_prediction is not None:

    final_score = (clinical_prediction + ecg_prediction) / 2

    if final_score >= 0.5:
        st.error("Final Prediction: High Heart Disease Risk")
    else:
        st.success("Final Prediction: Low Heart Disease Risk")