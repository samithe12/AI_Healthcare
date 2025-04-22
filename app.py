# filename: app.py
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model
from Patients import Patient # Import the new class

# Load the trained model
try:
    model = load_model('best_classification_model') # Ensure this file exists
    # st.success("Model loaded.") # Optional success message
except Exception as e:
    st.error(f"Error loading model 'best_classification_model.pkl': {e}")
    st.stop()

st.title("💓 Heart Disease Risk Predictor")
st.write("Enter patient data (numeric values where applicable):")

# --- Original Input Fields using st.text_input ---
age = st.text_input("Age")
sex = st.text_input("Sex (0 = Female, 1 = Male)")
cp = st.text_input("Chest Pain Type (0-3)")
trestbps = st.text_input("Resting Blood Pressure")
chol = st.text_input("Cholesterol")
fbs = st.text_input("Fasting Blood Sugar (0 = False, 1 = True)")
restecg = st.text_input("Resting ECG (0-2)")
thalach = st.text_input("Max Heart Rate Achieved")
exang = st.text_input("Exercise-Induced Angina (0 = No, 1 = Yes)")
oldpeak = st.text_input("Oldpeak (ST Depression)")
slope = st.text_input("Slope (0-2)")
ca = st.text_input("Number of Major Vessels (0-4)") # Adjust range based on training data
thal = st.text_input("Thalassemia (0-3)") # Adjust range based on training data

# --- Predict on Button Click ---
if st.button("Predict"):
    # Check if all fields are filled
    if not all([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
        st.warning("Please fill in all fields.")
    else:
        try:
            # 1. Create a Patient instance from the text inputs
            # Type conversion and validation happen within the Patient class __init__
            patient_data = Patient(
                age=age,
                sex=sex,
                cp=cp,
                trestbps=trestbps,
                chol=chol,
                fbs=fbs,
                restecg=restecg,
                thalach=thalach,
                exang=exang,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal
            )

            # 2. Convert the Patient object to a DataFrame
            input_df = patient_data.to_dataframe()

            # 3. Make prediction using the loaded model
            predictions = predict_model(estimator=model, data=input_df)

            # 4. Display Results
            risk_label = predictions['prediction_label'].iloc[0]
            confidence_score = predictions['prediction_score'].iloc[0] if 'prediction_score' in predictions.columns else None

            st.subheader("Prediction Result:")
            if risk_label == 1:
                st.error("⚠️ High Risk of Heart Disease Detected")
            else:
                st.success("✅ Low Risk of Heart Disease Detected")

            if confidence_score is not None:
                display_score = confidence_score if risk_label == 1 else (1 - confidence_score)
                st.write(f"Confidence Score (for predicted class): {display_score:.2%}")
            st.write("---")
            st.write("*Disclaimer: This prediction is based on a machine learning model and should not replace professional medical advice.*")


        except ValueError as ve: # Catch specific errors from Patient class validation
            st.error(f"Input Error: {ve}")
        except Exception as e: # Catch other errors (e.g., prediction errors)
            st.error(f"An error occurred:")
            st.error(e)
            st.warning("Please check your input values.")