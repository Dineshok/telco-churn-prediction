import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ── Load model, scaler, feature columns ──────────────────────────────
model           = pickle.load(open('model.pkl', 'rb'))
scaler          = pickle.load(open('scaler.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))

num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📡", layout="centered")

st.title("📡 Customer Churn Predictor")
st.markdown("Fill in the customer details below and click **Predict** to see if they're likely to churn.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    gender           = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen   = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner          = st.selectbox("Partner", ["Yes", "No"])
    dependents       = st.selectbox("Dependents", ["Yes", "No"])
    tenure           = st.slider("Tenure (months)", 0, 72, 12)
    phone_service    = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines   = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    online_security  = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup    = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protect   = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract         = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless        = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment          = st.selectbox("Payment Method", [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])

st.divider()
monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
total_charges   = st.slider("Total Charges ($)", 0.0, 9000.0, float(tenure * monthly_charges))

# ── Predict button ────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True):

    # Build raw input dict
    raw = {
        'SeniorCitizen':    1 if senior_citizen == "Yes" else 0,
        'tenure':           tenure,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
        'gender_Male':                              1 if gender == "Male" else 0,
        'Partner_Yes':                              1 if partner == "Yes" else 0,
        'Dependents_Yes':                           1 if dependents == "Yes" else 0,
        'PhoneService_Yes':                         1 if phone_service == "Yes" else 0,
        'MultipleLines_No phone service':           1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes':                        1 if multiple_lines == "Yes" else 0,
        'InternetService_Fiber optic':              1 if internet_service == "Fiber optic" else 0,
        'InternetService_No':                       1 if internet_service == "No" else 0,
        'OnlineSecurity_No internet service':       1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes':                       1 if online_security == "Yes" else 0,
        'OnlineBackup_No internet service':         1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes':                         1 if online_backup == "Yes" else 0,
        'DeviceProtection_No internet service':     1 if device_protect == "No internet service" else 0,
        'DeviceProtection_Yes':                     1 if device_protect == "Yes" else 0,
        'TechSupport_No internet service':          1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes':                          1 if tech_support == "Yes" else 0,
        'StreamingTV_No internet service':          1 if streaming_tv == "No internet service" else 0,
        'StreamingTV_Yes':                          1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No internet service':      1 if streaming_movies == "No internet service" else 0,
        'StreamingMovies_Yes':                      1 if streaming_movies == "Yes" else 0,
        'Contract_One year':                        1 if contract == "One year" else 0,
        'Contract_Two year':                        1 if contract == "Two year" else 0,
        'PaperlessBilling_Yes':                     1 if paperless == "Yes" else 0,
        'PaymentMethod_Credit card (automatic)':    1 if payment == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check':           1 if payment == "Electronic check" else 0,
        'PaymentMethod_Mailed check':               1 if payment == "Mailed check" else 0,
    }

    # Align to training feature order
    input_df = pd.DataFrame([raw])[feature_columns]

    # Scale numerical columns
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    prob      = model.predict_proba(input_df)[0][1]
    threshold = 0.4   # tuned threshold for better recall
    pred      = 1 if prob >= threshold else 0

    st.divider()
    if pred == 1:
        st.error(f"⚠️ This customer is **likely to churn**.")
    else:
        st.success(f"✅ This customer is **likely to stay**.")

    st.metric(label="Churn Probability", value=f"{prob:.1%}")
    st.progress(float(prob))

    # Risk breakdown
    st.divider()
    st.markdown("### 🔍 Risk Factors")
    risk_factors = []
    if tenure < 12:
        risk_factors.append("🔴 Low tenure (less than 1 year)")
    if contract == "Month-to-month":
        risk_factors.append("🔴 Month-to-month contract (no commitment)")
    if internet_service == "Fiber optic":
        risk_factors.append("🟡 Fiber optic internet (high churn segment)")
    if monthly_charges > 70:
        risk_factors.append("🟡 High monthly charges")
    if payment == "Electronic check":
        risk_factors.append("🟡 Electronic check payment method")
    if not risk_factors:
        risk_factors.append("🟢 No major risk factors detected")

    for r in risk_factors:
        st.markdown(r)