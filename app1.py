
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EMI Risk Assessment", page_icon="📊", layout="centered")

MODELS_DIR = "Model"

def load_artifact(name):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return joblib.load(path)

def safe_div(a, b):
    try:
        return float(a)/float(b) if float(b) != 0 else 0
    except:
        return 0

def engineer_features(raw):
    monthly_salary = float(raw["monthly_salary"])
    total_expenses = (
        float(raw["monthly_rent"]) +
        float(raw["school_fees"]) +
        float(raw["college_fees"]) +
        float(raw["travel_expenses"]) +
        float(raw["groceries_utilities"]) +
        float(raw["other_monthly_expenses"]) +
        float(raw["current_emi_amount"])
    )

    df = pd.DataFrame([{
        "monthly_salary": monthly_salary,
        "total_expenses": total_expenses,
        "monthly_savings": monthly_salary - total_expenses,
        "expense_ratio": safe_div(total_expenses, monthly_salary),
        "emi_salary_ratio": safe_div(raw["current_emi_amount"], monthly_salary),
        "balance_salary_ratio": safe_div(raw["bank_balance"], monthly_salary),
        "credit_score": float(raw["credit_score"]),
        "requested_amount": float(raw["requested_amount"]),
        "requested_tenure": float(raw["requested_tenure"]),
        "bank_balance": float(raw["bank_balance"]),
        "years_of_employment": float(raw["years_of_employment"]),
        "dependents_ratio": safe_div(raw["dependents"], raw["family_size"])
    }])

    return df

def predict(clf_model, clf_scaler, clf_features, reg_model, reg_scaler, reg_features, df):
    for col in clf_features:
        if col not in df.columns:
            df[col] = 0

    for col in reg_features:
        if col not in df.columns:
            df[col] = 0

    Xc = clf_scaler.transform(df[clf_features])
    Xr = reg_scaler.transform(df[reg_features])

    eligibility = int(clf_model.predict(Xc)[0])
    max_emi = float(reg_model.predict(Xr)[0])

    return eligibility, max_emi


st.title("📊 EMI Prediction & Financial Risk Assessment")

st.sidebar.title("Model Loading Status")

try:
    clf_model = load_artifact("best_eligibility_model_compressed.pkl")
    clf_scaler = load_artifact("eligibility_scaler.pkl")
    clf_features = load_artifact("eligibility_features.pkl")

    reg_model = load_artifact("best_max_emi_model_compressed.pkl")
    reg_scaler = load_artifact("emi_scaler.pkl")
    reg_features = load_artifact("emi_features.pkl")

    st.sidebar.success("✅ Models Loaded Successfully")
    ready = True
except Exception as e:
    st.sidebar.error(f"❌ Model load error: {e}")
    ready = False


with st.form("emi_form"):
    col1, col2 = st.columns(2)

    with col1:
        monthly_salary = st.number_input("Monthly Salary (₹)", 1000, 2000000, 50000)
        credit_score = st.number_input("Credit Score", 300, 900, 720)
        bank_balance = st.number_input("Bank Balance (₹)", 0, 99999999, 50000)
        years_of_employment = st.number_input("Years of Employment", 0.0, 50.0, 3.0)
        family_size = st.number_input("Family Size", 1, 20, 4)
        dependents = st.number_input("Dependents", 0, 15, 1)

    with col2:
        requested_amount = st.number_input("Requested Loan Amount (₹)", 0, 9999999, 300000)
        requested_tenure = st.number_input("Tenure (Months)", 1, 360, 24)
        current_emi_amount = st.number_input("Current EMI (₹)", 0, 500000, 0)
        monthly_rent = st.number_input("Monthly Rent (₹)", 0, 200000, 8000)
        school_fees = st.number_input("School Fees (₹)", 0, 200000, 0)
        college_fees = st.number_input("College Fees (₹)", 0, 200000, 0)
        travel_expenses = st.number_input("Travel Expenses (₹)", 0, 200000, 3000)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 200000, 10000)
        other_monthly_expenses = st.number_input("Other Expenses (₹)", 0, 200000, 2000)

    submit = st.form_submit_button("Predict EMI & Eligibility")

if submit:
    if not ready:
        st.error("❌ Models not loaded properly.")
    else:
        raw = {
            "monthly_salary": monthly_salary,
            "monthly_rent": monthly_rent,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "years_of_employment": years_of_employment,
            "family_size": family_size,
            "dependents": dependents,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure
        }

        df = engineer_features(raw)
        elig, emi_amount = predict(
            clf_model, clf_scaler, clf_features,
            reg_model, reg_scaler, reg_features,
            df
        )

        st.success(f"✅ Eligibility: {'Eligible' if elig == 1 else 'Not Eligible'}")
        st.info(f"✅ Maximum EMI you can afford: ₹{emi_amount:,.0f}")
