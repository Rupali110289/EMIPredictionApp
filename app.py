# app.py
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="EMI Risk Assessment", page_icon="📊", layout="centered")

def safe_div(a, b):
    try:
        if b == 0:
            return 0
        return float(a) / float(b)
    except:
        return 0

def engineer_features(raw):
    monthly_salary = float(raw["monthly_salary"])
    monthly_rent = float(raw.get("monthly_rent", 0))
    school_fees = float(raw.get("school_fees", 0))
    college_fees = float(raw.get("college_fees", 0))
    travel_expenses = float(raw.get("travel_expenses", 0))
    groceries_utilities = float(raw.get("groceries_utilities", 0))
    other_monthly_expenses = float(raw.get("other_monthly_expenses", 0))
    current_emi_amount = float(raw.get("current_emi_amount", 0))
    credit_score = float(raw["credit_score"])
    bank_balance = float(raw["bank_balance"])
    years_of_employment = float(raw["years_of_employment"])
    family_size = float(raw["family_size"])
    dependents = float(raw["dependents"])
    requested_amount = float(raw["requested_amount"])
    requested_tenure = float(raw["requested_tenure"])

    total_expenses = (monthly_rent + school_fees + college_fees + travel_expenses +
                      groceries_utilities + other_monthly_expenses + current_emi_amount)
    monthly_savings = monthly_salary - total_expenses
    expense_ratio = safe_div(total_expenses, monthly_salary)
    emi_salary_ratio = safe_div(current_emi_amount, monthly_salary)
    balance_salary_ratio = safe_div(bank_balance, monthly_salary)
    dependents_ratio = safe_div(dependents, family_size)

    data = {
        "monthly_salary": monthly_salary,
        "total_expenses": total_expenses,
        "monthly_savings": monthly_savings,
        "expense_ratio": expense_ratio,
        "emi_salary_ratio": emi_salary_ratio,
        "balance_salary_ratio": balance_salary_ratio,
        "credit_score": credit_score,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "bank_balance": bank_balance,
        "years_of_employment": years_of_employment,
        "dependents_ratio": dependents_ratio
    }
    return pd.DataFrame([data]), data

def predict_with_local(clf_model, clf_scaler, clf_features, reg_model, reg_scaler, reg_features, df):
    for c in clf_features:
        if c not in df.columns:
            df[c] = 0
    for c in reg_features:
        if c not in df.columns:
            df[c] = 0

    Xc = clf_scaler.transform(df[clf_features])
    Xr = reg_scaler.transform(df[reg_features])

    elig = int(clf_model.predict(Xc)[0])
    emi = float(reg_model.predict(Xr)[0])
    return elig, emi

st.title("📊 EMI Risk Assessment & EMI Calculator")

try:
    clf_model = joblib.load("best_eligibility_model.pkl")
    clf_scaler = joblib.load("eligibility_scaler.pkl")
    clf_features = joblib.load("eligibility_features.pkl")

    reg_model = joblib.load("best_max_emi_model.pkl")
    reg_scaler = joblib.load("emi_scaler.pkl")
    reg_features = joblib.load("emi_features.pkl")

    artifacts_loaded = True
except:
    artifacts_loaded = False
    st.error("❌ Model files missing. Upload .pkl files to run predictions.")

st.subheader("Enter Applicant Details")

with st.form("form"):
    c1, c2 = st.columns(2)
    with c1:
        monthly_salary = st.number_input("Monthly Salary (₹)", 1000, 10000000, 60000)
        credit_score = st.number_input("Credit Score", 300, 900, 720)
        bank_balance = st.number_input("Bank Balance (₹)", 0, 50000000, 150000)
        years_of_employment = st.number_input("Years of Employment", 0.0, 50.0, 4.0)
        family_size = st.number_input("Family Size", 1, 20, 4)
        dependents = st.number_input("Dependents", 0, 10, 2)

    with c2:
        requested_amount = st.number_input("Requested Loan Amount (₹)", 0, 50000000, 300000)
        requested_tenure = st.number_input("Requested Tenure (Months)", 3, 360, 24)
        current_emi_amount = st.number_input("Current EMI (₹)", 0, 300000, 0)
        monthly_rent = st.number_input("Monthly Rent (₹)", 0, 200000, 10000)
        school_fees = st.number_input("School Fees (₹)", 0, 200000, 0)
        college_fees = st.number_input("College Fees (₹)", 0, 200000, 0)
        travel_expenses = st.number_input("Travel Expenses (₹)", 0, 50000, 3000)
        groceries_utilities = st.number_input("Groceries & Utilities (₹)", 0, 50000, 10000)
        other_monthly_expenses = st.number_input("Other Expenses (₹)", 0, 50000, 2000)

    submit = st.form_submit_button("🔮 Predict")

if submit and artifacts_loaded:
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

    df_eng, snapshot = engineer_features(raw)

    st.write("### ✅ Engineered Features Used by Model")
    st.dataframe(pd.DataFrame([snapshot]).T)

    elig, emi = predict_with_local(
        clf_model, clf_scaler, clf_features,
        reg_model, reg_scaler, reg_features,
        df_eng.copy()
    )

    st.success(f"✅ EMI Eligibility: {'Eligible ✅' if elig == 1 else 'Not Eligible ❌'}")
    st.info(f"✅ Maximum Affordable EMI: ₹{emi:,.0f}")
