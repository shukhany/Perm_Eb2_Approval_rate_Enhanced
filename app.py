
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="EB-2 PERM Approval Probability Estimator (FY2024)", layout="wide")

# Load model
model = joblib.load("model_perm_best.pkl")

st.title("EB-2 PERM Approval Probability Estimator (FY2024)")
st.write("This tool estimates the probability that a PERM case will be certified based on FY2024 patterns.")

# ----- User Inputs -----
pw_soc_code = st.text_input("Prevailing Wage SOC Code", "15-1252")
naics_code = st.text_input("Employer NAICS Code", "5415")
pw_wage = st.number_input("Prevailing Wage Amount", 0.0, step=1000.0, value=115000.0)
pw_unit = st.selectbox("Prevailing Wage Unit", ["Year", "Hour", "Month", "Week"])
min_edu = st.selectbox("Minimum Education Required", ["Bachelor's", "Master's", "High School", "Doctorate"])
state = st.text_input("Worksite State (e.g., CA, TX)", "CA")
wage_from = st.number_input("Wage Offer From", 0.0, step=1000.0, value=130000.0)
wage_to = st.number_input("Wage Offer To", 0.0, step=1000.0, value=140000.0)
wage_unit = st.selectbox("Wage Offer Unit", ["Year", "Hour", "Month", "Week"])
ownership = st.selectbox("Ownership Interest?", ["N", "Y"])

# ---- Convert to model features ----
def to_annual(amount, unit):
    if unit == "Year":
        return amount
    elif unit == "Hour":
        return amount * 2080
    elif unit == "Week":
        return amount * 52
    elif unit == "Month":
        return amount * 12
    else:
        return amount

pw_wage_annual = to_annual(pw_wage, pw_unit)
offer_wage_annual = to_annual((wage_from + wage_to) / 2, wage_unit)
wage_ratio = offer_wage_annual / pw_wage_annual if pw_wage_annual > 0 else 1.0

input_data = pd.DataFrame([{
    "PW_SOC_CODE": pw_soc_code,
    "NAICS_CODE": naics_code,
    "PW_UNIT_OF_PAY": pw_unit,
    "WAGE_OFFER_UNIT_OF_PAY": wage_unit,
    "MINIMUM_EDUCATION": min_edu,
    "WORKSITE_STATE": state,
    "FW_OWNERSHIP_INTEREST": ownership,
    "PW_WAGE_ANNUAL": pw_wage_annual,
    "OFFER_WAGE_ANNUAL": offer_wage_annual,
    "WAGE_RATIO": wage_ratio
}])

# ---- Prediction ----
if st.button("Estimate Approval Probability"):
    try:
        prob = model.predict_proba(input_data)[0, 1] * 100
        st.success(f"Estimated Approval Probability: **{prob:.1f}%**")
    except Exception as e:
        st.error("An error occurred during scoring:")
        st.error(str(e))
