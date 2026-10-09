
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦")
st.title("Loan Approval Prediction")
st.write("Enter the applicant's details below.")

@st.cache_resource
def train_model():
    df = pd.read_csv("loan_approval_dataset.csv")
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["loan_id"], errors="ignore")
    df["loan_status"] = df["loan_status"].str.strip()

    X = pd.get_dummies(df.drop(columns=["loan_status"]), drop_first=True)
    y = df["loan_status"]

    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X, y)
    return model, X.columns

model, columns = train_model()

dependents = st.number_input("Number of Dependents", 0, 10, 0)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Annual Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.number_input("Loan Term (Months)", min_value=1, value=12)
cibil = st.number_input("CIBIL Score", 300, 900, 700)
residential = st.number_input("Residential Assets Value", min_value=0)
commercial = st.number_input("Commercial Assets Value", min_value=0)
luxury = st.number_input("Luxury Assets Value", min_value=0)
bank = st.number_input("Bank Asset Value", min_value=0)

if st.button("Predict Loan Status"):
    row = pd.DataFrame([{
        "no_of_dependents": dependents,
        "education": education,
        "self_employed": employed,
        "income_annum": income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil,
        "residential_assets_value": residential,
        "commercial_assets_value": commercial,
        "luxury_assets_value": luxury,
        "bank_asset_value": bank
    }])

    row = pd.get_dummies(row, drop_first=True)
    row = row.reindex(columns=columns, fill_value=0)
    prediction = model.predict(row)[0]

    if prediction.lower() == "approved":
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")