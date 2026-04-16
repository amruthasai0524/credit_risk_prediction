import streamlit as st
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner="Loading model... ⏳")
def load_pipeline():
    return joblib.load("pipeline_final.pkl")

pipeline = load_pipeline()

# ---------------- STYLE ----------------
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
    }
    .metric-card {
        background: linear-gradient(135deg, #6366f1, #06b6d4);
        padding: 15px;
        border-radius: 12px;
        color: white;
        text-align: center;
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Customer Inputs")

person_age = st.sidebar.number_input("Age", 18, 100, 30)

person_income = st.sidebar.number_input(
    "Income",
    min_value=0,
    max_value=10000000,
    value=50000,
    step=1000
)

person_emp_length = st.sidebar.number_input("Employment Years", 0, 50, 5)

loan_amnt = st.sidebar.number_input(
    "Loan Amount",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

loan_int_rate = st.sidebar.number_input(
    "Interest Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=10.0,
    step=0.5
)

loan_percent_income = st.sidebar.number_input(
    "Loan % Income",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.01
)

cb_person_cred_hist_length = st.sidebar.number_input(
    "Credit History Length",
    min_value=0,
    max_value=50,
    value=5
)

# ✅ AUTO-CALCULATED FEATURE (IMPORTANT)
income_to_loan_ratio = person_income / loan_amnt if loan_amnt != 0 else 0

person_home_ownership = st.sidebar.selectbox(
    "Home Ownership", ["RENT","OWN","MORTGAGE","OTHER"]
)

loan_intent = st.sidebar.selectbox(
    "Loan Purpose", ["PERSONAL","EDUCATION","MEDICAL","VENTURE","HOMEIMPROVEMENT","DEBTCONSOLIDATION"]
)

loan_grade = st.sidebar.selectbox(
    "Loan Grade", ["A","B","C","D","E","F","G"]
)

cb_person_default_on_file = st.sidebar.selectbox(
    "Previous Default", ["Y","N"]
)

# ---------------- MAIN ----------------
st.title("📊 Credit Risk Dashboard")
st.markdown("---")

# -------- METRIC CARDS --------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<div class='metric-card'>💰 Income<br>{person_income}</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div class='metric-card'>🏦 Loan<br>{loan_amnt}</div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div class='metric-card'>📈 Interest<br>{loan_int_rate}%</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Risk"):

    input_data = pd.DataFrame({
        'person_age':[person_age],
        'person_income':[person_income],
        'person_emp_length':[person_emp_length],
        'loan_amnt':[loan_amnt],
        'loan_int_rate':[loan_int_rate],
        'loan_percent_income':[loan_percent_income],
        'cb_person_cred_hist_length':[cb_person_cred_hist_length],
        'income_to_loan_ratio':[income_to_loan_ratio],
        'person_home_ownership':[person_home_ownership],
        'loan_intent':[loan_intent],
        'loan_grade':[loan_grade],
        'cb_person_default_on_file':[cb_person_default_on_file]
    })

    # Optional debug (remove later)
    st.write("Input Data:", input_data)

    prediction = pipeline.predict(input_data)[0]
    prob = pipeline.predict_proba(input_data)[0][1]

    st.markdown("## 📊 Risk Analysis")

    col4, col5 = st.columns(2)

    with col4:
        if prediction == 1:
            st.error("❌ High Risk Customer")
        else:
            st.success("✅ Low Risk Customer")

        st.progress(int(prob * 100))

    with col5:
        st.subheader("📌 Details")
        st.write(f"Risk Probability: {round(prob*100,2)}%")
        st.write(f"Loan Grade: {loan_grade}")
        st.write(f"Loan Purpose: {loan_intent}")

st.markdown("---")

# ---------------- FOOTER ----------------
st.markdown(
    "<center style='color:gray;'>🚀 Final Credit Risk ML Dashboard</center>",
    unsafe_allow_html=True
)