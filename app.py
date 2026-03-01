import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="Aura Finance AI", page_icon="🏦")
st.title("Aura Finance: Smart Loan Approval System 🏦🤖")


st.markdown("Enter the applicant's details below to get an AI-powered credit decision.")

# Sidebar for additional info
with st.sidebar:
    st.header("System Status")
    st.success("Backend: Connected")
    st.info("Model: LangChain + Scikit-Learn")

# Input Form
with st.form("loan_application_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        full_name = st.text_input("Full Name")
        income = st.number_input("Annual Income ($)", min_value=0, step=1000)
        employment_years = st.number_input("Years of Employment", min_value=0, max_value=50)

    with col2:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0, step=500)
        loan_purpose = st.selectbox("Purpose", ["Home", "Education", "Personal", "Business"])

    submit_button = st.form_submit_button(label="Analyze Application")

# Logic to connect to the Backend
if submit_button:
    if not full_name:
        st.error("Please enter the applicant's name.")
    else:
        with st.spinner("Agent is analyzing creditworthiness..."):
            try:
                # This matches the FastAPI endpoint we'll build next
                payload = {
                    "income": income,
                    "credit_score": credit_score,
                    "loan_amount_requested": loan_amount,
                    "employment_years": employment_years
                }
                
                # Sending request to the FastAPI backend (Port 8000)
                response = requests.post("http://localhost:8000/analyze-loan", json=payload)
                
                if response.status_status == 200:
                    result = response.json().get("decision")
                    st.subheader(f"Final Decision for {full_name}:")
                    st.write(result)
                else:
                    st.error("Backend error. Make sure the FastAPI server is running.")
            
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

st.divider()
st.caption("Experimental AI Model - For demonstration purposes only.")