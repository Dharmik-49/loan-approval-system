import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="AI Loan Officer", page_icon="🏦")

st.title("🏦 Smart Loan Approval System")
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
        income = st.number_input("Annual Income (₹)", min_value=0, step=1000)
        employment_years = st.number_input("Years of Employment", min_value=0, max_value=50)

    with col2:
        credit_score = st.slider("Credit Score", 300, 850, 650)
        loan_amount = st.number_input("Requested Loan Amount (₹)", min_value=0, step=500)
        loan_purpose = st.selectbox("Purpose", ["Home", "Education", "Personal", "Business"])

    submit_button = st.form_submit_button(label="Analyze Application")

# Logic to connect to the Backend
## Logic to connect to the Backend
if submit_button:
    if not full_name:
        st.error("Please enter the applicant's name.")
    else:
        with st.spinner("Agent is analyzing creditworthiness..."):
            try:
                payload = {
                    "income": income,
                    "credit_score": credit_score,
                    "requested_amount": loan_amount, 
                    "employment_years": employment_years
                }
                
                # Sending request to the FastAPI backend
                response = requests.post("http://localhost:8000/analyze-loan", json=payload)
                
                if response.status_code == 200:
                    # Parse the JSON response
                    data = response.json()
                    decision = data.get("decision", "Unknown")
                    max_amount = data.get("max_amount", 0)
                    explanation = data.get("explanation", "No explanation provided.")
                    
                    st.subheader(f"Decision for {full_name}:")
                    
                    # --- DYNAMIC UI COLOR LOGIC ---
                    if decision == "Approved":
                        st.success(f"🎉 **Status: {decision}**")
                        st.write(f"**Approved Amount:** ${max_amount:,.2f}")
                        st.info(f"**AI Officer Note:** {explanation}")
                        st.balloons() # Adds a nice UI celebration
                        
                    elif decision == "Counter-Offer":
                        st.warning(f"⚠️ **Status: {decision}**")
                        st.write(f"**Maximum Eligible Amount:** ${max_amount:,.2f}")
                        st.info(f"**AI Officer Note:** {explanation}")
                        
                    elif decision == "Rejected":
                        st.error(f"❌ **Status: {decision}**")
                        st.info(f"**AI Officer Note:** {explanation}")
                        
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"Could not connect to backend: {e}")

st.divider()
st.caption("Experimental AI Model - For demonstration purposes only.")