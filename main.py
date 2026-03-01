import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# --- NEW: Import Google GenAI ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables (gets your GOOGLE_API_KEY)
load_dotenv()

app = FastAPI()

# 1. Load the ML model
try:
    with open("model.pkl", "rb") as f:
        model_bundle = pickle.load(f)
        clf = model_bundle["classifier"]
        regr = model_bundle["regressor"]
except FileNotFoundError:
    print("Error: model.pkl not found! Run train_model.py first.")

# 2. Define Data Structure
class LoanData(BaseModel):
    income: float
    credit_score: int
    requested_amount: float
    employment_years: int

# 3. LangChain Setup for Gemini
# We use gemini-1.5-flash as it is fast, highly capable, and has a great free tier
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
# --- UPDATED PROMPT TEMPLATE ---
template = """
You are a professional Loan Officer. Based on the following data:
- Application Status: {status} (Approved, Rejected, or Counter-Offer)
- Amount Requested: ${requested_amt}
- Max Approved Amount: ${suggested_amt}
- Applicant Income: ${income}
- Credit Score: {credit_score}

Provide a polite, concise explanation to the applicant.
- If "Approved": Congratulate them and confirm the requested amount.
- If "Counter-Offer": Politely explain that the requested amount is too high based on their current income-to-loan ratio, but explicitly offer them the "Max Approved Amount" instead.
- If "Rejected": Explain why (e.g., credit score is too low) and give one actionable tip to improve.
"""
prompt = PromptTemplate.from_template(template)

# --- UPDATED BUSINESS LOGIC ---
@app.post("/analyze-loan")
async def analyze_loan(data: LoanData):
    features = [[data.income, data.credit_score, data.requested_amount, data.employment_years]]
    
    # 1. Get the absolute max the model thinks they can afford
    max_safe_amount = round(float(regr.predict(features)[0]), 2)
    
    # 2. Smart Business Logic
    if data.credit_score < 600:
        # Hard reject for bad credit
        final_status = "Rejected"
        offered_amount = 0
    elif data.requested_amount <= max_safe_amount:
        # Fully approved for what they asked
        final_status = "Approved"
        offered_amount = data.requested_amount
    else:
        # The Counter-Offer (They asked for too much, but we can give them the max_safe_amount)
        final_status = "Counter-Offer"
        offered_amount = max_safe_amount
    
    # 3. Let Gemini explain the logic beautifully
    chain = prompt | llm
    ai_explanation = chain.invoke({
        "status": final_status,
        "requested_amt": data.requested_amount,
        "suggested_amt": offered_amount,
        "income": data.income,
        "credit_score": data.credit_score
    })

    return {
        "decision": final_status,
        "max_amount": offered_amount,
        "explanation": ai_explanation.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)