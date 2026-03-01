import pickle
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

app = FastAPI()

# 1. Load the model from your Desktop folder
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

# 3. LangChain Setup (Set your OPENAI_API_KEY in environment)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

template = """
You are a professional Loan Officer. Based on the following data:
- ML Approval Status: {status} (1=Approved, 0=Rejected)
- ML Suggested Max Loan: ${suggested_amt}
- Applicant Income: ${income}
- Credit Score: {credit_score}

Provide a polite, concise explanation to the applicant about why they were approved or rejected. 
If approved, mention the limit. If rejected, give one tip to improve.
"""
prompt = PromptTemplate.from_template(template)

@app.post("/analyze-loan")
async def analyze_loan(data: LoanData):
    # Prepare data for ML model
    features = [[data.income, data.credit_score, data.requested_amount, data.employment_years]]
    
    # Get ML Predictions
    status_pred = int(clf.predict(features)[0])
    amt_pred = round(float(regr.predict(features)[0]), 2)
    
    # LangChain Reasoning
    chain = prompt | llm
    ai_explanation = chain.invoke({
        "status": status_pred,
        "suggested_amt": amt_pred,
        "income": data.income,
        "credit_score": data.credit_score
    })

    return {
        "decision": "Approved" if status_pred == 1 else "Rejected",
        "max_amount": amt_pred if status_pred == 1 else 0,
        "explanation": ai_explanation.content
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)