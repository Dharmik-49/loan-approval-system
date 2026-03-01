import os
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Load environment variables (Google API Key)
load_dotenv()

app = FastAPI()

# ==========================================
# 1. MACHINE LEARNING: LOAN MODEL TRAINING
# ==========================================
def train_and_save_loan_model():
    print("⚙️ Training Loan ML models with custom weights...")
    np.random.seed(42)
    n = 2000
    income = np.random.randint(30000, 150000, n)
    credit_score = np.random.randint(300, 850, n)
    employment_years = np.random.randint(0, 20, n)
    requested_amount = np.random.randint(5000, 50000, n)
    
    # Custom Weighted Logic: Income(45%), Employment(40%), Credit(15%)
    norm_income = income / 150000
    norm_years = employment_years / 20
    norm_credit = credit_score / 850
    applicant_score = (norm_income * 0.45) + (norm_years * 0.40) + (norm_credit * 0.15)
    
    status = ((applicant_score > 0.35) & (requested_amount < (income * 0.6))).astype(int)
    suggested_amount = (income * 0.4) * (1 + (norm_years * 0.5)) * norm_credit
    
    df = pd.DataFrame({
        'income': income, 'credit_score': credit_score,
        'requested_amount': requested_amount, 'employment_years': employment_years,
        'status': status, 'suggested_amount': suggested_amount
    })
    
    X = df[['income', 'credit_score', 'requested_amount', 'employment_years']]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, df['status'])
    
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X, df['suggested_amount'])
    
    with open("model.pkl", "wb") as f:
        pickle.dump({"classifier": clf, "regressor": regr}, f)
    print("✅ Loan models saved!")

# Initialize models on startup
if not os.path.exists("model.pkl"):
    train_and_save_loan_model()

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)
    loan_clf = bundle["classifier"]
    loan_regr = bundle["regressor"]

# ==========================================
# 2. SCHEMAS & AI CONFIG
# ==========================================
class LoanData(BaseModel):
    income: float
    credit_score: int
    requested_amount: float
    employment_years: int

class StockData(BaseModel):
    prices: list[float]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# ==========================================
# 3. ENDPOINTS
# ==========================================

@app.post("/analyze-loan")
async def analyze_loan(data: LoanData):
    features = [[data.income, data.credit_score, data.requested_amount, data.employment_years]]
    
    ml_approved = int(loan_clf.predict(features)[0])
    max_safe = round(float(loan_regr.predict(features)[0]), 2)
    
    final_status = "Rejected"
    offered_amt = 0
    if ml_approved == 1:
        if data.requested_amount <= max_safe:
            final_status = "Approved"
            offered_amt = data.requested_amount
        else:
            final_status = "Counter-Offer"
            offered_amt = max_safe

    # Gemini AI Explanation
    template = "As a loan officer for Aura Finance, explain why this application was {s}. Requested: ${r}, Max Allowed: ${m}. Income: ${i}, Credit: {c}."
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    ai_msg = chain.invoke({"s": final_status, "r": data.requested_amount, "m": offered_amt, "i": data.income, "c": data.credit_score})

    return {"decision": final_status, "max_amount": offered_amt, "explanation": ai_msg.content}

@app.post("/predict-stock")
async def predict_stock(data: StockData):
    if len(data.prices) < 5:
        raise HTTPException(status_code=400, detail="Provide at least 5 days of price history")
    
    prices = data.prices
    X, y = [], []
    for i in range(len(prices) - 3):
        X.append(prices[i:i+3])
        y.append(prices[i+3])
    
    stock_model = RandomForestRegressor(n_estimators=50)
    stock_model.fit(X, y)
    
    prediction = stock_model.predict([prices[-3:]])[0]
    return {"predicted_price": round(float(prediction), 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)