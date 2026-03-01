import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

def generate_data(n=2000): # Increased sample size for better learning
    np.random.seed(42)
    income = np.random.randint(30000, 150000, n)
    credit_score = np.random.randint(300, 850, n)
    employment_years = np.random.randint(0, 20, n)
    requested_amount = np.random.randint(5000, 50000, n)
    
    # --- NEW WEIGHTED LOGIC ---
    # Normalize the values to a 0-1 scale so we can apply weights properly
    norm_income = income / 150000
    norm_years = employment_years / 20
    norm_credit = credit_score / 850
    
    # Apply your custom weights: 45% Income, 40% Work Duration, 15% Credit Score
    applicant_score = (norm_income * 0.45) + (norm_years * 0.40) + (norm_credit * 0.15)
    
    # 1. Approval Logic: Approve if their weighted score is decent AND they aren't asking for more than 60% of their income
    status = ((applicant_score > 0.35) & (requested_amount < (income * 0.6))).astype(int)
    
    # 2. Suggested Amount Logic: Heavily tied to income and employment duration now
    suggested_amount = (income * 0.4) * (1 + (norm_years * 0.5)) * norm_credit
    
    df = pd.DataFrame({
        'income': income,
        'credit_score': credit_score,
        'requested_amount': requested_amount,
        'employment_years': employment_years,
        'status': status,
        'suggested_amount': suggested_amount
    })
    return df

def train():
    df = generate_data()
    X = df[['income', 'credit_score', 'requested_amount', 'employment_years']]
    
    # Train the Classifier (Approval)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, df['status'])
    
    # Train the Regressor (Amount)
    regr = RandomForestRegressor(n_estimators=100, random_state=42)
    regr.fit(X, df['suggested_amount'])
    
    model_bundle = {
        "classifier": clf,
        "regressor": regr
    }
    
    with open("model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)
    
    print("✅ model.pkl retrained successfully with new Income/Employment heavy weights!")

if __name__ == "__main__":
    train()