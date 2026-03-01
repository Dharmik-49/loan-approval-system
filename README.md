# Aura Finance: AI-Powered Loan Approval System 🏦🤖

**Aura Finance** is a full-stack, intelligent fintech application that uses Machine Learning to evaluate loan applications and leverages Google's Gemini AI to provide personalized, human-readable explanations for every financial decision.



## 🌟 Key Features
* **Smart Decision Engine:** Evaluates applications based on a custom-weighted Random Forest model prioritizing **Income (45%)**, **Employment History (40%)**, and **Credit Score (15%)**.
* **Dynamic Counter-Offers:** Instead of a rigid "Yes/No", the system calculates a `max_safe_amount` and automatically issues counter-offers if the requested amount is too high for the profile.
* **AI Loan Officer:** Integrates **Gemini 2.5 Flash** (via LangChain) to generate polite, contextual explanations and actionable financial advice for the applicant.
* **Interactive UI:** A responsive Streamlit frontend featuring dynamic color-coded alerts:
    * 💚 **Green:** Full Approval
    * 💛 **Yellow:** Partial Approval / Counter-Offer
    * ❤️ **Red:** Rejection with improvement tips

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Backend:** FastAPI, Uvicorn, Pydantic
* **Machine Learning:** Scikit-Learn (Random Forest), Pandas, NumPy
* **Generative AI:** LangChain, Google GenAI (`gemini-1.5-flash`)

## 🚀 Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/aura-finance.git](https://github.com/your-username/aura-finance.git)
cd aura-finance
```
2. Create a Virtual Environment (Recommended)
This keeps your project dependencies isolated from your global Python installation:

```Bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
3. Install Required Dependencies
Install the Machine Learning, Web, and AI libraries using the requirements file:

```Bash
pip install -r requirements.txt
```
4. Setup Environment Variables
The "AI Loan Officer" requires a Google Gemini API Key.

Create a new file named .env in the root folder.

Paste your key inside like this:

Code snippet
GOOGLE_API_KEY=your_actual_api_key_here

💻 How to Run the Application

Aura Finance operates with a Microservices Architecture, requiring two separate terminals.

Step 1: Start the Backend (API)
In your first terminal, run:

```Bash
python main.py
```
Note: The system checks for model.pkl. If not found, it automatically trains the Random Forest model using the weighted logic.

Status: Wait until you see Uvicorn running on http://0.0.0.0:8000.

Step 2: Start the Frontend (UI)
In a second terminal, run:

```Bash
streamlit run app.py
```
Note: A browser window will open automatically. Input applicant data to see the ML model and Gemini AI generate a decision and explanation in real-time.

📈 Decision Outcomes
Approved: Applicant meets the ML safety threshold for the full amount.

Counter-Offer: Applicant is approved for credit, but the requested amount exceeds the ML Regressor's "Safe Max." The AI offers a lower, safer amount instead.

Rejected: Applicant fails to meet the combined weighted criteria.