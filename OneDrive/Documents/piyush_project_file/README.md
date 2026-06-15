🏦 AI-Powered Customer Churn Intelligence Dashboard

This project is a machine learning-based web application built using Streamlit that predicts customer churn for a banking system. It combines predictive modeling with explainable AI to provide both predictions and insights into why a customer is likely to leave.

🚀 Live Demo

https://churn-dashboard-vwm3kvveuot3wkormztr8t.streamlit.app/

📊 Project Overview

Customer churn is a major challenge in the banking industry, directly affecting revenue and customer retention strategies. This application uses historical banking data to predict whether a customer will exit the bank.

Along with prediction, the system also explains model behavior so that decisions are interpretable and actionable.

🧠 Machine Learning Models

The system supports multiple machine learning models for comparison and selection:

Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost

Each model can be selected dynamically from the dashboard to analyze performance differences.

⚙️ Key Features
📊 Real-Time Prediction

The application takes customer input and predicts churn probability instantly, along with a risk classification based on a configurable threshold.

🔍 Explainable AI (SHAP)

SHAP-based explanations are used to interpret individual predictions and identify the most influential features contributing to churn.

📈 Model Evaluation

Performance is evaluated using ROC curve, AUC score, and standard classification metrics including accuracy, precision, recall, and F1-score. A calibration curve is also included to assess probability reliability.

🎛 Interactive Interface

Built with Streamlit, the dashboard allows interactive input, model switching, threshold adjustment, and real-time visualization of results.

📁 Project Structure
European_Bank.csv
README.md
app.py
feature_engineering.py
models.pkl
requirements.txt
train_model.py
📊 Dataset Information

The dataset contains customer-level banking information including:

Credit Score
Geography
Gender
Age
Tenure
Balance
Number of Products
Credit Card Status
Activity Status
Estimated Salary

Target variable:

Exited (1 = Churn, 0 = Retained)
🧰 Technologies Used
Python
Streamlit
Scikit-learn
XGBoost
SHAP
Plotly
Pandas
NumPy
📌 Notes

<<<<<<< HEAD
This project is designed for learning and demonstration purposes, showcasing end-to-end machine learning workflow including data processing, model training, explainability, and deployment in an interactive dashboard.
=======
### Input Features:
- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Has Credit Card
- Is Active Member
- Estimated Salary

### Target Variable:
- `Exited`
  - 1 → Customer churned
  - 0 → Customer retained

---

## 🛠 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/piyush391/churn-dashboard.git
<<<<<<< HEAD
cd churn-dashboard
=======
cd churn-dashboard
>>>>>>> 29481f8 (Initial commit)
>>>>>>> 69c26ea (upload full churn dashboard project)
