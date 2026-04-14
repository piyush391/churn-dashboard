# ================================
# TRAIN MULTI-MODEL CHURN SYSTEM
# ================================

import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import calibration_curve 
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Safe feature engineering
        X["BalanceSalaryRatio"] = X["Balance"] / (X["EstimatedSalary"] + 1)
        X["ProductDensity"] = X["NumOfProducts"] / (X["Age"] + 1)
        X["AgeTenureInteraction"] = X["Age"] * X["Tenure"]

        return X
    
# Optional XGBoost (safe fallback)
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False


# ================================
# LOAD DATA
# ================================
df = pd.read_csv("European_Bank.csv")

# ================================
# BASIC CLEANING
# ================================
df = df.drop(columns=["CustomerId", "Surname"], errors="ignore")
df = df.drop(columns=["Year"], errors="ignore")

target = "Exited"
X = df.drop(columns=[target])
y = df[target]

# ================================
# FEATURE TYPES
# ================================
cat_features = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]
num_features = [col for col in X.columns if col not in cat_features]

# force string consistency (IMPORTANT for Streamlit + ROC)
for col in cat_features:
    X[col] = X[col].astype(str)

# ================================
# PREPROCESSOR
# ================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

# ================================
# MODELS
# ================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier()
}

from xgboost import XGBClassifier

models["XGBoost"] = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss"
)

# ================================
# TRAIN / TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ================================
# TRAIN ALL MODELS
# ================================
trained_models = {}

print("\nTraining Models...\n")

for name, clf in models.items():

    print(f"Training: {name}")

    pipeline = Pipeline(steps=[
    ("feature_engineering", FeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", clf)
])

    pipeline.fit(X_train, y_train)

    # evaluation
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_prob)
        print(f"{name} ROC-AUC: {score:.4f}")
    except:
        print(f"{name} ROC-AUC: Not available")

    trained_models[name] = pipeline

# ================================
# SAVE ALL MODELS
# ================================
with open("models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

print("\nSaved models.pkl successfully!")