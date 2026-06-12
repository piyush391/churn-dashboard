# train_model.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ==================================

# LOAD DATA

# ==================================

df = pd.read_csv("European_Bank.csv")

# ==================================

# CLEAN DATA

# ==================================

df = df.drop(columns=["CustomerId", "Surname"], errors="ignore")
df = df.drop(columns=["Year"], errors="ignore")

# ==================================

# FEATURE ENGINEERING

# (DIRECTLY IN DATAFRAME)

# ==================================

df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
df["ProductDensity"] = df["NumOfProducts"] / (df["Age"] + 1)
df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

# ==================================

# TARGET

# ==================================

target = "Exited"

X = df.drop(columns=[target])
y = df[target]

# ==================================

# FEATURE TYPES

# ==================================

cat_features = [
"Geography",
"Gender",
"HasCrCard",
"IsActiveMember"
]

for col in cat_features:
X[col] = X[col].astype(str)

num_features = [c for c in X.columns if c not in cat_features]

# ==================================

# PREPROCESSOR

# ==================================

preprocessor = ColumnTransformer(
transformers=[
(
"num",
StandardScaler(),
num_features
),
(
"cat",
OneHotEncoder(handle_unknown="ignore"),
cat_features
)
]
)

# ==================================

# MODELS

# ==================================

models = {
"Logistic Regression": LogisticRegression(
max_iter=1000
),

```
"Decision Tree": DecisionTreeClassifier(
    max_depth=6,
    random_state=42
),

"Random Forest": RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
),

"Gradient Boosting": GradientBoostingClassifier(
    random_state=42
)
```

}

# ==================================

# OPTIONAL XGBOOST

# ==================================

try:
from xgboost import XGBClassifier

```
models["XGBoost"] = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
```

except Exception:
print("XGBoost not available")

# ==================================

# TRAIN TEST SPLIT

# ==================================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42,
stratify=y
)

# ==================================

# TRAIN MODELS

# ==================================

trained_models = {}

print("\nTraining Models...\n")

for name, clf in models.items():

```
print(f"Training {name}")

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", clf)
    ]
)

pipeline.fit(X_train, y_train)

try:
    probs = pipeline.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, probs)

    print(
        f"{name} ROC-AUC: {auc_score:.4f}"
    )

except Exception as e:
    print(
        f"{name} evaluation failed: {e}"
    )

trained_models[name] = pipeline
```

# ==================================

# SAVE MODELS

# ==================================

with open("models.pkl", "wb") as f:
pickle.dump(
trained_models,
f,
protocol=pickle.HIGHEST_PROTOCOL
)

print("\nmodels.pkl saved successfully")
