# ================================
# ⚡ IMPORTS
# ================================
import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import shap
from sklearn.metrics import roc_curve, auc

# ================================
# ⚡ CACHE (STABLE LOADING)
# ================================
@st.cache_data
def load_data():
    return pd.read_csv("European_Bank.csv")

@st.cache_resource
def load_models():
    with open("models.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()
models = load_models()

# ================================
# CONFIG (FIRST STREAMLIT CALL)
# ================================
st.set_page_config(layout="wide")

# ================================
# 🏦 HEADER
# ================================
st.title("  AI-Powered Customer Churn Intelligence Dashboard")

st.caption(
    "Multi-Model ML System | SHAP Explainability | ROC-AUC | Business Intelligence"
)

# ================================
# IMAGE (SAFE)
# ================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(BASE_DIR, "assets", "bank.jpg")

if os.path.exists(image_path):
    st.image(image_path, use_container_width=True)
else:
    st.error(f"Image NOT found at: {image_path}")

# ================================
# MODEL SELECTION (FIXED - ONLY ONCE)
# ================================
model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

model = models[model_name]

# ================================
# DYNAMIC MODEL INFO
# ================================
model_info = {
    "Logistic Regression": "Baseline model (interpretable)",
    "Decision Tree": "Rule-based model",
    "Random Forest": "Ensemble model",
    "Gradient Boosting": "Boosted trees",
    "XGBoost": "High-performance boosting"
}

st.sidebar.markdown("### Model Info")
st.sidebar.info(model_info.get(model_name, "No info available"))

# ================================
# LIVE MODEL DISPLAY (FIXED)
# ================================
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Active Model", model_name)
col2.metric("Type", "Ensemble")
col3.metric("Explainability", "SHAP")
col4.metric("Metric", "ROC-AUC")
col5.metric("System", "Fintech AI")

# ================================
# FEATURE FUNCTION
# ================================
def get_feature_names(model):
    preprocessor = model.named_steps["preprocessor"]

    num_features = list(preprocessor.transformers_[0][2])
    cat_features = list(preprocessor.transformers_[1][2])

    ohe = preprocessor.named_transformers_["cat"]
    cat_names = list(ohe.get_feature_names_out(cat_features))

    return num_features + cat_names

# ================================
# INPUT SECTION
# ================================
st.sidebar.header("Customer Input")

threshold = st.sidebar.slider(
    "Churn Threshold",
    0.1, 0.9, 0.5, 0.05
)

input_df = pd.DataFrame([{
    "CreditScore": float(st.sidebar.number_input("CreditScore", 600)),
    "Geography": st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"]),
    "Gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
    "Age": float(st.sidebar.number_input("Age", 30)),
    "Tenure": float(st.sidebar.number_input("Tenure", 2)),
    "Balance": float(st.sidebar.number_input("Balance", 8000)),
    "NumOfProducts": float(st.sidebar.number_input("NumOfProducts", 1)),
    "HasCrCard": str(st.sidebar.selectbox("HasCrCard", [0, 1])),
    "IsActiveMember": str(st.sidebar.selectbox("IsActiveMember", [0, 1])),
    "EstimatedSalary": float(st.sidebar.number_input("EstimatedSalary", 50000))
}])

# ================================
# TABS
# ================================
tab1, tab2, tab3 = st.tabs([
    "📊 Prediction",
    "🔍 SHAP Analysis",
    "📈 ROC Curve"
])

# =========================================================
# TAB 1 - PREDICTION
# =========================================================
with tab1:
    st.subheader("Customer Prediction")

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    churn_prob = prob[1]
    retain_prob = prob[0]

    # ================================
    # THRESHOLD-BASED DECISION
    # ================================
    st.info(f"Current Threshold: {threshold}")

    if churn_prob >= threshold:
        st.error("🔴 High Churn Risk (Based on Threshold)")
    else:
        st.success("🟢 Low Churn Risk (Based on Threshold)")

    # METRICS
    col1, col2 = st.columns(2)
    col1.metric("Churn Probability", f"{churn_prob:.2%}")
    col2.metric("Retain Probability", f"{retain_prob:.2%}")

    st.info(f"Model in use: {model_name}")

    # BAR CHART
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Retain", "Churn"],
        y=[retain_prob, churn_prob],
        marker_color=["green", "red"]
    ))

    fig.update_layout(title="Prediction Confidence")
    st.plotly_chart(fig, use_container_width=True)

    # ✅ CAPTION
    st.caption("Shows probability of customer churn vs retention.")

# =========================================================
# TAB 2 - SHAP ANALYSIS (CLEAN & STABLE)
# =========================================================
with tab2:
    st.subheader("SHAP Explainability")

    try:
        X_transformed = model[:-1].transform(input_df)
        ml_model = model.named_steps["model"]

        explainer = shap.TreeExplainer(ml_model)
        shap_values = explainer(X_transformed)

        values = shap_values.values[0].reshape(-1)
        feature_names = get_feature_names(model)

        min_len = min(len(values), len(feature_names))
        values = values[:min_len]
        feature_names = feature_names[:min_len]

        df_shap = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Impact": values
        })

        st.dataframe(df_shap)

        fig = go.Figure()
        colors = np.where(values < 0, "red", "green")

        fig.add_trace(go.Bar(
            x=values,
            y=feature_names,
            orientation="h",
            marker_color=colors
        ))

        fig.update_layout(title="SHAP Feature Impact")

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"shap_chart_{model_name}"
        )

        # ✅ CAPTION
        st.caption("Explains how each feature influences this prediction.")

    except Exception as e:
        st.info("Using coefficient-based explainability (fallback)")

        ml_model = model.named_steps["model"]
        feature_names = get_feature_names(model)

        if hasattr(ml_model, "coef_"):
            importances = np.abs(ml_model.coef_[0])
        elif hasattr(ml_model, "feature_importances_"):
            importances = ml_model.feature_importances_
        else:
            st.error("No explainability available")
            st.stop()

        min_len = min(len(importances), len(feature_names))

        df_imp = pd.DataFrame({
            "Feature": feature_names[:min_len],
            "Impact": importances[:min_len]
        }).sort_values("Impact")

        st.dataframe(df_imp)

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=df_imp["Impact"],
            y=df_imp["Feature"],
            orientation="h"
        ))

        fig.update_layout(title="Feature Importance")

        st.plotly_chart(
            fig,
            use_container_width=True,
            key=f"importance_chart_{model_name}"
        )

        # ================================
        # TOP DRIVERS INSIGHT
        # ================================
        st.subheader(" Top Drivers of Churn")

        try:
            impact_values = values
            feature_list = feature_names
        except:
            impact_values = importances[:min_len]
            feature_list = feature_names[:min_len]

        df_driver = pd.DataFrame({
            "Feature": feature_list,
            "Impact": impact_values
        })

        df_driver["AbsImpact"] = np.abs(df_driver["Impact"])
        df_driver = df_driver.sort_values("AbsImpact", ascending=False)

        top3 = df_driver.head(3)

        for i, row in top3.iterrows():
            direction = "🔴 increases churn" if row["Impact"] > 0 else "🟢 reduces churn"
            st.write(f"**{row['Feature']}** → {direction}")

        st.caption("Top factors driving this customer's churn risk.")

        # ================================
        # PDP
        # ================================
        st.subheader(" PDP (Interactive)")

        try:
            df = load_data()

            target = "Exited"
            X_full = df.drop(columns=[target, "Year"], errors="ignore")

            for col in ["Geography", "Gender", "HasCrCard", "IsActiveMember"]:
                X_full[col] = X_full[col].astype(str)

            feature = st.selectbox(
                "Select Feature for PDP",
                ["Age", "Balance", "EstimatedSalary"],
                key=f"pdp_feature_{model_name}"
            )

            values_grid = np.linspace(
                X_full[feature].min(),
                X_full[feature].max(),
                30
            )

            pdp_values = []

            for val in values_grid:
                X_temp = X_full.copy()
                X_temp[feature] = val

                preds = model.predict_proba(X_temp)[:, 1]
                pdp_values.append(np.mean(preds))

            fig_pdp = go.Figure()

            fig_pdp.add_trace(go.Scatter(
                x=values_grid,
                y=pdp_values,
                mode="lines+markers"
            ))

            fig_pdp.update_layout(
                title=f"PDP - {feature}",
                height=350
            )

            st.plotly_chart(
                fig_pdp,
                use_container_width=True,
                key=f"pdp_plot_{feature}_{model_name}"
            )

            st.caption("Shows how a feature affects churn probability across all customers.")

        except Exception as e:
            st.warning(f"PDP not available: {e}")

# =========================================================
# TAB 3 - ROC
# =========================================================
with tab3:
    st.subheader("ROC Curve (Selected Model)")

    df = load_data()

    target = "Exited"
    X = df.drop(columns=[target, "Year"], errors="ignore")
    y = df[target]

    for col in ["Geography", "Gender", "HasCrCard", "IsActiveMember"]:
        X[col] = X[col].astype(str)

    try:
        y_prob = model.predict_proba(X)[:, 1]
        y_prob = np.nan_to_num(y_prob)

        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"AUC = {roc_auc:.2f}"
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Random"
        ))

        fig.update_layout(
            title=f"ROC Curve - {model_name}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("Measures how well the model distinguishes churn vs non-churn.")

    except Exception as e:
        st.error(f"ROC Error: {e}")
        st.stop()

    st.subheader(" Model Evaluation Metrics")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.2f}")
    col2.metric("Precision", f"{prec:.2f}")
    col3.metric("Recall", f"{rec:.2f}")
    col4.metric("F1 Score", f"{f1:.2f}")

    st.metric("ROC-AUC", f"{roc_auc:.2f}")

    st.caption("Evaluation metrics based on selected threshold.")

    st.subheader("🎯 Probability Calibration Check")

    try:
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash"),
            name="Perfect Calibration"
        ))

        fig.add_trace(go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name="Model Calibration"
        ))

        fig.update_layout(
            title="Reliability Curve"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption("Shows how well predicted probabilities match actual outcomes.")

    except Exception as e:
        st.warning(f"Calibration plot not available: {e}")