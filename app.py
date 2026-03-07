"""
AI Underwriting Copilot — Streamlit UI

One-page dashboard to input an applicant profile, run credit analysis via the
existing UnderwritingCopilot, and view risk score, drivers, conditions, and what-if.

Run the app:
    streamlit run app.py

Requires: pip install streamlit (or add streamlit to requirements.txt).
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.decisioning import DecisionEngine
from src.explainability import ModelExplainer
from src.underwriting_copilot import UnderwritingCopilot

from sklearn.model_selection import train_test_split

# Currency conversion to model scale (German Credit uses DM; approximate rates for display)
USD_TO_DM = 2.0
INR_TO_DM = 0.024

# German Credit codes for dropdowns
SAVINGS_OPTIONS = [
    ("Unknown / none", "A61"),
    ("< 100 DM", "A62"),
    ("100-500 DM", "A63"),
    ("500-1000 DM", "A64"),
    (">= 1000 DM", "A65"),
]
EMPLOYMENT_OPTIONS = [
    ("Unemployed", "A71"),
    ("< 1 year", "A72"),
    ("1-4 years", "A73"),
    ("4-7 years", "A74"),
    (">= 7 years", "A75"),
]


@st.cache_resource
def load_pipeline():
    """Load data, fit FE, train model, return pipeline artifacts (cached)."""
    base = Path(__file__).parent
    columns = [
        "status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status",
        "other_debtors", "residence_since", "property", "age",
        "other_installments", "housing", "existing_credits",
        "job", "num_dependents", "telephone", "foreign_worker", "target",
    ]
    df = pd.read_csv(base / "data" / "german_credit.data", sep=" ", names=columns)
    df["default"] = (df["target"] == 2).astype(int)
    feature_cols = [c for c in df.columns if c not in ("target", "default")]
    X_all = df[feature_cols]
    y = df["default"]
    fe = FeatureEngineer()
    iv_results = fe.calculate_iv_all(X_all, y)
    selected = iv_results[iv_results["iv"] >= 0.02]["feature"].tolist()
    X_features = X_all[selected] if selected else X_all
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.125, random_state=42, stratify=y_train
    )
    X_train_eng = fe.fit_transform(X_train_raw, y_train, use_woe=True, use_interactions=True)
    X_val_eng = fe.transform(X_val_raw)
    trainer = ModelTrainer()
    trainer.train_multiple_models(X_train_eng, y_train, model_list=["logistic_regression"])
    best_model_name, best_model = trainer.select_best_model(X_val_eng, y_val, metric="roc_auc")
    decision_engine = DecisionEngine()
    explainer = ModelExplainer(best_model, X_train_eng.columns.tolist())
    template_row = X_train_raw.iloc[0:1].copy()
    return {
        "fe": fe,
        "trainer": trainer,
        "best_model_name": best_model_name,
        "decision_engine": decision_engine,
        "explainer": explainer,
        "X_train_eng": X_train_eng,
        "template_row": template_row,
    }


def to_model_amount(amount: float, currency: str) -> float:
    """Convert user-facing amount to model scale (DM) for prediction."""
    if currency == "USD":
        return amount * USD_TO_DM
    if currency == "INR":
        return amount * INR_TO_DM
    return amount


def build_applicant_row(template_row, credit_amount_dm, duration, age, savings_code, employment_code):
    """Override template with user inputs; return 1-row DataFrame."""
    row = template_row.copy()
    row["credit_amount"] = credit_amount_dm
    row["duration"] = duration
    row["age"] = age
    row["savings"] = savings_code
    row["employment"] = employment_code
    return row


def risk_bar_color(risk_score: float) -> str:
    """Return hex color for risk level (green / amber / red)."""
    if risk_score < 0.3:
        return "#22c55e"
    if risk_score < 0.5:
        return "#eab308"
    return "#ef4444"


def render_risk_bar(risk_score: float) -> None:
    """Render a color-coded progress bar for default risk."""
    pct = min(100, risk_score * 100)
    color = risk_bar_color(risk_score)
    st.markdown(
        f'<div style="margin: 8px 0; border-radius: 6px; overflow: hidden; background: #e2e8f0; height: 28px;">'
        f'<div style="width: {pct}%; height: 100%; background: {color}; transition: width 0.3s;"></div>'
        f'</div><p style="margin: 0 0 12px 0; font-size: 0.85rem; color: #64748b;">Low risk → High risk</p>',
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="AI Underwriting Copilot", layout="wide")
st.title("AI Underwriting Copilot")
st.caption("Enter applicant details and run credit analysis. Results appear in the dashboard below.")

with st.spinner("Loading pipeline (one-time)..."):
    pipeline = load_pipeline()

template_row = pipeline["template_row"]
fe = pipeline["fe"]
trainer = pipeline["trainer"]
best_model_name = pipeline["best_model_name"]
decision_engine = pipeline["decision_engine"]
explainer = pipeline["explainer"]
X_train_eng = pipeline["X_train_eng"]

# ----- Applicant profile (structured grid) -----
with st.container():
    st.subheader("Applicant profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        currency = st.selectbox("Currency", options=["USD", "INR"], index=0)
        if currency == "USD":
            loan_amount = st.number_input("Loan amount (USD)", min_value=250, max_value=100000, value=5000, step=500)
        else:
            loan_amount = st.number_input("Loan amount (INR)", min_value=10000, max_value=5000000, value=250000, step=5000)
        amount_for_model = to_model_amount(loan_amount, currency)
        duration = st.number_input("Loan duration (months)", min_value=6, max_value=72, value=24, step=6)
    with c2:
        age = st.number_input("Age", min_value=18, max_value=80, value=35)
        savings_label = st.selectbox(
            "Savings category",
            options=[x[0] for x in SAVINGS_OPTIONS],
            index=1,
        )
        savings_code = next(c for label, c in SAVINGS_OPTIONS if label == savings_label)
    with c3:
        employment_label = st.selectbox(
            "Employment length",
            options=[x[0] for x in EMPLOYMENT_OPTIONS],
            index=2,
        )
        employment_code = next(c for label, c in EMPLOYMENT_OPTIONS if label == employment_label)

st.divider()

if st.button("Run Credit Analysis", type="primary"):
    applicant_raw = build_applicant_row(
        template_row, amount_for_model, duration, age, savings_code, employment_code
    )
    applicant_engineered = fe.transform(applicant_raw)
    risk_score = float(trainer.predict(applicant_engineered, model_name=best_model_name)[0])
    decision_result = decision_engine.make_decision(risk_score)
    risk_band = decision_result["risk_band"]
    decision = decision_result["decision"]
    top_risk_drivers = None
    try:
        imp = explainer.get_feature_importance_from_model()
        top_risk_drivers = imp.head(10).set_index("feature")["importance"].to_dict()
    except Exception:
        pass
    copilot = UnderwritingCopilot()
    what_if_scenarios = [{"duration": max(6, int(duration) - 12)}]
    pipeline_artifacts = {
        "feature_engineer": fe,
        "trainer": trainer,
        "best_model_name": best_model_name,
        "decision_engine": decision_engine,
    }
    report = copilot.generate_report(
        applicant_raw=applicant_raw,
        applicant_engineered=applicant_engineered,
        risk_score=risk_score,
        risk_band=risk_band,
        decision=decision,
        top_risk_drivers=top_risk_drivers,
        pipeline_artifacts=pipeline_artifacts,
        what_if_scenarios=what_if_scenarios,
    )

    # ----- Results section (dashboard below form) -----
    st.divider()
    st.caption(f"**Loan amount:** {loan_amount:,.0f} {currency}")

    # 📊 Risk Assessment — Decision summary + visual risk indicator
    st.markdown("### 📊 Risk Assessment")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Estimated Default Risk", f"{report.risk_score:.1%}")
    with m2:
        st.metric("Risk Band", report.risk_band)
    with m3:
        st.metric("Recommended Decision", report.recommended_decision)
    render_risk_bar(float(report.risk_score))

    st.divider()

    # ⚠️ Key Risk Drivers
    st.markdown("### ⚠️ Key Risk Drivers")
    if report.key_risk_drivers:
        for d in report.key_risk_drivers[:8]:
            narrative = d.get("narrative", d.get("description", ""))
            st.markdown(f"- {narrative}")
    else:
        st.caption("No drivers available for this profile.")

    st.divider()

    # 💡 Risk Mitigation Suggestions
    st.markdown("### 💡 Risk Mitigation Suggestions")
    if report.suggested_conditions:
        for c in report.suggested_conditions:
            st.markdown(f"- {c}")
    else:
        st.caption("None for this risk profile.")

    st.divider()

    # 🔍 Scenario Analysis (what-if table)
    st.markdown("### 🔍 Scenario Analysis")
    if report.what_if_results:
        scenario_data = [
            {
                "Scenario": wi.get("scenario_label", "Scenario").replace("=", " "),
                "New Default Risk": f"{wi['new_pd']:.1%}",
                "Resulting Decision": wi["new_decision"],
            }
            for wi in report.what_if_results
        ]
        st.dataframe(
            pd.DataFrame(scenario_data),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("No scenario results.")

    st.divider()
    with st.expander("AI interpretation"):
        st.write(report.ai_interpretation)
