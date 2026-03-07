"""
AI Underwriting Copilot — CLI demo.

Runs a minimal pipeline (load data, feature engineering, train model, one applicant)
and prints a sample credit report with risk drivers, suggested conditions, and
one what-if simulation. Use this to quickly show what the copilot produces.

Usage:
    python demo_copilot.py
"""

import sys
from pathlib import Path

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.decisioning import DecisionEngine
from src.explainability import ModelExplainer
from src.underwriting_copilot import UnderwritingCopilot, get_portfolio_risk_insights

from sklearn.model_selection import train_test_split


def main():
    print("Running AI underwriting copilot for sample applicant...\n")
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
    X_test_eng = fe.transform(X_test_raw)
    trainer = ModelTrainer()
    trainer.train_multiple_models(
        X_train_eng, y_train,
        model_list=["logistic_regression"],
    )
    best_model_name, best_model = trainer.select_best_model(X_val_eng, y_val, metric="roc_auc")
    decision_engine = DecisionEngine()
    i = 0
    applicant_raw = X_test_raw.iloc[i : i + 1]
    applicant_engineered = X_test_eng.iloc[i : i + 1]
    risk_score = float(trainer.predict(applicant_engineered, model_name=best_model_name)[0])
    decision_result = decision_engine.make_decision(risk_score)
    risk_band = decision_result["risk_band"]
    decision = decision_result["decision"]
    explainer = ModelExplainer(best_model, X_train_eng.columns.tolist())
    top_risk_drivers = None
    try:
        imp = explainer.get_feature_importance_from_model()
        top_risk_drivers = imp.head(10).set_index("feature")["importance"].to_dict()
    except Exception:
        pass
    copilot = UnderwritingCopilot()
    current_duration = int(applicant_raw["duration"].iloc[0])
    what_if_scenarios = [{"duration": max(6, current_duration - 12)}]
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
    print("Risk Score: {:.2f}".format(report.risk_score))
    print("Risk Band: {}".format(report.risk_band))
    print()
    print("Key Risk Drivers")
    for d in report.key_risk_drivers[:5]:
        name = d.get("feature", d.get("description", ""))
        narrative = d.get("narrative", d.get("description", name))
        print("  - {}".format(narrative))
    print()
    print("Suggested Conditions")
    for c in report.suggested_conditions:
        print("  - {}".format(c.lower()))
    if report.what_if_results:
        wi = report.what_if_results[0]
        print()
        print("What-if: {}".format(wi.get("scenario_label", "scenario")))
        print("  PD: {:.2f}".format(wi["new_pd"]))
        print("  Decision: {}".format(wi["new_decision"]))
    print()
    print("Done.")


if __name__ == "__main__":
    main()
