"""
Credit Risk Modeling and Decision Framework
Main script to run the enhanced credit risk model.
Can be executed directly in Cursor/VS Code or from terminal.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from src.feature_engineering import FeatureEngineer
from src.models import ModelTrainer
from src.evaluation import CreditRiskMetrics, print_evaluation_report
from src.monitoring import ModelMonitor
from src.decisioning import DecisionEngine, ProfitOptimizer, EarlyWarningSystem
from src.explainability import ModelExplainer
from src.underwriting_copilot import UnderwritingCopilot, get_portfolio_risk_insights

from sklearn.model_selection import train_test_split

import json

def main():
    print("=" * 60)
    print("Credit Risk Modeling and Decision Framework")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1/8] Loading data...")
    columns = [
        "status", "duration", "credit_history", "purpose", "credit_amount",
        "savings", "employment", "installment_rate", "personal_status",
        "other_debtors", "residence_since", "property", "age",
        "other_installments", "housing", "existing_credits",
        "job", "num_dependents", "telephone", "foreign_worker", "target"
    ]
    
    data_path = Path(__file__).parent / "data" / "german_credit.data"
    df = pd.read_csv(data_path, sep=" ", names=columns)
    df["default"] = (df["target"] == 2).astype(int)
    
    print(f"  Dataset: {df.shape[0]} samples, {df.shape[1]} features")
    print(f"  Default rate: {df['default'].mean():.2%}")
    
    # 2. Feature preparation
    print("\n[2/8] Preparing features...")
    feature_cols = [col for col in df.columns if col not in ['target', 'default']]
    X_all = df[feature_cols]
    y = df['default']
    
    # Calculate IV for feature selection
    fe = FeatureEngineer()
    iv_results = fe.calculate_iv_all(X_all, y)
    selected_features = iv_results[iv_results['iv'] >= 0.02]['feature'].tolist()
    X_features = X_all[selected_features] if selected_features else X_all
    
    print(f"  Selected {len(selected_features)} features (IV >= 0.02)")
    
    # 3. Split data
    print("\n[3/8] Splitting data...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.125, random_state=42, stratify=y_train
    )
    
    print(f"  Training: {X_train_raw.shape[0]} samples")
    print(f"  Validation: {X_val_raw.shape[0]} samples")
    print(f"  Test: {X_test_raw.shape[0]} samples")
    
    # 4. Feature engineering
    print("\n[4/8] Feature engineering...")
    X_train_eng = fe.fit_transform(X_train_raw, y_train, use_woe=True, use_interactions=True)
    X_val_eng = fe.transform(X_val_raw)
    X_test_eng = fe.transform(X_test_raw)
    
    print(f"  Engineered features: {X_train_eng.shape[1]}")
    
    # 5. Model training
    print("\n[5/8] Training models...")
    trainer = ModelTrainer()
    model_results = trainer.train_multiple_models(
        X_train_eng, y_train,
        model_list=["logistic_regression", "random_forest", "gradient_boosting"]
    )
    
    best_model_name, best_model = trainer.select_best_model(X_val_eng, y_val, metric="roc_auc")
    print(f"  Best model: {best_model_name}")
    print(f"  Validation ROC-AUC: {model_results[best_model_name]['cv_mean']:.4f}")
    
    # 6. Evaluation
    print("\n[6/8] Evaluating model...")
    y_pred_proba = trainer.predict(X_test_eng, model_name=best_model_name)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    print_evaluation_report(y_test.values, y_pred_proba, y_pred)
    
    metrics = CreditRiskMetrics.calculate_all_metrics(y_test.values, y_pred_proba, y_pred)
    print(f"\n  Performance Summary:")
    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"    Gini:    {metrics['gini']:.4f}")
    print(f"    KS:      {metrics['ks']:.4f}")
    
    # 7. Decisioning
    print("\n[7/8] Creating decisions...")
    decision_engine = DecisionEngine()
    decisions = decision_engine.batch_decisions(pd.Series(y_pred_proba, index=y_test.index))
    
    print(f"  Decision distribution:")
    print(decisions['decision'].value_counts().to_string())
    
    # 8. Monitoring setup
    print("\n[8/9] Setting up monitoring...")
    y_val_pred_proba = trainer.predict(X_val_eng, model_name=best_model_name)
    monitor = ModelMonitor(pd.Series(y_val_pred_proba))
    
    current_scores = pd.Series(y_pred_proba)
    monitoring_report = monitor.generate_monitoring_report(
        current_scores,
        current_features=X_test_eng,
        baseline_features=X_val_eng
    )
    
    print(f"  Monitoring status: {monitoring_report['overall_status']}")
    print(f"  PSI: {monitoring_report['score_drift']['psi']:.4f}")
    
    # 9. AI Underwriting Copilot
    print("\n[9/10] Generating AI underwriting reports...")
    copilot = UnderwritingCopilot()
    explainer = ModelExplainer(best_model, X_train_eng.columns.tolist())
    try:
        explainer.fit_shap_explainer(X_train_eng, explainer_type="auto")
    except Exception:
        pass
    copilot_reports_dir = Path(__file__).parent / "results" / "copilot_reports"
    copilot_reports_dir.mkdir(parents=True, exist_ok=True)
    sample_size = min(5, len(X_test_eng))
    for i in range(sample_size):
        applicant_raw = X_test_raw.iloc[i : i + 1]
        applicant_engineered = X_test_eng.iloc[i : i + 1]
        risk_score = float(y_pred_proba[i])
        risk_band = decisions.iloc[i]["risk_band"]
        decision = decisions.iloc[i]["decision"]
        top_risk_drivers = None
        try:
            explanation = explainer.explain_prediction(X_test_eng, idx=i)
            fc = explanation.get("feature_contributions", {})
            top_risk_drivers = dict(
                sorted(fc.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
            )
        except Exception:
            pass
        report = copilot.generate_report(
            applicant_raw=applicant_raw,
            applicant_engineered=applicant_engineered,
            risk_score=risk_score,
            risk_band=risk_band,
            decision=decision,
            top_risk_drivers=top_risk_drivers,
        )
        report_dict = {
            "applicant_summary": report.applicant_summary,
            "risk_score": report.risk_score,
            "risk_band": report.risk_band,
            "key_risk_drivers": report.key_risk_drivers,
            "ai_interpretation": report.ai_interpretation,
            "recommended_decision": report.recommended_decision,
            "suggested_conditions": report.suggested_conditions,
            "suggested_terms": report.suggested_terms,
            "what_if_results": report.what_if_results,
        }
        report_path = copilot_reports_dir / f"report_{i}.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        print(f"  ✓ Saved {report_path.name}")
    print(f"  ✓ Saved {sample_size} copilot reports to results/copilot_reports/")
    
    print("\n[10/10] Saving results...")
    
    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    # Save predictions and decisions
    results_df = pd.DataFrame({
        'risk_score': y_pred_proba,
        'prediction': y_pred,
        'actual': y_test.values,
        'risk_band': decisions['risk_band'].values,
        'decision': decisions['decision'].values
    })
    results_df.to_csv(output_dir / "predictions.csv", index=False)
    print(f"  ✓ Saved predictions to results/predictions.csv")
    
    # Save metrics summary
    metrics_summary = {
        'model': best_model_name,
        'roc_auc': float(metrics['roc_auc']),
        'gini': float(metrics['gini']),
        'ks': float(metrics['ks']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1']),
        'accuracy': float(metrics['accuracy'])
    }
    
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"  ✓ Saved metrics to results/metrics.json")
    
    # Save feature importance
    try:
        explainer = ModelExplainer(best_model, X_train_eng.columns.tolist())
        importance_df = explainer.get_feature_importance_from_model()
        importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
        print(f"  ✓ Saved feature importance to results/feature_importance.csv")
    except:
        pass
    
    # Save risk band analysis
    risk_band_metrics = CreditRiskMetrics.calculate_risk_band_metrics(
        results_df, 'risk_score', 'actual', n_bands=5
    )
    risk_band_metrics.to_csv(output_dir / "risk_bands.csv", index=False)
    print(f"  ✓ Saved risk band analysis to results/risk_bands.csv")
    
    # Portfolio risk insights (highest-default segment)
    portfolio_df = X_test_raw.copy()
    portfolio_df["risk_score"] = y_pred_proba
    portfolio_insights = get_portfolio_risk_insights(
        portfolio_df,
        risk_score_col="risk_score",
        duration_col="duration",
        savings_col="savings",
        duration_high_threshold=24,
        savings_low_values=["A61", "A62"],
    )
    
    # Create summary report
    report = f"""
Credit Risk Modeling and Decision Framework - Analysis Report
{'=' * 60}

MODEL PERFORMANCE
{'-' * 60}
Best Model: {best_model_name}
ROC-AUC: {metrics['roc_auc']:.4f}
Gini Coefficient: {metrics['gini']:.4f}
KS Statistic: {metrics['ks']:.4f}

CLASSIFICATION METRICS
{'-' * 60}
Precision: {metrics['precision']:.4f}
Recall: {metrics['recall']:.4f}
F1-Score: {metrics['f1']:.4f}
Accuracy: {metrics['accuracy']:.4f}

DECISION DISTRIBUTION
{'-' * 60}
{decisions['decision'].value_counts().to_string()}

RISK BAND ANALYSIS
{'-' * 60}
{risk_band_metrics.to_string(index=False)}

MONITORING
{'-' * 60}
Status: {monitoring_report['overall_status']}
PSI: {monitoring_report['score_drift']['psi']:.4f}

PORTFOLIO RISK INSIGHTS
{'-' * 60}
Highest default segment: {portfolio_insights['segment_description']}
Expected default rate: {portfolio_insights['default_rate']:.1%}
Applicants in segment: {portfolio_insights['count']}
Recommendation: {portfolio_insights['recommendation']}

FILES GENERATED
{'-' * 60}
- results/predictions.csv
- results/metrics.json
- results/risk_bands.csv
- results/feature_importance.csv
- results/copilot_reports/
- results/report.txt
"""
    
    with open(output_dir / "report.txt", 'w') as f:
        f.write(report)
    print(f"  ✓ Saved summary report to results/report.txt")
    
    print("\n" + "=" * 60)
    print("✓ Analysis complete!")
    print(f"✓ Results saved to: {output_dir}/")
    print("=" * 60)
    
    return {
        'model': best_model,
        'metrics': metrics,
        'decisions': decisions,
        'monitoring_report': monitoring_report,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    results = main()

