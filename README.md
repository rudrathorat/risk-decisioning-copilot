# AI Credit Risk Platform

Production-ready credit risk assessment system with **AI-assisted underwriting**: industry-standard modeling, explainable AI, decision engine, scenario simulation, and automated credit reports.

## Overview

End-to-end credit risk framework: feature engineering (WOE/IV), multi-algorithm training, evaluation (ROC-AUC, KS, Gini, PSI), SHAP explainability, monitoring, and an **AI Underwriting Copilot** that turns model outputs into analyst-style reports with risk driver narratives, suggested conditions, and what-if simulation.

**Performance**: ROC-AUC improved from 0.683 to 0.78 (14% improvement) through advanced feature engineering and model optimization.

## Features

- **Feature Engineering**: WOE transformation, IV calculation, optimal binning, feature interactions
- **Model Development**: Logistic Regression, Random Forest, Gradient Boosting with cross-validation
- **Evaluation**: ROC-AUC, KS statistic, Gini coefficient, PSI, risk band analysis
- **Explainability**: SHAP values for feature importance and prediction explanations
- **Monitoring**: Score drift detection (PSI), feature distribution tracking, automated alerting
- **Decisioning**: Risk-based decision engine, profit optimization, early warning system
- **AI Underwriting Copilot**: Risk driver narratives, suggested conditions, what-if simulation, automated credit reports
- **Portfolio risk insights**: Highest-default segment analysis and recommendations for risk managers

## AI Underwriting Copilot Example

Recruiters and engineers can see at a glance what the system produces.

**Input applicant (example):**

| Loan Amount | Duration | Savings | Employment   |
|-------------|----------|---------|--------------|
| 5000        | 36 months| Low     | &lt;1 year   |

**Output report:**

**Credit Risk Assessment**

- **Probability of Default:** 0.32  
- **Risk Band:** Medium  

**Key Risk Drivers**
- Long loan duration increases exposure to default risk.
- Low savings balance indicates limited financial cushion.
- Short or unstable employment history increases income risk.

**Recommended Decision**  
Approve with Conditions  

**Suggested Conditions**
- Consider shortening loan duration  
- Consider reducing loan amount  
- Consider verification of employment stability  

**What-if Simulation**  
Reducing duration from 36 → 24 months  
- **New PD:** 0.24  
- **Decision:** Approve  

---

## System Architecture

```
                    ┌───────────────────────┐
                    │ AI Underwriting Copilot │
                    │ Risk explanation       │
                    │ What-if simulation     │
                    │ Credit report          │
                    └───────────┬────────────┘
                                │
                     SHAP Explainability
                                │
                     Decision Engine
                                │
                        Risk Model
                                │
                     Feature Engineering
                                │
                             Data
```

---

## Quick Start

**Option 1: Run full pipeline (recommended for complete results)**
```bash
pip install -r requirements.txt
python run.py
```
Generates all results in `results/` (predictions, metrics, risk bands, **copilot_reports/**).

**Option 2: Demo the copilot only (fast product-style demo)**
```bash
pip install -r requirements.txt
python demo_copilot.py
```
Runs a minimal pipeline for one applicant and prints a sample report with risk drivers, conditions, and a what-if result.

**Option 3: Run enhanced notebook (interactive with visualizations)**
```bash
jupyter lab notebooks/02_enhanced_credit_risk_model.ipynb
```
Complete workflow, ROC curves, risk bands, and model explanations.

See [QUICKSTART.md](QUICKSTART.md) for detailed setup instructions.

## Project Structure

```
├── notebooks/
│   ├── 01_underwriting_model.ipynb         # Basic implementation
│   └── 02_enhanced_credit_risk_model.ipynb  # ⭐ Enhanced model (main showcase)
├── src/                                     # Production-ready modules
│   ├── feature_engineering.py              # WOE, IV, binning
│   ├── models.py                           # Model training & comparison
│   ├── evaluation.py                       # Metrics & evaluation
│   ├── monitoring.py                       # Production monitoring
│   ├── decisioning.py                      # Decision engine
│   ├── explainability.py                   # SHAP explanations
│   └── underwriting_copilot.py             # AI copilot, what-if, portfolio insights
├── run.py                                  # Full pipeline (model → copilot → results)
├── demo_copilot.py                         # CLI demo: one applicant, one report
├── data/
│   └── german_credit.data                  # German Credit Dataset
├── results/                                # Generated results (created on run)
│   ├── predictions.csv
│   ├── metrics.json
│   ├── feature_importance.csv
│   ├── risk_bands.csv
│   ├── report.txt
│   └── copilot_reports/                    # AI underwriting reports (JSON)
│       ├── report_0.json
│       └── ...
├── docs/
│   └── ai_underwriting_copilot.md          # Copilot architecture & usage
├── requirements.txt
└── README.md
```

## Usage

### Feature Engineering
```python
from src.feature_engineering import FeatureEngineer

fe = FeatureEngineer()
X_engineered = fe.fit_transform(X_train, y_train, use_woe=True)
```

### Model Training
```python
from src.models import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_multiple_models(X_train, y_train)
best_model = trainer.select_best_model(X_val, y_val)
```

### Evaluation & Monitoring
```python
from src.evaluation import CreditRiskMetrics
from src.monitoring import ModelMonitor

metrics = CreditRiskMetrics.calculate_all_metrics(y_true, y_pred_proba)
monitor = ModelMonitor(baseline_scores)
report = monitor.generate_monitoring_report(current_scores)
```

### AI Underwriting Copilot & Portfolio Insights
```python
from src.underwriting_copilot import UnderwritingCopilot, get_portfolio_risk_insights

copilot = UnderwritingCopilot()
report = copilot.generate_report(applicant_raw, applicant_engineered, risk_score, risk_band, decision, top_risk_drivers=drivers)

# Portfolio risk: highest-default segment and recommendation
insights = get_portfolio_risk_insights(portfolio_df, risk_score_col="risk_score", duration_col="duration", savings_col="savings")
# insights["segment_description"], ["default_rate"], ["recommendation"]
```

## Results

| Metric | Baseline | Enhanced | Improvement |
|--------|----------|----------|-------------|
| ROC-AUC | 0.683 | 0.78 | +14% |
| Gini | 0.37 | 0.56 | +51% |
| KS Statistic | - | 0.44 | Strong separation |
| Features | 5 numeric | All features with WOE | Comprehensive |

**Key Achievements:**
- Achieved 0.78 ROC-AUC through advanced feature engineering
- Implemented comprehensive model evaluation (KS, Gini, PSI)
- Production-ready monitoring and explainability framework

## Technologies

- Python 3.8+
- scikit-learn, pandas, numpy
- XGBoost (optional), SHAP (optional)
- matplotlib, seaborn (for visualizations)

## Portfolio Risk Insights

For risk managers, the copilot module includes a simple **portfolio risk insight** function. It segments the portfolio (e.g. by duration and savings), identifies the highest expected-default segment, and suggests an action.

**Example output:**

| Segment | Default Rate | Recommendation |
|---------|--------------|----------------|
| Duration ≥ 36 months AND low savings | 48% | Tighten approval threshold for this segment. |

Use `get_portfolio_risk_insights(portfolio_df)` after running the pipeline; pass a DataFrame that includes `risk_score` and feature columns such as `duration` and `savings`.

---

## Best Practices

- Data leakage prevention through proper train/validation/test splits
- IV-based feature selection (IV > 0.02 threshold)
- Cross-validation with stratified splits
- Industry-standard credit risk metrics (KS, Gini, PSI)
- Production-ready monitoring and explainability

---

## What This Project Demonstrates

This repo is built to showcase **fintech-ready** skills:

- **Credit risk modeling** — WOE/IV feature engineering, multi-model comparison, risk bands  
- **Explainable AI** — SHAP, risk driver narratives, analyst-style reports  
- **Decision systems** — Risk bands, approve/reject/conditions, policy rules  
- **Monitoring** — PSI, score drift, alerting  
- **Product thinking** — AI underwriting copilot, what-if simulation, portfolio insights  

That combination is rare in ML portfolios and aligns with what companies like **PayPal**, **Groww**, **Pine Labs**, and **JPMorgan Chase** look for: someone who understands fintech systems, not just ML models.
