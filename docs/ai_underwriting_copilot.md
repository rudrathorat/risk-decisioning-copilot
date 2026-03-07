# AI Underwriting Copilot

The **AI Underwriting Copilot** turns pipeline outputs (risk score, decision, SHAP) into analyst-style credit reports with risk driver narratives, suggested conditions, and what-if simulation.

## Architecture

- **Module:** `src/underwriting_copilot.py`
- **Flow:** Data → Feature Engineering → Risk Model → Decision Engine → SHAP → **Copilot** → Credit Report (JSON + narratives)
- **Key components:** `UnderwritingCopilot.generate_report()`, `run_what_if()`, `get_portfolio_risk_insights()`

## Quick usage

```python
from src.underwriting_copilot import UnderwritingCopilot, get_portfolio_risk_insights

copilot = UnderwritingCopilot()
report = copilot.generate_report(applicant_raw, applicant_engineered, risk_score, risk_band, decision, top_risk_drivers=drivers)
# report.applicant_summary, .key_risk_drivers (with .narrative), .suggested_conditions, .what_if_results

insights = get_portfolio_risk_insights(portfolio_df, risk_score_col="risk_score", duration_col="duration", savings_col="savings")
# insights["segment_description"], ["default_rate"], ["recommendation"]
```

## Demo

Run the CLI demo:

```bash
python demo_copilot.py
```

Full pipeline (including copilot reports and portfolio insights):

```bash
python run.py
```

Reports are written to `results/copilot_reports/report_0.json`, etc. Each report includes `key_risk_drivers` with a `narrative` sentence per driver (e.g. *"Long loan duration increases exposure to default risk."*).
