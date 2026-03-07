"""
AI Underwriting Copilot — turns model outputs into human-style credit reports.

This module consumes risk scores, decisions, and explainability outputs from the
credit risk pipeline and produces structured underwriting reports, including
optional what-if simulations.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


# -----------------------------------------------------------------------------
# Data structures for report content
# -----------------------------------------------------------------------------


@dataclass
class CopilotReport:
    """
    Structured output of the AI underwriting copilot for a single applicant.

    All fields are populated by generate_report(). Optional fields may be None
    when not requested or when pipeline artifacts are not provided.
    """

    applicant_summary: Dict[str, Any]
    """Key applicant attributes (e.g. credit_amount, duration, age, employment).
    Used for the 'Applicant Summary' section and for narrative context."""

    risk_score: float
    """Probability of default (PD) from the risk model."""

    risk_band: str
    """Risk band label (e.g. 'Low', 'Medium', 'High') from the decision engine."""

    key_risk_drivers: List[Dict[str, Any]]
    """Top factors driving this applicant's risk. Each item typically has
    'feature', 'contribution' (e.g. SHAP value), 'value', and optionally
    'direction' ('increases' / 'decreases')."""

    ai_interpretation: str
    """Short narrative (2–4 sentences) summarizing risk and rationale.
    Can be template-based or LLM-generated."""

    recommended_decision: str
    """Final decision: 'Approve', 'Reject', or 'Approve with Conditions'."""

    suggested_conditions: List[str]
    """Suggested risk-mitigation conditions (e.g. 'Reduce loan amount',
    'Shorten tenure'). May be empty."""

    suggested_terms: Optional[Dict[str, Any]] = None
    """Optional suggested loan terms, e.g. loan_amount, tenure_months,
    interest_rate_pct. Used when the copilot suggests modified terms;
    may be illustrative rather than from a pricing model."""

    what_if_results: Optional[List[Dict[str, Any]]] = None
    """Optional list of what-if scenario results. Each item typically has
    'scenario', 'overrides', 'original_pd', 'new_pd', 'new_risk_band',
    'new_decision'. Populated when pipeline_artifacts and scenarios are provided."""


# -----------------------------------------------------------------------------
# Underwriting Copilot
# -----------------------------------------------------------------------------


class UnderwritingCopilot:
    """
    Generates human-style underwriting reports and what-if simulations
    from pipeline outputs (risk score, decision, SHAP, etc.).
    """

    def generate_report(
        self,
        applicant_raw: Any,
        applicant_engineered: Any,
        risk_score: float,
        risk_band: str,
        decision: str,
        *,
        top_risk_drivers: Optional[List[Dict[str, Any]]] = None,
        pipeline_artifacts: Optional[Dict[str, Any]] = None,
        what_if_scenarios: Optional[List[Dict[str, Any]]] = None,
    ) -> CopilotReport:
        """
        Build a structured underwriting report for one applicant.

        Uses precomputed risk score, band, and decision; optionally
        incorporates SHAP-based risk drivers and runs what-if scenarios
        when pipeline_artifacts and what_if_scenarios are provided.

        Args:
            applicant_raw: One row of raw features (Series or 1-row DataFrame)
                with the same columns as used for FeatureEngineer.transform.
            applicant_engineered: Same applicant after feature engineering
                (for consistent column/context; may be used for explanations).
            risk_score: Probability of default from the model.
            risk_band: Risk band label from the decision engine.
            decision: Recommended decision from the decision engine.
            top_risk_drivers: Optional list of {feature, contribution, value, ...}
                e.g. from ModelExplainer.explain_prediction(...)['top_contributors'].
            pipeline_artifacts: Optional dict with keys such as feature_engineer,
                trainer, best_model_name, decision_engine (used for what-if).
            what_if_scenarios: Optional list of override dicts, e.g.
                [{"duration": 24, "credit_amount": 4000}]. Ignored if
                pipeline_artifacts is None.

        Returns:
            A CopilotReport instance with applicant_summary, risk_score,
            risk_band, key_risk_drivers, ai_interpretation, recommended_decision,
            suggested_conditions, and optionally suggested_terms and what_if_results.
        """
        applicant_summary = self._extract_applicant_summary(applicant_raw)
        drivers_formatted = self._format_risk_drivers(top_risk_drivers or [])
        ai_interpretation = self._build_ai_interpretation(
            risk_score, risk_band, decision, drivers_formatted
        )
        suggested_conditions = self._suggest_conditions(
            risk_band, decision, top_risk_drivers or []
        )

        what_if_results = None
        suggested_terms = None
        if pipeline_artifacts and what_if_scenarios:
            fe = pipeline_artifacts.get("feature_engineer")
            trainer = pipeline_artifacts.get("trainer")
            best_model_name = pipeline_artifacts.get("best_model_name")
            de = pipeline_artifacts.get("decision_engine")
            if fe is not None and trainer is not None and best_model_name and de is not None:
                what_if_results = self.run_what_if(
                    applicant_raw,
                    what_if_scenarios,
                    fe,
                    trainer,
                    best_model_name,
                    de,
                    original_pd=risk_score,
                )

        return CopilotReport(
            applicant_summary=applicant_summary,
            risk_score=risk_score,
            risk_band=risk_band,
            key_risk_drivers=drivers_formatted,
            ai_interpretation=ai_interpretation,
            recommended_decision=decision,
            suggested_conditions=suggested_conditions,
            suggested_terms=suggested_terms,
            what_if_results=what_if_results,
        )

    _SUMMARY_KEYS = (
        "credit_amount",
        "duration",
        "age",
        "employment",
        "savings",
        "purpose",
        "installment_rate",
        "housing",
        "job",
    )

    def _extract_applicant_summary(self, applicant_raw: Any) -> Dict[str, Any]:
        """
        Extract key applicant attributes for the Applicant Summary section.

        Pulls only fields that are useful for underwriting narrative (e.g. loan
        amount, duration, age). Handles both Series and 1-row DataFrame.
        """
        if applicant_raw is None:
            return {}
        if isinstance(applicant_raw, pd.Series):
            row = applicant_raw
        else:
            row = applicant_raw.iloc[0] if len(applicant_raw) else pd.Series()
        summary = {}
        for key in self._SUMMARY_KEYS:
            if key in row.index:
                val = row[key]
                summary[key] = val.item() if hasattr(val, "item") else val
        return summary

    def _format_risk_drivers(
        self, top_risk_drivers: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Turn raw SHAP/driver dicts into readable items with feature label and direction.

        Accepts top_risk_drivers as a dict (e.g. SHAP feature_contributions) or list
        of dicts with 'feature'/'name' and 'contribution'/'importance'. Maps
        engineered names (e.g. duration_woe) to human labels and sets direction
        from the sign of the contribution.
        """
        formatted = []
        if not top_risk_drivers:
            return formatted
        if isinstance(top_risk_drivers, dict):
            drivers_list = [
                {"feature": k, "contribution": v} for k, v in top_risk_drivers.items()
            ]
        else:
            drivers_list = list(top_risk_drivers)
        feature_labels = {
            "duration": "Loan duration",
            "credit_amount": "Credit amount",
            "savings": "Savings balance",
            "employment": "Employment history",
            "age": "Age",
            "installment_rate": "Installment rate",
            "credit_history": "Credit history",
            "purpose": "Loan purpose",
            "housing": "Housing",
            "job": "Job type",
            "property": "Property",
            "existing_credits": "Existing credits",
            "residence_since": "Residence stability",
        }
        for item in drivers_list:
            feat = item.get("feature", item.get("name", ""))
            contrib = item.get("contribution", item.get("importance", 0))
            if isinstance(contrib, (int, float)):
                direction = "increases" if contrib > 0 else "decreases"
            else:
                direction = "affects"
            base_name = feat.replace("_woe", "").split("_mul_")[0].split("_div_")[0]
            label = feature_labels.get(base_name, base_name.replace("_", " ").title())
            description = f"{label} ({direction} risk)"
            narrative = self._get_driver_narrative(base_name, direction, label)
            formatted.append({
                "feature": feat,
                "contribution": contrib,
                "direction": direction,
                "description": description,
                "narrative": narrative,
            })
        return formatted

    _DRIVER_NARRATIVES = {
        ("duration", "increases"): "Long loan duration increases exposure to default risk.",
        ("duration", "decreases"): "Short loan duration reduces exposure to default risk.",
        ("credit_amount", "increases"): "High credit amount relative to income increases repayment burden.",
        ("credit_amount", "decreases"): "Moderate credit amount reduces repayment burden.",
        ("savings", "increases"): "Low savings balance indicates limited financial cushion.",
        ("savings", "decreases"): "Strong savings balance indicates financial cushion.",
        ("employment", "increases"): "Short or unstable employment history increases income risk.",
        ("employment", "decreases"): "Stable employment history supports repayment capacity.",
        ("age", "increases"): "Age profile may affect repayment capacity and stability.",
        ("age", "decreases"): "Age profile supports stable repayment capacity.",
        ("installment_rate", "increases"): "High installment rate relative to income increases repayment burden.",
        ("installment_rate", "decreases"): "Moderate installment rate supports affordability.",
        ("credit_history", "increases"): "Adverse credit history indicates higher default risk.",
        ("credit_history", "decreases"): "Good credit history supports lower default risk.",
        ("purpose", "increases"): "Loan purpose may indicate higher risk profile.",
        ("purpose", "decreases"): "Loan purpose aligns with lower risk profile.",
        ("housing", "increases"): "Housing situation may affect financial stability.",
        ("housing", "decreases"): "Stable housing situation supports financial stability.",
        ("job", "increases"): "Job type and stability affect income reliability.",
        ("job", "decreases"): "Job type and stability support income reliability.",
        ("property", "increases"): "Limited property or collateral may increase risk.",
        ("property", "decreases"): "Property or collateral supports lower risk.",
        ("existing_credits", "increases"): "High number of existing credits increases debt burden.",
        ("existing_credits", "decreases"): "Moderate existing credits support manageable debt burden.",
        ("residence_since", "increases"): "Short time at current residence may indicate instability.",
        ("residence_since", "decreases"): "Long tenure at current residence indicates stability.",
    }

    def _get_driver_narrative(self, base_name: str, direction: str, label: str) -> str:
        """
        Return a one-sentence analyst-style narrative for a risk driver.

        Used so reports read like real credit analyst commentary (e.g. "Long loan
        duration increases exposure to default risk.") instead of raw feature names.
        """
        key = (base_name, direction)
        return self._DRIVER_NARRATIVES.get(
            key,
            f"{label} {direction} risk.",
        )

    def _build_ai_interpretation(
        self,
        risk_score: float,
        risk_band: str,
        decision: str,
        drivers_formatted: List[Dict[str, Any]],
    ) -> str:
        """
        Build a short narrative from rule-based templates (no LLM).

        Produces 2–4 sentences: risk level, PD, top driver descriptions, and
        recommended decision. Uses only the formatted key_risk_drivers passed in.
        """
        risk_level = risk_band.lower()
        driver_phrases = [d["description"] for d in drivers_formatted[:5]]
        if driver_phrases:
            driver_text = ", ".join(driver_phrases)
            factors = f" Key factors include: {driver_text}."
        else:
            factors = ""
        return (
            f"The applicant shows {risk_level} credit risk "
            f"(estimated default risk: {risk_score:.1%}).{factors} "
            f"Recommended decision: {decision}."
        )

    def _suggest_conditions(
        self,
        risk_band: str,
        decision: str,
        top_risk_drivers: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Suggest risk-mitigation conditions using simple rules from top drivers.

        Only applies to Medium/High/Very High risk bands. Rules: long duration →
        suggest shorter tenure; high credit amount → suggest lower amount;
        savings in drivers → evidence of savings; employment → employment
        verification; High/Very High with credit_history/credit_amount →
        collateral or guarantor. No LLM; purely rule-based.
        """
        conditions = []
        high_risk_bands = ("Medium", "High", "Very High")
        if risk_band not in high_risk_bands:
            return conditions

        driver_features = set()
        if isinstance(top_risk_drivers, dict):
            for k in top_risk_drivers:
                base = k.replace("_woe", "").split("_mul_")[0].split("_div_")[0]
                driver_features.add(base)
        else:
            for item in top_risk_drivers:
                feat = item.get("feature", item.get("name", ""))
                base = feat.replace("_woe", "").split("_mul_")[0].split("_div_")[0]
                driver_features.add(base)

        if "duration" in driver_features:
            conditions.append("Consider shortening loan duration")
        if "credit_amount" in driver_features:
            conditions.append("Consider reducing loan amount")
        if "savings" in driver_features:
            conditions.append("Consider requesting evidence of savings or reserves")
        if "employment" in driver_features:
            conditions.append("Consider verification of employment stability")
        if ("credit_history" in driver_features or "credit_amount" in driver_features) and risk_band in (
            "High",
            "Very High",
        ):
            conditions.append("Consider collateral or guarantor")

        return conditions

    def run_what_if(
        self,
        applicant_raw: Any,
        scenarios: List[Dict[str, Any]],
        feature_engineer: Any,
        trainer: Any,
        best_model_name: str,
        decision_engine: Any,
        *,
        original_pd: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run what-if simulations: apply scenario overrides, re-transform, rescore,
        and optionally re-decide.

        For each scenario, overrides are applied to a copy of applicant_raw (only
        for keys that exist in the raw data). The modified row is transformed
        with feature_engineer.transform(), then scored with trainer.predict(...).
        Optionally, the decision_engine is used to get new_risk_band and new_decision.

        Args:
            applicant_raw: One row of raw features (1-row DataFrame) with columns
                matching what feature_engineer was fitted on.
            scenarios: List of override dicts, e.g. [{"duration": 24}, {"credit_amount": 4000}].
            feature_engineer: Fitted FeatureEngineer instance.
            trainer: ModelTrainer instance with the selected model.
            best_model_name: Name of the model to use for prediction.
            decision_engine: DecisionEngine instance for new band/decision.
            original_pd: If provided, used as baseline PD in each result; otherwise
                computed once from applicant_raw.

        Returns:
            List of dicts, one per scenario. Each dict contains:
            original_pd, new_pd, new_risk_band, new_decision, scenario_label, overrides.
        """
        # Ensure 1-row DataFrame
        if isinstance(applicant_raw, pd.Series):
            raw_df = applicant_raw.to_frame().T
        else:
            raw_df = applicant_raw.copy()
        if len(raw_df) != 1:
            raise ValueError("applicant_raw must be a single row (Series or 1-row DataFrame)")

        # Baseline PD if not provided
        if original_pd is None:
            baseline_eng = feature_engineer.transform(raw_df)
            original_pd = float(trainer.predict(baseline_eng, model_name=best_model_name)[0])

        results = []
        for overrides in scenarios:
            # Clone and apply overrides (only for columns that exist)
            modified_raw = raw_df.copy()
            for key, value in overrides.items():
                if key in modified_raw.columns:
                    modified_raw.loc[modified_raw.index[0], key] = value

            # Transform and predict
            modified_eng = feature_engineer.transform(modified_raw)
            new_pd = float(trainer.predict(modified_eng, model_name=best_model_name)[0])

            # New decision from decision engine
            decision_result = decision_engine.make_decision(new_pd)
            new_risk_band = decision_result["risk_band"]
            new_decision = decision_result["decision"]

            # Human-readable scenario label
            scenario_label = ", ".join(f"{k}={v}" for k, v in overrides.items())

            results.append({
                "original_pd": original_pd,
                "new_pd": new_pd,
                "new_risk_band": new_risk_band,
                "new_decision": new_decision,
                "scenario_label": scenario_label,
                "overrides": overrides,
            })
        return results


# -----------------------------------------------------------------------------
# Portfolio risk insights (for risk managers)
# -----------------------------------------------------------------------------


def get_portfolio_risk_insights(
    portfolio_df: pd.DataFrame,
    risk_score_col: str = "risk_score",
    duration_col: str = "duration",
    savings_col: Optional[str] = "savings",
    duration_high_threshold: float = 36,
    savings_low_values: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Summarize portfolio risk by segment and return the highest-risk segment
    with a recommendation. Useful for risk managers to tighten policies.

    Segments by simple rules (e.g. long duration, low savings), computes
    expected default rate (mean risk_score) per segment, and returns the
    worst segment plus a recommendation.

    Args:
        portfolio_df: DataFrame with risk_score and feature columns (duration, savings, etc.).
        risk_score_col: Column name for estimated default risk.
        duration_col: Column name for loan duration (numeric, e.g. months).
        savings_col: Column name for savings (numeric or categorical). Optional.
        duration_high_threshold: Treat duration >= this as "long" (months).
        savings_low_values: If savings is categorical, list of values considered "low".

    Returns:
        Dict with segment_description, default_rate, count, recommendation.
    """
    if risk_score_col not in portfolio_df.columns or duration_col not in portfolio_df.columns:
        return {
            "segment_description": "N/A",
            "default_rate": 0.0,
            "count": 0,
            "recommendation": "Insufficient data for segment analysis.",
        }
    df = portfolio_df.copy()
    df["_duration_high"] = df[duration_col] >= duration_high_threshold
    if savings_col and savings_col in df.columns:
        if savings_low_values is not None:
            df["_savings_low"] = df[savings_col].isin(savings_low_values)
        else:
            # Assume numeric: low = bottom quartile
            q = df[savings_col].quantile(0.25)
            df["_savings_low"] = df[savings_col] <= q
        segment = df[df["_duration_high"] & df["_savings_low"]]
        segment_desc = f"Duration >= {duration_high_threshold} months AND low savings"
    else:
        segment = df[df["_duration_high"]]
        segment_desc = f"Duration >= {duration_high_threshold} months"
    if len(segment) == 0:
        return {
            "segment_description": segment_desc,
            "default_rate": 0.0,
            "count": 0,
            "recommendation": "No applicants in this segment; no action required.",
        }
    default_rate = float(segment[risk_score_col].mean())
    count = int(len(segment))
    recommendation = (
        "Tighten approval threshold for this segment."
        if default_rate > 0.35
        else "Monitor this segment; consider targeted conditions."
    )
    return {
        "segment_description": segment_desc,
        "default_rate": default_rate,
        "count": count,
        "recommendation": recommendation,
    }
