"""SHAP helpers for the employee attrition API."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd


def risk_level(probability: float) -> str:
    if probability < 0.30:
        return "LOW"
    if probability < 0.60:
        return "MEDIUM"
    if probability < 0.80:
        return "HIGH"
    return "CRITICAL"


def risk_description(level: str) -> str:
    return {
        "LOW": "Low risk - employee likely to stay",
        "MEDIUM": "Moderate risk - monitor closely",
        "HIGH": "High risk - intervention should be considered",
        "CRITICAL": "Critical risk - urgent review recommended",
    }[level]


def _base_feature_name(encoded_name: str, raw_columns: list[str]) -> str:
    # ColumnTransformer with names like num__Age / cat__Department_Sales.
    s = encoded_name
    if "__" in s:
        s = s.split("__", 1)[1]

    # Longest-first prevents Department being matched before Department_Code etc.
    for col in sorted(raw_columns, key=len, reverse=True):
        if s == col or s.startswith(col + "_"):
            return col
    return s


def normalize_shap_values(values, expected_class: int = 1) -> np.ndarray:
    """Normalize SHAP output across SHAP versions/model types to 1-D."""
    arr = np.asarray(values)

    # Newer SHAP classifiers can return (n, features, classes).
    if arr.ndim == 3:
        arr = arr[0, :, expected_class]
    elif arr.ndim == 2:
        # For a single row this is (1, features).
        arr = arr[0]
    elif arr.ndim != 1:
        arr = arr.reshape(-1)

    return arr.astype(float)


def _value_text(value) -> str:
    if pd.isna(value):
        return "missing"
    if isinstance(value, (float, np.floating)):
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):.2f}"
    return str(value)


def _label(feature: str) -> str:
    return feature.replace("_", " ").strip().title()


def human_reason(feature: str, value, shap_value: float) -> str:
    direction = "increases" if shap_value > 0 else "decreases"
    label = _label(feature)
    v = _value_text(value)

    special = {
        "Years_Since_Last_Promotion": f"{label}: {v} years",
        "Years_In_Current_Role": f"{label}: {v} years",
        "Years_At_Company": f"{label}: {v} years",
        "Overtime_Hours_30D": f"{label}: {v} hours in last 30 days",
        "Monthly_Income": f"{label}: {v}",
        "Percent_Salary_Hike": f"{label}: {v}%",
        "Engagement_Score": f"{label}: {v}/5",
        "Job_Satisfaction": f"{label}: {v}/5",
        "Manager_Satisfaction": f"{label}: {v}/5",
        "Promotion_Gap_Flag": f"{label}: {'Yes' if str(v) == '1' else 'No'}",
        "Role_Stagnation_Flag": f"{label}: {'Yes' if str(v) == '1' else 'No'}",
    }
    detail = special.get(feature, f"{label}: {v}")
    return f"{detail} — {direction} predicted attrition risk"


def build_explanation(
    shap_values,
    feature_names: list[str],
    raw_row: pd.Series,
    raw_columns: list[str],
    top_n: int = 10,
):
    rows = []
    for name, value in zip(feature_names, shap_values):
        base = _base_feature_name(name, raw_columns)
        rows.append((base, float(value)))

    # Aggregate one-hot encoded columns back to the original business feature.
    grouped: dict[str, float] = {}
    for base, value in rows:
        grouped[base] = grouped.get(base, 0.0) + value

    result = []
    for feature, value in grouped.items():
        raw_value = raw_row.get(feature, None)
        result.append({
            "feature": feature,
            "label": _label(feature),
            "value": _value_text(raw_value),
            "shap_value": round(float(value), 6),
            "abs_shap": round(abs(float(value)), 6),
            "direction": "INCREASES_RISK" if value > 0 else "DECREASES_RISK",
            "reason": human_reason(feature, raw_value, value),
        })

    result.sort(key=lambda x: x["abs_shap"], reverse=True)
    top = result[:top_n]

    return {
        "top_factors": top,
        "risk_drivers": [x for x in result if x["shap_value"] > 0][:top_n],
        "risk_reducers": [x for x in result if x["shap_value"] < 0][:top_n],
        "all_factors": result,
    }
