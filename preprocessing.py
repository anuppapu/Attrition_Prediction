"""Training/inference feature preparation.

This module is imported by BOTH train.py and the FastAPI runtime so that the
same feature schema and feature engineering logic is used in both paths.
"""

from __future__ import annotations

from typing import Iterable
import pandas as pd

TARGET = "Attrition"
ID_COLUMNS = {"Employee_ID", "EmployeeNumber", "Employee_Number"}

# These are succession/workforce-continuity variables, not attrition predictors.
EXCLUDED_MODEL_COLUMNS = {
    "Manager_ID",
    "Joining_Date",
    "Last_Promotion_Date",
    "Critical_Responsibility_Count",
    "Critical_Responsibility_Without_Backup",
    "Backup_Coverage_Pct",
    "Successor_Readiness_Pct",
    "Production_Task_Count",
}

def prepare_raw_dataframe(df: pd.DataFrame, require_target: bool = False) -> pd.DataFrame:
    """Normalize incoming raw employee data without fitting anything."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if require_target and TARGET not in out.columns:
        raise ValueError(f"Training data must contain '{TARGET}'.")

    # Employee_ID is retained outside X for reporting but never used as a feature.
    return out

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the exact model input feature frame used at training and inference."""
    out = prepare_raw_dataframe(df)

    drop_cols = {TARGET, *ID_COLUMNS, *EXCLUDED_MODEL_COLUMNS}
    X = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore").copy()

    # The source notebook treats Skills as a categorical field. Keep it as-is.
    # Derived columns already present in the supplied ML-ready workbook are retained.
    return X

def feature_columns(df: pd.DataFrame) -> list[str]:
    return build_features(df).columns.tolist()

def split_target(df: pd.DataFrame):
    raw = prepare_raw_dataframe(df, require_target=True)
    X = build_features(raw)
    y = raw[TARGET].astype(str).str.strip().str.lower().eq("yes").astype(int)
    return X, y
