"""FastAPI inference service.

Models are loaded exactly once at process startup and reused for every request.
No training, fitting, or SHAP explainer creation occurs in request handlers.
"""

from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.preprocessing import build_features
from app.shap_utils import build_explanation, risk_description, risk_level

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
STATIC_DIR = ROOT / "static"
MODEL_VERSION = os.getenv("ATTRITION_MODEL_VERSION", "1.0.0")
BUNDLE_PATH = MODEL_DIR / f"attrition_bundle_v{MODEL_VERSION}.joblib"

STATE: dict[str, Any] = {}


def _load_bundle():
    if not BUNDLE_PATH.exists():
        raise RuntimeError(
            f"Model bundle not found: {BUNDLE_PATH}. "
            "Run the training script first."
        )
    bundle = joblib.load(BUNDLE_PATH)

    # Load all persisted artifacts once into process memory.
    STATE["bundle"] = bundle
    STATE["pipeline"] = bundle["pipeline"]
    STATE["preprocessor"] = bundle["preprocessor"]
    STATE["base_tree_model"] = bundle["base_tree_model"]
    STATE["explainer"] = bundle["shap_explainer"]
    STATE["feature_names"] = bundle["feature_names"]
    STATE["metadata"] = bundle["metadata"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_bundle()
    yield
    STATE.clear()


app = FastAPI(
    title="Employee Attrition Intelligence API",
    version="1.0.0",
    lifespan=lifespan,
)


class EmployeeRequest(BaseModel):
    data: dict[str, Any]


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _predict_dataframe(raw_df: pd.DataFrame) -> list[dict[str, Any]]:
    if raw_df.empty:
        return []

    expected = STATE["metadata"]["raw_feature_columns"]
    missing = [c for c in expected if c not in raw_df.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"message": "Missing model input columns", "missing_columns": missing},
        )

    X_raw = raw_df[expected].copy()

    # Prediction path: raw data -> persisted pipeline -> calibrated probability.
    probabilities = STATE["pipeline"].predict_proba(X_raw)[:, 1]

    # SHAP path: raw data -> persisted preprocessor -> preloaded TreeExplainer.
    X_transformed = STATE["preprocessor"].transform(X_raw)
    shap_values = STATE["explainer"].shap_values(X_transformed)

    results = []
    for i, probability in enumerate(probabilities):
        prob = float(probability)
        level = risk_level(prob)

        # Normalize per-row SHAP values without recreating the explainer.
        from app.shap_utils import normalize_shap_values
        row_shap = normalize_shap_values(shap_values[i:i+1])

        explanation = build_explanation(
            row_shap,
            STATE["feature_names"],
            X_raw.iloc[i],
            expected,
            top_n=10,
        )
        for factor in explanation["top_factors"]:
            factor["review_action"] = _review_action(factor)

        employee_id = _json_scalar(raw_df.iloc[i].get("Employee_ID"))
        results.append({
            "employee_id": employee_id,
            "attrition_probability": round(prob, 6),
            "attrition_probability_pct": round(prob * 100, 2),
            "prediction": "Attrition" if prob >= 0.50 else "Stay",
            "risk": level,
            "risk_description": risk_description(level),
            "top_factors": explanation["top_factors"],
            "risk_drivers": explanation["risk_drivers"],
            "risk_reducers": explanation["risk_reducers"],
            "recommendations": _recommendations(explanation["risk_drivers"]),
        })
    return results


def _recommendations(drivers):
    recs = []
    seen = set()
    for factor in drivers[:5]:
        text = _review_action(factor)
        if text not in seen:
            recs.append(text)
            seen.add(text)
    return recs[:5]


def _review_action(factor):
    f = factor["feature"]
    if f in {"Years_Since_Last_Promotion", "Promotion_Gap_Flag"}:
        return "Review career development and promotion path."
    if "Overtime" in f:
        return "Assess workload and rebalance sustained overtime."
    if "Engagement" in f or "Satisfaction" in f:
        return "Schedule an engagement/1:1 discussion and review support needs."
    if "Salary" in f or "Income" in f or "Compensation" in f:
        return "Review compensation against role/grade benchmarks."
    if "Role_Stagnation" in f or "Years_In_Current_Role" in f:
        return "Explore role rotation, skill growth or new responsibilities."
    if "Manager" in f:
        return "Review manager relationship, support and team stability."
    return f"Review {factor['label'].lower()} as a potential intervention area."


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "pipeline" in STATE,
        "bundle_version": STATE.get("metadata", {}).get("bundle_version"),
        "selected_model": STATE.get("metadata", {}).get("selected_model"),
    }


@app.get("/model/info")
def model_info():
    if "metadata" not in STATE:
        raise HTTPException(503, "Model is not loaded.")
    return {
        **STATE["metadata"],
        "model_comparison": STATE["bundle"]["model_comparison"],
        "holdout_metrics": STATE["bundle"]["holdout_metrics"],
    }


@app.post("/predict")
def predict(request: EmployeeRequest):
    try:
        df = pd.DataFrame([request.data])
        return _predict_dataframe(df)[0]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        content = await file.read()
        name = (file.filename or "").lower()
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(415, "Upload CSV or XLSX.")
        return {
            "count": len(df),
            "results": _predict_dataframe(df),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, detail=str(exc))
