"""Production-style training entry point.

Usage:
    python -m app.train --data "Employee_Attrition_ML_Ready (1).xlsx" --version 1.0.0

Training is an offline operation. The FastAPI service never calls this module.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from app.preprocessing import split_target

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models"


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numerical),
        ("cat", categorical_pipe, categorical),
    ], remainder="drop")


def model_factories(scale_pos_weight: float):
    return {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight, random_state=42,
            eval_metric="logloss", n_jobs=-1
        ),
        "LightGBM": lambda: LGBMClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            class_weight="balanced", random_state=42,
            verbose=-1, n_jobs=-1
        ),
        "CatBoost": lambda: CatBoostClassifier(
            iterations=250, depth=6, learning_rate=0.08,
            l2_leaf_reg=3, scale_pos_weight=scale_pos_weight,
            random_state=42, verbose=False,
            allow_writing_files=False, thread_count=-1
        ),
    }


def cv_compare(X_train, y_train, factories, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows = []

    # Manual CV is intentional: it avoids estimator-tag incompatibilities
    # between some CatBoost versions and newer scikit-learn releases.
    for name, factory in factories.items():
        pr_scores, roc_scores = [], []

        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr = X_train.iloc[tr_idx]
            X_va = X_train.iloc[va_idx]
            y_tr = y_train.iloc[tr_idx]
            y_va = y_train.iloc[va_idx]

            prep = make_preprocessor(X_tr)
            A = prep.fit_transform(X_tr, y_tr)
            B = prep.transform(X_va)

            model = factory()
            model.fit(A, y_tr)
            p = model.predict_proba(B)[:, 1]

            pr_scores.append(average_precision_score(y_va, p))
            roc_scores.append(roc_auc_score(y_va, p))

        rows.append({
            "model": name,
            "cv_pr_auc_mean": float(np.mean(pr_scores)),
            "cv_pr_auc_std": float(np.std(pr_scores)),
            "cv_roc_auc_mean": float(np.mean(roc_scores)),
            "cv_roc_auc_std": float(np.std(roc_scores)),
        })

    return pd.DataFrame(rows).sort_values(
        ["cv_pr_auc_mean", "cv_roc_auc_mean"], ascending=False
    ).reset_index(drop=True)


def build_feature_metadata(preprocessor, raw_columns):
    names = preprocessor.get_feature_names_out().tolist()
    mapping = []
    for name in names:
        base = name.split("__", 1)[1] if "__" in name else name
        for col in sorted(raw_columns, key=len, reverse=True):
            if base == col or base.startswith(col + "_"):
                mapping.append({"encoded_feature": name, "business_feature": col})
                break
        else:
            mapping.append({"encoded_feature": name, "business_feature": base})

    return names, mapping


def main(args):
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(data_path, sheet_name=args.sheet) if data_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(data_path)
    X, y = split_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    scale_pos_weight = float((len(y_train) - y_train.sum()) / y_train.sum())
    factories = model_factories(scale_pos_weight)
    comparison = cv_compare(X_train, y_train, factories)

    best_name = comparison.iloc[0]["model"]

    # Fit one final preprocessor on the complete training partition.
    preprocessor = make_preprocessor(X_train)
    A_train = preprocessor.fit_transform(X_train, y_train)
    A_test = preprocessor.transform(X_test)

    # Base tree model: used for SHAP and retained in the bundle.
    base_model = factories[best_name]()
    base_model.fit(A_train, y_train)

    # Calibrated predictor: probability returned by the API.
    calibrated = CalibratedClassifierCV(
        estimator=clone(factories[best_name]()),
        method="sigmoid",
        cv=3,
    )
    calibrated.fit(A_train, y_train)

    calibrated_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", calibrated),
    ])

    p = calibrated_pipeline.predict_proba(X_test)[:, 1]
    pred = (p >= 0.50).astype(int)

    metrics = {
        "holdout_roc_auc": float(roc_auc_score(y_test, p)),
        "holdout_pr_auc": float(average_precision_score(y_test, p)),
        "holdout_brier": float(brier_score_loss(y_test, p)),
        "holdout_accuracy": float(accuracy_score(y_test, pred)),
        "holdout_f1": float(f1_score(y_test, pred, zero_division=0)),
        "holdout_precision": float(precision_score(y_test, pred, zero_division=0)),
        "holdout_recall": float(recall_score(y_test, pred, zero_division=0)),
    }

    # TreeExplainer is created once offline and persisted.
    explainer = shap.TreeExplainer(base_model)
    feature_names, mapping = build_feature_metadata(preprocessor, X.columns.tolist())

    # Bundle keeps the exact inference objects and metadata together.
    version = args.version
    metadata = {
        "bundle_version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target": "Attrition",
        "positive_class": "Yes",
        "selected_model": best_name,
        "risk_thresholds": {
            "LOW": [0.0, 0.30],
            "MEDIUM": [0.30, 0.60],
            "HIGH": [0.60, 0.80],
            "CRITICAL": [0.80, 1.0],
        },
        "raw_feature_columns": X.columns.tolist(),
        "encoded_feature_count": len(feature_names),
        "source_file": data_path.name,
        "python_version": platform.python_version(),
        "sklearn_version": __import__("sklearn").__version__,
        "shap_version": shap.__version__,
    }

    bundle = {
        "metadata": metadata,
        "pipeline": calibrated_pipeline,
        "preprocessor": preprocessor,
        "base_tree_model": base_model,
        "shap_explainer": explainer,
        "feature_names": feature_names,
        "feature_mapping": mapping,
        "model_comparison": comparison.to_dict(orient="records"),
        "holdout_metrics": metrics,
    }

    pipeline_path = model_dir / f"attrition_pipeline_v{version}.joblib"
    shap_path = model_dir / f"attrition_shap_v{version}.joblib"
    bundle_path = model_dir / f"attrition_bundle_v{version}.joblib"

    joblib.dump(calibrated_pipeline, pipeline_path, compress=3)
    joblib.dump({
        "explainer": explainer,
        "base_tree_model": base_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "feature_mapping": mapping,
    }, shap_path, compress=3)
    joblib.dump(bundle, bundle_path, compress=3)

    comparison.to_csv(model_dir / f"model_comparison_v{version}.csv", index=False)
    (model_dir / f"metadata_v{version}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / f"metrics_v{version}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({
        "selected_model": best_name,
        "model_comparison": comparison.to_dict(orient="records"),
        "holdout_metrics": metrics,
        "pipeline": str(pipeline_path),
        "shap": str(shap_path),
        "bundle": str(bundle_path),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--sheet", default="Emp_Data")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    main(parser.parse_args())
