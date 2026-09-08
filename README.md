# Employee Attrition ML + FastAPI + SHAP

This project is based on the supplied `Emp_Attrition_Prediction.ipynb` workflow and implements the requested production-style separation between **offline training** and **online inference**.

## Architecture

```text
TRAINING (run only when a new training dataset/model version is needed)
Excel/CSV
   |
   v
app/train.py
   |
   +--> preprocessing schema
   +--> 5-fold CV: Random Forest / XGBoost / LightGBM / CatBoost
   +--> select by CV PR-AUC, ROC-AUC tie-break
   +--> fit final tree model
   +--> calibrate probability
   +--> create SHAP TreeExplainer
   |
   +--> models/attrition_pipeline_vX.joblib
   +--> models/attrition_shap_vX.joblib
   +--> models/attrition_bundle_vX.joblib
   +--> metrics / metadata / comparison


RUNTIME (run repeatedly; NEVER retrains)
HTML UI
   |
   v
FastAPI
   |
   +--> raw employee data
   |       |
   |       v
   |   preloaded calibrated pipeline
   |       |
   |       +--> probability + risk
   |
   +--> same raw employee data
           |
           v
       preloaded preprocessor
           |
           v
       preloaded SHAP TreeExplainer
           |
           v
       top business-level factors
```

## Important design decision

The API uses **two persisted paths**:

1. **Prediction:** raw input -> saved preprocessing + calibrated classifier pipeline.
2. **SHAP:** raw input -> saved preprocessor -> preloaded TreeExplainer for the underlying tree model.

The SHAP explainer is created during training and loaded at startup. It is not created per request.

Because calibration is applied to the prediction probability, SHAP values explain the **underlying tree model's contribution**, not a causal statement and not a mathematically exact decomposition of the calibrated probability.

## Source data

The notebook references the `Emp_Data` sheet of the ML-ready employee dataset. The included training command below uses:

`Employee_Attrition_ML_Ready (1).xlsx`

The notebook/source dataset contains 1,470 records and the target is `Attrition` (`Yes`/`No`). `Employee_ID` is retained for reporting but excluded from model features.

The training code also excludes succession/continuity fields such as `Successor_Readiness_Pct` and `Backup_Coverage_Pct` from attrition prediction.

## 1. Install

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 2. Train a model version

Run this only when you want to create/recreate a model from a new training dataset:

```bash
python -m app.train --data "Employee_Attrition_ML_Ready (1).xlsx" --sheet Emp_Data --version 1.0.0
```

For a new training dataset/model release:

```bash
python -m app.train --data "new_training_data.xlsx" --sheet Emp_Data --version 1.1.0
```

The script compares:

- Random Forest
- XGBoost
- LightGBM
- CatBoost

Model selection prioritizes cross-validated **PR-AUC**, with **ROC-AUC** as the tie-break. This is intentional because attrition is imbalanced.

Probability calibration uses sigmoid calibration.

## 3. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## 4. API examples

### Single employee

POST `/predict`

```json
{
  "data": {
    "Employee_ID": 1001,
    "Department": "Engineering",
    "Job_Role": "Software Engineer",
    "Skills": "Python, AWS, Docker",
    "Overtime": "Yes",
    "Promotion_Gap_Flag": 1,
    "Role_Stagnation_Flag": 1,
    "Work_Arrangement": 0,
    "Age": 34,
    "Total_Working_Years": 10,
    "Years_At_Company": 7,
    "Years_In_Current_Role": 4,
    "Years_Since_Last_Promotion": 3,
    "Years_With_Current_Manager": 2,
    "Num_Companies_Worked": 3,
    "Monthly_Income": 7000,
    "Percent_Salary_Hike": 8,
    "Performance_Rating": 3,
    "Training_Hours_12M": 8,
    "Certification_Count": 1,
    "Engagement_Score": 2,
    "Recognition_Count_12M": 0,
    "Active_Project_Count": 4,
    "Tenure_Ratio": 0.70,
    "Total_Companies_Ratio": 0.30
  }
}
```

The response includes:

- calibrated attrition probability
- LOW / MEDIUM / HIGH / CRITICAL risk
- Attrition / Stay
- top SHAP factors
- risk drivers
- risk reducers
- suggested review actions

### Batch

POST `/predict/batch`

Upload CSV/XLSX with the same feature columns. `Employee_ID` may be included. `Attrition` can also be present in a test file; it is ignored by the prediction feature builder.

## 5. Model bundle

The primary runtime artifact is:

```text
models/attrition_bundle_v1.0.0.joblib
```

It contains:

- calibrated inference pipeline
- fitted preprocessor
- underlying fitted tree model
- SHAP TreeExplainer
- encoded feature names
- encoded-to-business feature mapping
- model comparison metrics
- holdout metrics
- risk thresholds
- feature schema
- version metadata

Separate artifacts are also emitted:

```text
attrition_pipeline_v1.0.0.joblib
attrition_shap_v1.0.0.joblib
```

## 6. Reusing the same model

When testing new employee data, do **not** run `train.py`.

The service loads the bundle once during FastAPI startup:

```python
STATE["pipeline"] = bundle["pipeline"]
STATE["explainer"] = bundle["shap_explainer"]
STATE["preprocessor"] = bundle["preprocessor"]
```

Every request reuses these objects.

For a production deployment, run multiple workers only after validating the memory footprint because each worker normally has its own Python process and its own loaded model copy.

## 7. Versioning

Do not overwrite production model files. Create a new version:

```text
attrition_bundle_v1.0.0.joblib
attrition_bundle_v1.1.0.joblib
attrition_bundle_v2.0.0.joblib
```

Then point the runtime configuration to the approved version.

## 8. Production considerations

This benchmark-style dataset is useful for demonstrating the end-to-end architecture. For a real enterprise attrition model, use longitudinal employee snapshots and define a prediction horizon, such as "employee leaves within the next 180 days." Then add temporal validation, calibration monitoring, data drift checks, subgroup/fairness monitoring, model governance, access control, audit logging, and human review.

SHAP indicates model contribution/correlation with the prediction. It should not be presented to HR users as proof that a factor caused an employee to leave.

## 9. Why preprocessing is persisted

The API must never rebuild encoders/scalers from the test dataset. The training-time fitted preprocessor stores:

- numeric imputation values
- scaling parameters
- categorical imputation values
- one-hot categories
- encoded feature order

This prevents training/test schema drift and guarantees that the tree model sees the same feature representation at inference time.
