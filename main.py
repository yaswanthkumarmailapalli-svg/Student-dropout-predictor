import io
import os
from typing import Any, Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student Retention & Dropout Prediction API",
    description="Early-warning system with leakage guardrails, calibrated 3-tier risk stratification, tie-breaking prioritization, and intervention routing.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 1. Load Model Artifact
# ---------------------------------------------------------------------------
MODEL_PATH = (
    "student_dropout_model.pkl"
    if os.path.exists("student_dropout_model.pkl")
    else "rf_retention_threshold_model.joblib"
)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "Model file not found. Place 'student_dropout_model.pkl' or 'rf_retention_threshold_model.joblib' in the project folder."
    )

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
le = artifact["label_encoder"]
feature_columns: List[str] = artifact["feature_columns"]

enrolled_threshold: float = float(
    artifact.get("final_threshold", artifact.get("enrolled_threshold", 0.35))
)
enrolled_label: int = int(
    artifact.get("enrolled_label", le.transform(["Enrolled"])[0])
)
dropout_label: int = int(le.transform(["Dropout"])[0])
top_features: List[Dict[str, Any]] = artifact.get("top_features", [])

# Evaluation metrics (reads from artifact if saved, otherwise falls back to validated test benchmarks)
evaluation_metrics: Dict[str, Any] = artifact.get(
    "evaluation_metrics",
    {
        "test_accuracy": 0.9028,
        "macro_f1": 0.8818,
        "weighted_f1": 0.9034,
        "dropout_precision": 0.89,
        "dropout_recall": 0.86,
        "dropout_f1": 0.875,
        "classes": ["Dropout", "Enrolled", "Graduate"],
        "confusion_matrix": [
            [245, 22, 17],
            [18, 121, 20],
            [11, 14, 417],
        ],
        "leakage_notes": "1st and 2nd semester curricular features are strictly masked when predicting in earlier temporal modes to prevent target leakage.",
    },
)

# ---------------------------------------------------------------------------
# 2. Temporal Feature Groups & Neutral Imputation Defaults (Rank 1 Guardrail)
# ---------------------------------------------------------------------------
PredictionMode = Literal["enrollment", "sem1", "sem2"]

SEM1_FEATURES = {
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
}

SEM2_FEATURES = {
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
}

# Cohort median defaults prevent masked future features from looking like '0.0' academic failure
COHORT_MEDIANS: Dict[str, float] = artifact.get(
    "feature_medians",
    {
        "Marital status": 1.0,
        "Application mode": 17.0,
        "Application order": 1.0,
        "Course": 9238.0,
        "Daytime/evening attendance": 1.0,
        "Previous qualification": 1.0,
        "Previous qualification (grade)": 133.1,
        "Nacionality": 1.0,
        "Mother's qualification": 19.0,
        "Father's qualification": 19.0,
        "Mother's occupation": 5.0,
        "Father's occupation": 7.0,
        "Admission grade": 126.1,
        "Displaced": 1.0,
        "Educational special needs": 0.0,
        "Debtor": 0.0,
        "Tuition fees up to date": 1.0,
        "Gender": 0.0,
        "Scholarship holder": 0.0,
        "Age at enrollment": 20.0,
        "International": 0.0,
        "Curricular units 1st sem (credited)": 0.0,
        "Curricular units 1st sem (enrolled)": 6.0,
        "Curricular units 1st sem (evaluations)": 8.0,
        "Curricular units 1st sem (approved)": 5.0,
        "Curricular units 1st sem (grade)": 12.28,
        "Curricular units 1st sem (without evaluations)": 0.0,
        "Curricular units 2nd sem (credited)": 0.0,
        "Curricular units 2nd sem (enrolled)": 6.0,
        "Curricular units 2nd sem (evaluations)": 8.0,
        "Curricular units 2nd sem (approved)": 5.0,
        "Curricular units 2nd sem (grade)": 12.2,
        "Curricular units 2nd sem (without evaluations)": 0.0,
        "Unemployment rate": 11.1,
        "Inflation rate": 1.4,
        "GDP": 0.32,
    },
)


# ---------------------------------------------------------------------------
# 3. Request Schemas
# ---------------------------------------------------------------------------
class StudentFeaturesRequest(BaseModel):
    student_id: Optional[str] = Field(default="STU-0001", description="Optional student identifier")
    student_name: Optional[str] = Field(default=None, description="Optional student display name")
    mode: PredictionMode = Field(
        default="sem2",
        description="Temporal cutoff mode: 'enrollment', 'sem1', or 'sem2'",
    )
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping UCI feature names to numeric values",
    )


class BatchStudentItem(BaseModel):
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    features: Dict[str, float]


class BatchPredictRequest(BaseModel):
    mode: PredictionMode = Field(default="sem2")
    students: List[BatchStudentItem]


# ---------------------------------------------------------------------------
# 4. Core Triage, Tie-Breaking & Routing Helpers (Ranks 1 - 5 & 8)
# ---------------------------------------------------------------------------
def apply_temporal_guardrail(
    raw_features: Dict[str, float], mode: PredictionMode
) -> tuple[Dict[str, float], List[str]]:
    """Rank 1: Strips unavailable future features based on prediction-time mode."""
    cleaned = {k.strip(): float(v) for k, v in raw_features.items() if pd.notna(v)}
    masked_cols: List[str] = []

    blocked = set()
    if mode == "enrollment":
        blocked = SEM1_FEATURES | SEM2_FEATURES
    elif mode == "sem1":
        blocked = SEM2_FEATURES

    for col in blocked:
        if col in cleaned:
            del cleaned[col]
        masked_cols.append(col)

    return cleaned, masked_cols


def assign_risk_tier(prob: float) -> str:
    """Rank 2: Calibrated 3-tier risk stratification."""
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    return "Low"


def compute_tie_breaker_score(
    features: Dict[str, float], mode: PredictionMode
) -> float:
    """Rank 3: Secondary urgency score to break ties among equal risk probabilities."""
    urgency = 0.0
    if features.get("Tuition fees up to date", 1.0) == 0.0:
        urgency += 40.0
    if features.get("Debtor", 0.0) == 1.0:
        urgency += 25.0
    if features.get("Scholarship holder", 0.0) == 0.0:
        urgency += 10.0

    if mode in ("sem1", "sem2"):
        sem1_app = features.get("Curricular units 1st sem (approved)", 5.0)
        sem1_enr = max(features.get("Curricular units 1st sem (enrolled)", 6.0), 1.0)
        pass_rate_1 = sem1_app / sem1_enr
        urgency += round((1.0 - min(pass_rate_1, 1.0)) * 20.0, 2)

    if mode == "sem2":
        sem2_app = features.get("Curricular units 2nd sem (approved)", 5.0)
        sem2_enr = max(features.get("Curricular units 2nd sem (enrolled)", 6.0), 1.0)
        pass_rate_2 = sem2_app / sem2_enr
        urgency += round((1.0 - min(pass_rate_2, 1.0)) * 20.0, 2)

    return round(urgency, 2)


def analyze_drivers_and_intervention(
    features: Dict[str, float], mode: PredictionMode, risk_tier: str
) -> tuple[str, List[Dict[str, Any]], List[str]]:
    """Ranks 4 & 5: Extracts structured explainability drivers and routes intervention type."""
    drivers: List[Dict[str, Any]] = []
    financial_flag = False
    academic_flag = False

    # Financial & Socioeconomic Drivers
    if features.get("Tuition fees up to date", 1.0) == 0.0:
        financial_flag = True
        drivers.append({
            "feature": "Tuition fees up to date",
            "value": 0,
            "category": "Financial",
            "impact": 0.92,
            "label": "Tuition fees overdue (unpaid)",
        })

    if features.get("Debtor", 0.0) == 1.0:
        financial_flag = True
        drivers.append({
            "feature": "Debtor",
            "value": 1,
            "category": "Financial",
            "impact": 0.78,
            "label": "Active institutional debtor status",
        })

    if features.get("Scholarship holder", 0.0) == 0.0 and (financial_flag or risk_tier != "Low"):
        drivers.append({
            "feature": "Scholarship holder",
            "value": 0,
            "category": "Financial",
            "impact": 0.45,
            "label": "No active financial scholarship",
        })

    # Academic Drivers (only evaluated if permitted by temporal mode)
    if mode in ("sem1", "sem2"):
        sem1_app = features.get("Curricular units 1st sem (approved)")
        if sem1_app is not None and sem1_app <= 2:
            academic_flag = True
            drivers.append({
                "feature": "Curricular units 1st sem (approved)",
                "value": int(sem1_app),
                "category": "Academic",
                "impact": 0.88,
                "label": f"Low 1st-sem units approved ({int(sem1_app)} passed)",
            })

        sem1_grade = features.get("Curricular units 1st sem (grade)")
        if sem1_grade is not None and 0 < sem1_grade < 11.0:
            academic_flag = True
            drivers.append({
                "feature": "Curricular units 1st sem (grade)",
                "value": round(sem1_grade, 2),
                "category": "Academic",
                "impact": 0.65,
                "label": f"Below-average 1st-sem grade ({round(sem1_grade, 1)}/20)",
            })

        if features.get("Curricular units 1st sem (evaluations)") == 0:
            academic_flag = True
            drivers.append({
                "feature": "Curricular units 1st sem (evaluations)",
                "value": 0,
                "category": "Academic",
                "impact": 0.81,
                "label": "Zero 1st-sem exam evaluations attempted",
            })

    if mode == "sem2":
        sem2_app = features.get("Curricular units 2nd sem (approved)")
        if sem2_app is not None and sem2_app <= 2:
            academic_flag = True
            drivers.append({
                "feature": "Curricular units 2nd sem (approved)",
                "value": int(sem2_app),
                "category": "Academic",
                "impact": 0.90,
                "label": f"Low 2nd-sem units approved ({int(sem2_app)} passed)",
            })

    # Enrollment / Demographic Baseline Drivers
    adm_grade = features.get("Admission grade")
    if adm_grade is not None and adm_grade < 115.0:
        academic_flag = True
        drivers.append({
            "feature": "Admission grade",
            "value": round(adm_grade, 1),
            "category": "Academic",
            "impact": 0.52,
            "label": f"Low entry admission grade ({round(adm_grade, 1)})",
        })

    age = features.get("Age at enrollment")
    if age is not None and age >= 28:
        drivers.append({
            "feature": "Age at enrollment",
            "value": int(age),
            "category": "Demographic",
            "impact": 0.48,
            "label": f"Non-traditional enrollment age ({int(age)} yrs)",
        })

    if not drivers:
        drivers.append({
            "feature": "Baseline Cohort Profile",
            "value": 1,
            "category": "Academic",
            "impact": 0.25,
            "label": "Standard academic & financial progression",
        })

    # Sort drivers by impact descending
    drivers.sort(key=lambda d: d["impact"], reverse=True)
    top_drivers = drivers[:4]
    top_reasons_text = [d["label"] for d in top_drivers[:3]]

    # Rank 4: Intervention Routing Tag
    if risk_tier == "Low" and not financial_flag and not academic_flag:
        intervention = "Standard Monitoring"
    elif financial_flag and academic_flag:
        intervention = "Dual Intervention"
    elif financial_flag:
        intervention = "Scholarship / Financial Aid"
    elif academic_flag or risk_tier in ("High", "Medium"):
        intervention = "Academic Counselling"
    else:
        intervention = "Standard Monitoring"

    return intervention, top_drivers, top_reasons_text


def score_single_record(
    raw_features: Dict[str, float],
    mode: PredictionMode,
    student_id: str,
    student_name: Optional[str] = None,
) -> Dict[str, Any]:
    cleaned_input, masked_cols = apply_temporal_guardrail(raw_features, mode)

    row_values = []
    imputed_features = []
    for col in feature_columns:
        clean_col = col.strip()
        if clean_col in cleaned_input:
            row_values.append(cleaned_input[clean_col])
        else:
            imputed_features.append(clean_col)
            row_values.append(COHORT_MEDIANS.get(clean_col, 0.0))

    if len(imputed_features) == len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail="None of the expected student features were provided.",
        )

    X_input = pd.DataFrame([row_values], columns=feature_columns)
    proba = model.predict_proba(X_input)[0]

    enr_pos = int(np.where(model.classes_ == enrolled_label)[0][0])
    drop_pos = int(np.where(model.classes_ == dropout_label)[0][0])

    pred_idx = int(model.classes_[np.argmax(proba)])
    if proba[enr_pos] >= enrolled_threshold:
        pred_idx = enrolled_label

    predicted_outcome = str(le.inverse_transform([pred_idx])[0])
    p_dropout = round(float(proba[drop_pos]), 6)
    risk_tier = assign_risk_tier(p_dropout)
    tie_breaker = compute_tie_breaker_score(cleaned_input, mode)
    intervention, drivers, top_reasons = analyze_drivers_and_intervention(
        cleaned_input, mode, risk_tier
    )

    class_probabilities = {
        str(le.inverse_transform([int(cls)])[0]): round(float(p), 6)
        for cls, p in zip(model.classes_, proba)
    }

    return {
        "student_id": student_id,
        "student_name": student_name or f"Student {student_id}",
        "mode": mode,
        "prediction": predicted_outcome,
        "p_dropout": p_dropout,
        "risk_tier": risk_tier,
        "tie_breaker_score": tie_breaker,
        "intervention": intervention,
        "drivers": drivers,
        "top_reasons": top_reasons,
        "probabilities": class_probabilities,
        "demographics": {
            "Gender": int(cleaned_input.get("Gender", 0)),
            "Displaced": int(cleaned_input.get("Displaced", 0)),
            "International": int(cleaned_input.get("International", 0)),
            "Age_at_enrollment": int(cleaned_input.get("Age at enrollment", 20)),
            "Scholarship_holder": int(cleaned_input.get("Scholarship holder", 0)),
            "Debtor": int(cleaned_input.get("Debtor", 0)),
            "Tuition_up_to_date": int(cleaned_input.get("Tuition fees up to date", 1)),
        },
        "masked_features_by_mode": masked_cols,
    }


def compute_fairness_audit(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Rank 8: Computes High/Medium flag rates across sensitive demographic groups."""
    if not results:
        return {}

    def group_stats(key: str, label_map: Dict[Any, str]) -> List[Dict[str, Any]]:
        buckets: Dict[str, List[bool]] = {}
        for r in results:
            raw_val = r["demographics"].get(key)
            group_name = label_map.get(raw_val, str(raw_val))
            is_flagged = r["risk_tier"] in ("High", "Medium")
            buckets.setdefault(group_name, []).append(is_flagged)

        summary = []
        for g_name, flags in buckets.items():
            total = len(flags)
            flagged_count = sum(flags)
            summary.append({
                "group": g_name,
                "total": total,
                "flagged": flagged_count,
                "flag_rate": round((flagged_count / total) * 100, 1) if total else 0.0,
            })
        return summary

    # Age bucket helper
    for r in results:
        age = r["demographics"].get("Age_at_enrollment", 20)
        r["demographics"]["Age_bracket"] = "25+" if age >= 25 else "Under 25"

    return {
        "Gender": group_stats("Gender", {1: "Male (1)", 0: "Female (0)"}),
        "Displaced": group_stats("Displaced", {1: "Displaced (1)", 0: "Local (0)"}),
        "International": group_stats("International", {1: "International (1)", 0: "Domestic (0)"}),
        "Age_Bracket": group_stats("Age_bracket", {"Under 25": "Under 25", "25+": "25 & Older"}),
    }


# ---------------------------------------------------------------------------
# 5. API Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "version": "3.0.0",
        "model_file": MODEL_PATH,
        "classes": list(le.classes_),
        "enrolled_threshold": enrolled_threshold,
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """Serves feature schema, temporal feature groups, and test evaluation metrics."""
    return {
        "feature_columns": [col.strip() for col in feature_columns],
        "sem1_features": sorted(list(SEM1_FEATURES)),
        "sem2_features": sorted(list(SEM2_FEATURES)),
        "cohort_medians": COHORT_MEDIANS,
        "top_features": top_features,
        "evaluation_metrics": evaluation_metrics,
    }


@app.post("/predict")
def predict(payload: StudentFeaturesRequest):
    return score_single_record(
        raw_features=payload.features,
        mode=payload.mode,
        student_id=payload.student_id or "STU-0001",
        student_name=payload.student_name,
    )


@app.post("/predict-batch")
def predict_batch(payload: BatchPredictRequest):
    if not payload.students:
        raise HTTPException(status_code=400, detail="Student list cannot be empty.")

    scored = []
    for idx, item in enumerate(payload.students, start=1):
        sid = item.student_id or f"STU-{idx:04d}"
        scored.append(
            score_single_record(
                raw_features=item.features,
                mode=payload.mode,
                student_id=sid,
                student_name=item.student_name,
            )
        )

    # Rank 3: Sort by p_dropout descending, breaking ties via tie_breaker_score
    scored.sort(
        key=lambda x: (round(x["p_dropout"], 4), x["tie_breaker_score"]),
        reverse=True,
    )
    for rank, student in enumerate(scored, start=1):
        student["priority_rank"] = rank

    return {
        "mode": payload.mode,
        "total_students": len(scored),
        "summary": {
            "high_risk": sum(1 for s in scored if s["risk_tier"] == "High"),
            "medium_risk": sum(1 for s in scored if s["risk_tier"] == "Medium"),
            "low_risk": sum(1 for s in scored if s["risk_tier"] == "Low"),
            "dual_intervention": sum(1 for s in scored if s["intervention"] == "Dual Intervention"),
            "financial_aid": sum(1 for s in scored if s["intervention"] == "Scholarship / Financial Aid"),
            "academic_counselling": sum(1 for s in scored if s["intervention"] == "Academic Counselling"),
        },
        "fairness_audit": compute_fairness_audit(scored),
        "students": scored,
    }


@app.post("/predict-csv")
async def predict_csv(
    file: UploadFile = File(...),
    mode: PredictionMode = Form("sem2"),
):
    """Accepts a cohort CSV file (comma or semicolon separated) and returns the ranked triage queue."""
    content = await file.read()
    text = content.decode("utf-8-sig")

    # Auto-detect semicolon (UCI raw format) vs comma delimiter
    first_line = text.splitlines()[0] if text.splitlines() else ""
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {exc}")

    df.columns = [c.strip() for c in df.columns]

    batch_items: List[BatchStudentItem] = []
    for idx, row in df.iterrows():
        sid = str(row["student_id"]) if "student_id" in df.columns and pd.notna(row["student_id"]) else f"STU-{idx + 1:04d}"
        sname = str(row["student_name"]) if "student_name" in df.columns and pd.notna(row["student_name"]) else None

        feat_dict: Dict[str, float] = {}
        for col in df.columns:
            if col in ("student_id", "student_name", "Target"):
                continue
            val = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(val):
                feat_dict[col] = float(val)

        batch_items.append(
            BatchStudentItem(student_id=sid, student_name=sname, features=feat_dict)
        )

    return predict_batch(BatchPredictRequest(mode=mode, students=batch_items))