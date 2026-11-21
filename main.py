
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict
from fastapi import FastAPI, HTTPException, Body

# Get the folder where main.py is saved
BASE_DIR = Path(__file__).resolve().parent

# Load the saved model
model_artifact = joblib.load(BASE_DIR / "student_dropout_model.pkl")

model = model_artifact["model"]
le = model_artifact["label_encoder"]
final_threshold = model_artifact["final_threshold"]

# Load feature columns
with open(BASE_DIR / "feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# Original input columns from the student dataset
INPUT_COLUMNS = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Previous qualification (grade)",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Age at enrollment",
    "International",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

# Create FastAPI application
app = FastAPI(
    title="Student Dropout Prediction API",
    description="Predict student academic outcomes using Random Forest",
    version="1.0"
)

# Home page
@app.get("/")
def home():
    return {
        "message": "Student Dropout Prediction API is running"
    }

# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }

# Student prediction
@app.post("/predict")
def predict(student_data: Dict[str, float] = Body(...)):

    # Check for missing input features
    missing_columns = [
        col for col in INPUT_COLUMNS if col not in student_data
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail={"missing_columns": missing_columns}
        )

    # Check for extra input features
    extra_columns = [
        col for col in student_data if col not in INPUT_COLUMNS
    ]

    if extra_columns:
        raise HTTPException(
            status_code=400,
            detail={"unexpected_columns": extra_columns}
        )

    # Convert input into a DataFrame
    input_df = pd.DataFrame([student_data])

    # Restore the original dataset column names
    input_df = input_df.rename(
        columns={
            "Daytime/evening attendance":
                "Daytime/evening attendance\t",
            "Mother's qualification":
                '"Mother\'s qualification"',
            "Father's qualification":
                '"Father\'s qualification"',
            "Mother's occupation":
                '"Mother\'s occupation"',
            "Father's occupation":
                '"Father\'s occupation"'
        }
    )

    # Apply the same preprocessing used during training
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Arrange columns in the exact training order
    input_df = input_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Generate prediction probabilities
    probabilities = model.predict_proba(input_df)

    # Find class positions
    enrolled_label = int(le.transform(["Enrolled"])[0])
    dropout_label = int(le.transform(["Dropout"])[0])

    enrolled_position = np.where(
        model.classes_ == enrolled_label
    )[0][0]

    dropout_position = np.where(
        model.classes_ == dropout_label
    )[0][0]

    # Apply the same Enrolled threshold used during training
    predicted_labels = model.classes_[
        np.argmax(probabilities, axis=1)
    ].copy()

    if probabilities[0, enrolled_position] >= final_threshold:
        predicted_labels[0] = enrolled_label

    # Convert encoded prediction into original class name
    prediction = le.inverse_transform(
        predicted_labels.astype(int)
    )[0]

    # Calculate dropout probability
    dropout_probability = float(
        probabilities[0, dropout_position]
    )

    # Assign risk category
    if dropout_probability >= 0.70:
        risk_tier = "High"
    elif dropout_probability >= 0.40:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    # Return prediction results
    class_probabilities = {
        str(le.inverse_transform([int(label)])[0]):
            round(float(probabilities[0, i]), 4)
        for i, label in enumerate(model.classes_)
    }

    return {
        "predicted_outcome": str(prediction),
        "dropout_probability": round(dropout_probability, 4),
        "risk_tier": risk_tier,
        "class_probabilities": class_probabilities
    }
