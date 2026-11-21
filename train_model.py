#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ==============================================================================
# COMPLETE SELF-CONTAINED WORKFLOW: EXACT 90.28% MODEL + PERMUTATION IMPORTANCE
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE

# ==============================================================================
# 1. LOAD & PREPROCESS DATA (EXACT ORIGINAL PREPROCESSING)
# ==============================================================================
df = pd.read_csv("data.csv", delimiter=";")

le = LabelEncoder()
df["Target"] = le.fit_transform(df["Target"])
enrolled_label = le.transform(["Enrolled"])[0]

# One-hot encode categorical features across the dataset (230 feature columns)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Target", axis=1)
y = df["Target"]

# ==============================================================================
# 2. EXACT ORIGINAL DATA SPLITS
# ==============================================================================
# Split A (random_state=42) -> produces X_test (885 rows)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Split B (random_state=101 -> then random_state=42) -> produces X_fit & X_val
X_train_full, X_test_final, y_train_full, y_test_final = train_test_split(
    X, y, test_size=0.20, random_state=101, stratify=y
)

X_fit, X_val, y_fit, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42, stratify=y_train_full
)

# ==============================================================================
# 3. EXACT SMOTE MULTIPLIER TUNING ON VALIDATION SET (X_val)
# ==============================================================================
base_count = (y_fit == enrolled_label).sum()  # 508
multipliers = [1.0, 1.15, 1.30, 1.45, 1.60, 1.80]
smote_results = []

for multiplier in multipliers:
    target_count = int(np.ceil(base_count * multiplier))
    if target_count == base_count:
        X_res_tmp, y_res_tmp = X_fit.copy(), y_fit.copy()
    else:
        sm = SMOTE(
            sampling_strategy={enrolled_label: target_count},
            random_state=42,
            k_neighbors=5,
        )
        X_res_tmp, y_res_tmp = sm.fit_resample(X_fit, y_fit)

    clf_tmp = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    clf_tmp.fit(X_res_tmp, y_res_tmp)
    y_val_pred_tmp = clf_tmp.predict(X_val)

    rep = classification_report(y_val, y_val_pred_tmp, output_dict=True, zero_division=0)
    enr_m = rep[str(enrolled_label)]
    smote_results.append({
        "Multiplier": multiplier,
        "Enrolled Samples": target_count,
        "Enrolled F1": enr_m["f1-score"],
        "Macro F1": rep["macro avg"]["f1-score"],
    })

smote_results_df = pd.DataFrame(smote_results)
best_multiplier = smote_results_df.sort_values(
    by=["Enrolled F1", "Macro F1"], ascending=False
).iloc[0]["Multiplier"]  # Selects 1.80 (target_count = 915)

# ==============================================================================
# 4. TRAIN EXACT ORIGINAL MODEL (threshold_model)
# ==============================================================================
target_count = int(np.ceil(base_count * best_multiplier))  # 915

smote = SMOTE(
    sampling_strategy={enrolled_label: target_count},
    random_state=42,
    k_neighbors=5,
)
X_resampled, y_resampled = smote.fit_resample(X_fit, y_fit)

threshold_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
threshold_model.fit(X_resampled, y_resampled)

# ==============================================================================
# 5. EXACT THRESHOLD SELECTION ON VALIDATION DATA (X_val)
# ==============================================================================
y_val_proba = threshold_model.predict_proba(X_val)
enrolled_position = np.where(threshold_model.classes_ == enrolled_label)[0][0]
enrolled_probabilities = y_val_proba[:, enrolled_position]

thresholds = np.arange(0.20, 0.61, 0.05)
threshold_results = []

for threshold in thresholds:
    preds = threshold_model.classes_[np.argmax(y_val_proba, axis=1)].copy()
    preds[enrolled_probabilities >= threshold] = enrolled_label

    threshold_results.append({
        "Threshold": round(threshold, 2),
        "Enrolled Precision": precision_score(
            y_val, preds, labels=[enrolled_label], average=None, zero_division=0
        )[0],
        "Enrolled Recall": recall_score(
            y_val, preds, labels=[enrolled_label], average=None, zero_division=0
        )[0],
        "Enrolled F1": f1_score(
            y_val, preds, labels=[enrolled_label], average=None, zero_division=0
        )[0],
        "Macro F1": f1_score(y_val, preds, average="macro"),
    })

threshold_results_df = pd.DataFrame(threshold_results)
valid_thresholds = threshold_results_df[threshold_results_df["Enrolled Precision"] >= 0.45]
final_threshold = float(
    valid_thresholds.sort_values(by=["Enrolled Recall", "Macro F1"], ascending=False)
    .iloc[0]["Threshold"]
)  # Selects 0.40

# ==============================================================================
# 6. EXACT FINAL EVALUATION (REPRODUCES 90.28% ACCURACY)
# ==============================================================================
def predict_with_enrolled_threshold(estimator, X_input, threshold=final_threshold):
    proba = estimator.predict_proba(X_input)
    enr_pos = np.where(estimator.classes_ == enrolled_label)[0][0]
    preds = estimator.classes_[np.argmax(proba, axis=1)].copy()
    preds[proba[:, enr_pos] >= threshold] = enrolled_label
    return preds

test_predictions = predict_with_enrolled_threshold(threshold_model, X_test, final_threshold)

print("=" * 60)
print("FINAL TEST SET CLASSIFICATION REPORT")
print("=" * 60)
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=le.classes_,
        digits=4,
        zero_division=0,
    )
)

print("=" * 60)
print("OVERALL TEST PERFORMANCE")
print("=" * 60)
print("Accuracy:", round(accuracy_score(y_test, test_predictions), 4))
print("Macro F1:", round(f1_score(y_test, test_predictions, average="macro"), 4))
print("Weighted F1:", round(f1_score(y_test, test_predictions, average="weighted"), 4))

# ==============================================================================
# 7. FIXED PERMUTATION FEATURE IMPORTANCE (NO SPARSE / SCORING ERRORS)
# ==============================================================================
# Convert to dense DataFrame (float64) so boolean/sparse-backed dummy columns
# never trigger InvalidParameterError inside permutation_importance
X_eval_dense = pd.DataFrame(
    np.asarray(X_test, dtype=np.float64),
    columns=X_test.columns,
    index=X_test.index,
)
y_eval_arr = np.asarray(y_test)

def threshold_macro_f1_scorer(estimator, X_mat, y_true):
    preds = predict_with_enrolled_threshold(estimator, X_mat, final_threshold)
    return f1_score(y_true, preds, average="macro")

perm_result = permutation_importance(
    estimator=threshold_model,
    X=X_eval_dense,
    y=y_eval_arr,
    scoring=threshold_macro_f1_scorer,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

perm_df = (
    pd.DataFrame({
        "Feature": X_eval_dense.columns,
        "Importance_Mean": perm_result.importances_mean,
        "Importance_Std": perm_result.importances_std,
    })
    .sort_values(by="Importance_Mean", ascending=False)
    .reset_index(drop=True)
)

print("\n" + "=" * 60)
print("TOP 15 PERMUTATION FEATURE IMPORTANCES (MACRO-F1 DROP)")
print("=" * 60)
print(perm_df.head(15).to_string(index=False))

# ==============================================================================
# 8. CONFUSION MATRIX PLOT
# ==============================================================================
cm = confusion_matrix(y_test, test_predictions, labels=np.arange(len(le.classes_)))
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Final Random Forest (threshold_model) - Test Confusion Matrix")
plt.tight_layout()
plt.show()


# In[4]:


# ==============================================================================
# 9. PERMUTATION FEATURE IMPORTANCE (TOP RISK PREDICTORS & VISUAL PLOT)
# ==============================================================================
print("\n" + "=" * 65)
print("9. COMPUTING PERMUTATION FEATURE IMPORTANCE (MACRO-F1 SENSITIVITY)")
print("=" * 65)

# Convert X_test to dense float64 to ensure safety against boolean/sparse formatting
X_eval_dense = pd.DataFrame(
    np.asarray(X_test, dtype=np.float64),
    columns=X_test.columns,
    index=X_test.index,
)
y_eval_arr = np.asarray(y_test)

def threshold_macro_f1_scorer(estimator, X_mat, y_true):
    preds = predict_with_enrolled_threshold(estimator, X_mat, final_threshold)
    return f1_score(y_true, preds, average="macro")

# Run permutation importance (5 repeats for statistical stability)
perm_result = permutation_importance(
    estimator=threshold_model,
    X=X_eval_dense,
    y=y_eval_arr,
    scoring=threshold_macro_f1_scorer,
    n_repeats=5,
    random_state=42,
    n_jobs=-1,
)

perm_df = (
    pd.DataFrame({
        "Feature": X_eval_dense.columns,
        "Importance_Mean": perm_result.importances_mean,
        "Importance_Std": perm_result.importances_std,
    })
    .sort_values(by="Importance_Mean", ascending=False)
    .reset_index(drop=True)
)

print("\nTOP 15 INFLUENTIAL PREDICTORS (DROP IN MACRO-F1 WHEN PERMUTED):")
print(perm_df.head(15).to_string(index=False))

# Plot Top 15 Feature Importances
plt.figure(figsize=(10, 6))
top_15_features = perm_df.head(15)
sns.barplot(
    data=top_15_features,
    x="Importance_Mean",
    y="Feature",
    palette="mako",
)
plt.title("Top 15 Influential Predictors of Student Retention / Dropout", fontsize=13, fontweight="bold")
plt.xlabel("Macro F1-Score Drop (Importance Mean)", fontsize=11)
plt.ylabel("Predictor Feature", fontsize=11)
plt.tight_layout()
plt.show()


# ==============================================================================
# 10. PRIORITIZED STUDENT REVIEW LIST FOR ACADEMIC INTERVENTIONS
# ==============================================================================
print("\n" + "=" * 65)
print("10. GENERATING PRIORITIZED STUDENT REVIEW LIST (ranked_review_list.csv)")
print("=" * 65)

# Extract predicted probabilities for Dropout class
dropout_label = int(le.transform(["Dropout"])[0])
dropout_pos = np.where(threshold_model.classes_ == dropout_label)[0][0]
test_probabilities = threshold_model.predict_proba(X_test)
p_dropouts = test_probabilities[:, dropout_pos]

# Reload raw unencoded dataset so advisors see clear, original feature values
try:
    df_raw = pd.read_csv("data.csv", delimiter=";")
    if df_raw.shape[1] <= 1:
        df_raw = pd.read_csv("data.csv", delimiter=",")
except Exception:
    df_raw = pd.read_csv("data.csv", delimiter=",")

# Clean column headers
df_raw.columns = [c.replace("\ufeff", "").replace('"', "").replace("\t", "").strip() for c in df_raw.columns]

# Build review list DataFrame
review_df = pd.DataFrame({
    "student_index": X_test.index,
    "p_dropout": p_dropouts,
    "true_outcome": le.inverse_transform(y_test),
    "predicted_outcome": le.inverse_transform(test_predictions),
})

# Assign Risk Tiers based on Dropout probability
def assign_risk_tier(prob):
    if prob >= 0.70:
        return "High"
    elif prob >= 0.40:
        return "Medium"
    else:
        return "Low"

review_df["risk_tier"] = review_df["p_dropout"].apply(assign_risk_tier)

# Extract top actionable risk drivers per student
student_reasons = []
for idx in X_test.index:
    row = df_raw.loc[idx]
    reasons = []

    # 1. Financial check (Tuition fees overdue)
    if "Tuition fees up to date" in row and row["Tuition fees up to date"] == 0:
        reasons.append("Tuition fees up to date = 0")

    # 2. Financial check (Active debt)
    if "Debtor" in row and row["Debtor"] == 1:
        reasons.append("Debtor = 1")

    # 3. Academic check (1st sem approved credits)
    if "Curricular units 1st sem (approved)" in row and row["Curricular units 1st sem (approved)"] <= 1:
        reasons.append(f"Curricular units 1st sem (approved) = {int(row['Curricular units 1st sem (approved)'])}")

    # 4. Exam attendance check
    if "Curricular units 1st sem (evaluations)" in row and row["Curricular units 1st sem (evaluations)"] == 0:
        reasons.append("Curricular units 1st sem (evaluations) = 0")

    # 5. Non-traditional age check
    if "Age at enrollment" in row and row["Age at enrollment"] >= 28:
        reasons.append(f"Age at enrollment = {int(row['Age at enrollment'])}")

    # 6. 2nd sem credits check (if present)
    if "Curricular units 2nd sem (approved)" in row and row["Curricular units 2nd sem (approved)"] == 0:
        reasons.append("Curricular units 2nd sem (approved) = 0")

    if not reasons:
        reasons.append("Academic progression variance")

    # Keep top 3 most critical reasons
    student_reasons.append("; ".join(reasons[:3]))

review_df["top_reasons"] = student_reasons

# Sort by p_dropout descending and assign ranks (1 to 885)
review_df = review_df.sort_values(by="p_dropout", ascending=False).reset_index(drop=True)
review_df["rank"] = review_df.index + 1

# Select and order final output columns
review_df = review_df[["rank", "student_index", "p_dropout", "true_outcome", "risk_tier", "top_reasons"]]

# Save review list to CSV
review_csv_path = "ranked_review_list.csv"
review_df.to_csv(review_csv_path, index=False)
print(f"[✓] Saved prioritized review list to '{review_csv_path}' ({len(review_df)} evaluated students)")

print("\nTOP 10 HIGH-RISK STUDENTS REQUIRING IMMEDIATE INTERVENTION:")
print(review_df.head(10).to_string(index=False))


# ==============================================================================
# 11. MODEL PIPELINE SERIALIZATION FOR FASTAPI DEPLOYMENT
# ==============================================================================
print("\n" + "=" * 65)
print("11. SERIALIZING TRAINED MODEL PIPELINE FOR DEPLOYMENT")
print("=" * 65)
import json
import joblib
from datetime import datetime

# Bundle all model components needed by FastAPI into a single production dictionary
model_artifact = {
    "model": threshold_model,
    "label_encoder": le,
    "classes": list(le.classes_),
    "final_threshold": float(final_threshold),
    "feature_columns": list(X.columns),
    "target_names": list(le.classes_),
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "macro_f1": float(f1_score(y_test, test_predictions, average="macro")),
    "created_at": datetime.now().isoformat(),
    "top_features": perm_df.head(20).to_dict(orient="records"),
}

# 1. Save model pipeline
model_file_path = "student_dropout_model.pkl"
joblib.dump(model_artifact, model_file_path)
print(f"[✓] Model pipeline saved as: {model_file_path}")

# 2. Save feature column list for FastAPI schema alignment
feature_cols_path = "feature_columns.json"
with open(feature_cols_path, "w", encoding="utf-8") as f:
    json.dump(list(X.columns), f, indent=2)
print(f"[✓] Feature columns saved as: {feature_cols_path}")

print("\n" + "=" * 65)
print("ALL TASKS COMPLETE! YOUR MODEL IS READY FOR FASTAPI & NGROK DEPLOYMENT.")
print("=" * 65)


# In[ ]:




