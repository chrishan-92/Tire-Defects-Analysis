# ============================================================
# Tire Defect Detection - Training Script
# Optimized for Dashboard Integration (Extended Version)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import shap
import joblib

# Configuration
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATA_DIR = "."
OUTPUT_DIR = os.path.join("dashboard", "data")
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

# Dictionary to hold all summary data
summary_data = {
    "sections": {}
}

print("Starting training pipeline...")

# -------------------------
# 1) Environment Setup (Metadata)
# -------------------------
summary_data["sections"]["1_environment"] = {
    "libraries": ["xgboost", "shap", "sklearn", "pandas", "matplotlib", "seaborn", "joblib"],
    "status": "Success"
}

# -------------------------
# 2) Data Loading
# -------------------------
try:
    train_df = pd.read_csv(os.path.join(DATA_DIR, "tires_train.csv"))
    val_df   = pd.read_csv(os.path.join(DATA_DIR, "tires_validation.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "tires_test.csv"))
    
    summary_data["sections"]["2_data_loading"] = {
        "status": "Data loaded successfully",
        "shapes": {
            "train": train_df.shape,
            "val": val_df.shape,
            "test": test_df.shape
        }
    }
    print(f"Data loaded: Train {train_df.shape}, Val {val_df.shape}, Test {test_df.shape}")
except FileNotFoundError:
    print("Error: Data files not found.")
    exit(1)

# -------------------------
# 3) Data Overview
# -------------------------
TARGET_COL = "defect_flag"
DROP_COLS = ["tire_id", "TireProductionID", "timestamp", "defect_type", "defect_probability_true"]

def split_xy(df):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target {TARGET_COL} not found")
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns], errors="ignore")
    return X, y

X_train, y_train = split_xy(train_df)
X_val, y_val = split_xy(val_df)
X_test, y_test = split_xy(test_df)

cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

summary_data["sections"]["3_data_overview"] = {
    "num_features": X_train.shape[1],
    "target_dist": y_train.value_counts().to_dict(),
    "defect_rate": float(y_train.mean()),
    "cat_cols": cat_cols,
    "num_cols": num_cols,
    "preview_columns": list(X_train.columns)
}

# -------------------------
# 4) Class Imbalance
# -------------------------
pos_rate = y_train.mean()
scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-9)

summary_data["sections"]["4_class_imbalance"] = {
    "defect_rate_percent": f"{pos_rate:.2%}",
    "scale_pos_weight": f"{scale_pos_weight:.2f}"
}

# -------------------------
# 5) Preprocessing
# -------------------------
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

preprocess.fit(X_train)
X_train_enc = preprocess.transform(X_train)
X_val_enc = preprocess.transform(X_val)
X_test_enc = preprocess.transform(X_test)

summary_data["sections"]["5_preprocessing"] = {
    "encoded_features_count": X_train_enc.shape[1]
}

# -------------------------
# 6) Model Training
# -------------------------
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.0,
    min_child_weight=1.0,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=50
)

print("Training model...")
xgb.fit(
    X_train_enc, y_train,
    eval_set=[(X_train_enc, y_train), (X_val_enc, y_val)],
    verbose=False
)

summary_data["sections"]["6_training"] = {
    "best_iteration": int(xgb.best_iteration),
    "status": "Training with early stopping complete."
}

# -------------------------
# 7) Evaluation
# -------------------------
def save_confusion_matrix(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    path = os.path.join(IMG_DIR, f"cm_{name.lower()}.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path

val_proba = xgb.predict_proba(X_val_enc)[:, 1]
val_preds = (val_proba >= 0.5).astype(int)
test_proba = xgb.predict_proba(X_test_enc)[:, 1]
test_preds = (test_proba >= 0.5).astype(int)

# Classification Report
clf_report = classification_report(test_preds, y_test, output_dict=True)

# Save CM images
cm_val_path = save_confusion_matrix(y_val, val_preds, "Validation")
cm_test_path = save_confusion_matrix(y_test, test_preds, "Test")

summary_data["sections"]["7_evaluation"] = {
    "test_metrics": {
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "pr_auc": float(average_precision_score(y_test, test_proba))
    },
    "classification_report": clf_report
    # Images are standard paths
}

# -------------------------
# 8) Global SHAP
# -------------------------
def get_feature_names(preprocessor):
    feature_names = []
    feature_names.extend(num_cols)
    if len(cat_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(ohe_names)
    return feature_names

feature_names = get_feature_names(preprocess)

# Use a subsample for speed
SAMPLE_N = 1000
sample_idx = np.random.choice(len(X_train_enc), size=min(SAMPLE_N, len(X_train_enc)), replace=False)
X_shap = X_train_enc[sample_idx]
X_shap_df = pd.DataFrame(X_shap, columns=feature_names)

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_shap_df)

# Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap_df, show=False, max_display=15)
plt.title("SHAP Feature Importance (Global)")
plt.savefig(os.path.join(IMG_DIR, "shap_summary.png"), bbox_inches='tight')
plt.close()

# Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap_df, plot_type="bar", show=False, max_display=15)
plt.title("Mean |SHAP| Value")
plt.savefig(os.path.join(IMG_DIR, "shap_bar.png"), bbox_inches='tight')
plt.close()

summary_data["sections"]["8_global_shap"] = {
    "status": "Generated summary and bar plots"
}

# -------------------------
# 9) Local SHAP
# -------------------------
# Pick a positive case if possible
pos_indices = np.where(test_preds == 1)[0]
if len(pos_indices) > 0:
    idx = pos_indices[0]
else:
    idx = 0

x_one = pd.DataFrame(X_test_enc[idx:idx+1], columns=feature_names)
sv_one = explainer.shap_values(x_one)

plt.figure(figsize=(8, 6))
# Try-catch for waterfall as APIs vary
try:
    shap.plots.waterfall(
        shap.Explanation(values=sv_one[0], base_values=explainer.expected_value, data=x_one.iloc[0], feature_names=feature_names),
        show=False
    )
except:
    pass # fallback or older matplotlib behavior might just work on current figure

plt.savefig(os.path.join(IMG_DIR, "shap_waterfall.png"), bbox_inches='tight')
plt.close()

summary_data["sections"]["9_local_shap"] = {
    "sample_index": int(idx),
    "predicted_prob": float(test_proba[idx]),
    "predicted_class": int(test_preds[idx]),
    "actual_class": int(y_test.iloc[idx])
}

# -------------------------
# 10) Model Saving
# -------------------------
joblib.dump(preprocess, "preprocess.joblib")
joblib.dump(xgb, "xgb_tire_defect_model.joblib")
summary_data["sections"]["10_model_saving"] = {
    "files": ["preprocess.joblib", "xgb_tire_defect_model.joblib"]
}

# -------------------------
# 11) Threshold Tuning
# -------------------------
thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
tuning_results = []
for t in thresholds:
    preds_t = (val_proba >= t).astype(int)
    tuning_results.append({
        "Threshold": t,
        "Precision": float(precision_score(y_val, preds_t, zero_division=0)),
        "Recall": float(recall_score(y_val, preds_t, zero_division=0)),
        "F1": float(f1_score(y_val, preds_t, zero_division=0))
    })

summary_data["sections"]["11_threshold_tuning"] = tuning_results

# Save Summary
with open(os.path.join(OUTPUT_DIR, "training_summary.json"), "w") as f:
    json.dump(summary_data, f, indent=4)

print("Pipeline finished. Summary saved to dashboard/data/training_summary.json")
