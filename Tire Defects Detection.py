#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================
# Tire Defect Detection (Tabular ML) with SHAP Explainability
# Google Colab-ready single notebook script
# ============================================================
# What this does:
# 1) Loads train/validation/test CSVs
# 2) Builds a preprocessing pipeline (impute + one-hot encode)
# 3) Trains an XGBoost classifier (Gradient Boosted Trees) with early stopping
# 4) Evaluates (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix)
# 5) Runs SHAP global + local explanations
#
# Files expected:
# - tires_train.csv
# - tires_validation.csv
# - tires_test.csv
# ============================================================

# -------------------------
# 0) Install dependencies
# -------------------------
get_ipython().system('pip -q install xgboost shap scikit-learn pandas matplotlib seaborn joblib')

import os
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
    roc_auc_score, average_precision_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)

from xgboost import XGBClassifier
import shap
import joblib

# For reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------------------------
# 1) Load data
# -------------------------
# Option A: If you uploaded files directly to Colab, use:
DATA_DIR = "./"

# Option B: If using Google Drive, uncomment:
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_DIR = "/content/drive/MyDrive/<YOUR_FOLDER>"

train_path = os.path.join(DATA_DIR, "tires_train.csv")
val_path   = os.path.join(DATA_DIR, "tires_validation.csv")
test_path  = os.path.join(DATA_DIR, "tires_test.csv")

# Load data with error handling
try:
    train_df = pd.read_csv(train_path)
    val_df   = pd.read_csv(val_path)
    test_df  = pd.read_csv(test_path)
    print(f"✓ Data loaded successfully")
    print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
except FileNotFoundError as e:
    print(f"❌ Error loading data: {e}")
    print("Please upload the CSV files to the correct directory.")
    raise

# -------------------------
# 2) Define target and features
# -------------------------
TARGET_COL = "defect_flag"

# Columns to exclude from modeling (IDs + leakage/testing helper)
DROP_COLS = [
    "TireProductionID",
    "timestamp",
    "defect_type",             # keep for analysis, not for binary training
    "defect_probability_true"  # synthetic helper column; must not be used in training
]

# Separate X/y
def split_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features (X) and target (y)."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe")

    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL] + [c for c in DROP_COLS if c in df.columns], errors="ignore")
    return X, y

X_train, y_train = split_xy(train_df)
X_val, y_val     = split_xy(val_df)
X_test, y_test   = split_xy(test_df)

print(f"\n📊 Model features: {X_train.shape[1]}")
print(f"Target distribution (train): {y_train.value_counts().to_dict()}")
print(f"Defect rate (train): {y_train.mean():.2%}")
print("\nSample of features:")
display(X_train.head())

# Identify categorical vs numeric columns
cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

print(f"\nNumeric columns: {len(num_cols)}")
print(f"Categorical columns: {len(cat_cols)}")
if cat_cols:
    print(f"  {cat_cols}")

# -------------------------
# 3) Preprocessing pipeline
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# -------------------------
# 4) Model (XGBoost: Gradient Boosting Trees)
# -------------------------
# Handle class imbalance
pos_rate = y_train.mean()
scale_pos_weight = (1 - pos_rate) / max(pos_rate, 1e-9)
print(f"\n⚖️ Train defect rate: {pos_rate:.4f} → scale_pos_weight: {scale_pos_weight:.2f}")

xgb = XGBClassifier(
    n_estimators=2000,          # use early stopping to find best iteration
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
    scale_pos_weight=scale_pos_weight
)

# -------------------------
# 5) Train with early stopping
# -------------------------
print("\n🚀 Training model...")

# Fit preprocess on train data only
preprocess.fit(X_train)
X_train_enc = preprocess.transform(X_train)
X_val_enc   = preprocess.transform(X_val)
X_test_enc  = preprocess.transform(X_test)

print(f"Encoded feature dimensions: {X_train_enc.shape[1]}")

# Fit classifier with early stopping
# Note: early_stopping_rounds is now passed to the constructor in newer XGBoost versions
xgb.set_params(early_stopping_rounds=50)
xgb.fit(
    X_train_enc, y_train,
    eval_set=[(X_train_enc, y_train), (X_val_enc, y_val)],
    verbose=50  # Show progress every 50 iterations
)

print(f"✓ Training complete. Best iteration: {xgb.best_iteration}")

# -------------------------
# 6) Evaluate
# -------------------------
def evaluate_split(name: str, X_enc, y_true, show_report: bool = False):
    """Evaluate model performance on a dataset split."""
    proba = xgb.predict_proba(X_enc)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc = roc_auc_score(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)

    print(f"\n{'='*50}")
    print(f"📈 [{name} Set Evaluation]")
    print(f"{'='*50}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    if show_report:
        print(f"\nClassification Report:")
        print(classification_report(y_true, preds, target_names=["OK", "Defect"]))

    # Confusion matrix
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["OK", "Defect"])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f"Confusion Matrix - {name} Set", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "roc_auc": roc, "pr_auc": pr_auc}

# Evaluate all splits
val_metrics = evaluate_split("Validation", X_val_enc, y_val, show_report=True)
test_metrics = evaluate_split("Test", X_test_enc, y_test, show_report=True)

# -------------------------
# 7) Feature names after encoding (needed for SHAP plots)
# -------------------------
def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """Extract feature names from ColumnTransformer after encoding."""
    feature_names = []

    # Numeric features
    feature_names.extend(num_cols)

    # Categorical features -> one-hot encoded names
    if len(cat_cols) > 0:
        ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        feature_names.extend(ohe_names)

    return feature_names

feature_names = get_feature_names(preprocess)
print(f"\n✓ Encoded feature count: {len(feature_names)}")

# Convert encoded matrices to DataFrame for nicer SHAP plots
X_train_enc_df = pd.DataFrame(X_train_enc, columns=feature_names)
X_val_enc_df   = pd.DataFrame(X_val_enc, columns=feature_names)
X_test_enc_df  = pd.DataFrame(X_test_enc, columns=feature_names)

# -------------------------
# 8) SHAP Explainability
# -------------------------
print("\n🔍 Computing SHAP explanations...")

# TreeExplainer works well for XGBoost
explainer = shap.TreeExplainer(xgb)

# For speed, sample a subset for SHAP global plots
SAMPLE_N = 1500
sample_size = min(SAMPLE_N, len(X_train_enc_df))
sample_idx = np.random.choice(len(X_train_enc_df), size=sample_size, replace=False)
X_shap = X_train_enc_df.iloc[sample_idx]

shap_values = explainer.shap_values(X_shap)
print(f"✓ SHAP values computed for {sample_size} samples")

# 8.1 Global explanation: summary plot (beeswarm)
print("\n📊 Generating SHAP visualizations...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, show=False, max_display=20)
plt.title("SHAP Summary Plot (Global Feature Impact)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 8.2 Global: bar plot (mean |SHAP|)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False, max_display=20)
plt.title("SHAP Feature Importance (Mean |SHAP|)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 8.3 Local explanation: explain a single test prediction
# Pick one predicted-defect example if possible; otherwise pick any example
test_proba = xgb.predict_proba(X_test_enc_df)[:, 1]
test_pred  = (test_proba >= 0.5).astype(int)

# Try to find a true positive or interesting case
if (test_pred == 1).any():
    idx = int(np.where(test_pred == 1)[0][0])
else:
    idx = 0

x_one = X_test_enc_df.iloc[[idx]]
sv_one = explainer.shap_values(x_one)

print(f"\n🔬 Local Explanation for Test Sample #{idx}")
print(f"   Predicted Probability: {test_proba[idx]:.4f}")
print(f"   Predicted Class: {'Defect' if test_pred[idx] == 1 else 'OK'}")
print(f"   True Label: {'Defect' if y_test.iloc[idx] == 1 else 'OK'}")

# Waterfall plot (newer SHAP API)
plt.figure(figsize=(10, 6))
try:
    shap.plots.waterfall(
        shap.Explanation(
            values=sv_one[0],
            base_values=explainer.expected_value,
            data=x_one.iloc[0],
            feature_names=feature_names
        )
    )
except Exception as e:
    # Fallback for older SHAP versions
    try:
        shap.waterfall_plot(
            shap.Explanation(
                values=sv_one[0],
                base_values=explainer.expected_value,
                data=x_one.iloc[0],
                feature_names=feature_names
            )
        )
    except Exception as e2:
        print(f"⚠️ Could not create waterfall plot: {e2}")

# -------------------------
# 9) Save artifacts (model + preprocess)
# -------------------------
print("\n💾 Saving model artifacts...")
joblib.dump(preprocess, "preprocess.joblib")
joblib.dump(xgb, "xgb_tire_defect_model.joblib")

print("✓ Saved:")
print("  - preprocess.joblib")
print("  - xgb_tire_defect_model.joblib")

# -------------------------
# 10) Optional: Threshold tuning (recommended for imbalanced data)
# -------------------------
print("\n🎯 Threshold Tuning Analysis")
print("="*50)

def evaluate_threshold(threshold: float = 0.5):
    """Evaluate model at a specific decision threshold."""
    proba = xgb.predict_proba(X_val_enc_df)[:, 1]
    preds = (proba >= threshold).astype(int)
    prec = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    return prec, rec, f1

print("Threshold | Precision | Recall | F1-Score")
print("-" * 50)
for t in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    prec, rec, f1 = evaluate_threshold(t)
    print(f"  {t:.2f}    |   {prec:.3f}   | {rec:.3f} |  {f1:.3f}")

print("\n💡 Tips for threshold selection:")
print("  • Lower threshold (0.2-0.3): Higher recall, catch more defects")
print("  • Higher threshold (0.6-0.7): Higher precision, fewer false alarms")
print("  • Consider business costs of false positives vs false negatives")

print("\n✅ Pipeline complete!")

