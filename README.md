# Tire Defects Detection & Analysis

A comprehensive machine learning pipeline for detecting tire defects using **XGBoost** classification with **SHAP explainability**. This project combines advanced gradient boosting with interpretable AI to identify defective tires and explain the factors driving predictions.

## 📋 Overview

This project builds a binary classification model to predict tire defects from tabular manufacturing data. It handles class imbalance, provides detailed model evaluation metrics, and includes SHAP (SHapley Additive exPlanations) for both global and local feature importance analysis.

### Key Features

- **🎯 Binary Classification**: Predicts tire defects (OK vs. Defect)
- **⚖️ Class Imbalance Handling**: Uses `scale_pos_weight` to balance skewed defect rates
- **📊 Comprehensive Preprocessing**: 
  - Median imputation for numeric features
  - One-hot encoding for categorical features
  - Automated feature type detection
- **🚀 Gradient Boosting**: XGBoost with early stopping for optimal performance
- **📈 Detailed Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC and Precision-Recall AUC
  - Confusion matrices with visualizations
  - Classification reports
- **🔍 SHAP Explainability**:
  - **Global explanations**: Summary plots showing overall feature impact
  - **Feature importance**: Mean absolute SHAP values ranking
  - **Local explanations**: Waterfall plots explaining individual predictions
- **🎚️ Threshold Tuning**: Analyze precision-recall tradeoffs at different decision thresholds
- **💾 Model Persistence**: Save trained models and preprocessors for production use

## 📁 Project Structure

```
├── Tire Defects Detection.ipynb    # Main analysis notebook (Colab-ready)
├── Tire Defects Detection.py       # Python script version
├── train_model.py                  # Model training script
├── fix_notebook.py                 # Notebook utilities
├── dashboard/                      # Interactive dashboard
│   ├── app.py
│   ├── style.css
│   ├── utils.py
│   └── data/
│       ├── feature_importance.csv
│       ├── metrics.json
│       ├── training_summary.json
│       └── images/
├── tires_train.csv                 # Training dataset
├── tires_validation.csv            # Validation dataset
├── tires_test.csv                  # Test dataset
├── tires_12000_full.csv            # Full dataset (12k samples)
├── preprocess.joblib               # Saved preprocessing pipeline
├── xgb_tire_defect_model.joblib    # Saved trained model
├── requirements.txt                # Python dependencies
├── run_dashboard.bat               # Dashboard launcher
├── Tire Defects Detection.html     # Exported notebook (HTML)
└── Backup Data/                    # Dataset backups
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Jupyter Notebook or Google Colab
- Required packages: `xgboost`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `shap`, `joblib`

### Installation

```bash
pip install -r requirements.txt
```

### Running the Pipeline

#### Option 1: Jupyter Notebook (Local)
```bash
jupyter notebook "Tire Defects Detection.ipynb"
```

#### Option 2: Google Colab
1. Upload the notebook to Google Colab
2. Upload the CSV files (tires_train.csv, tires_validation.csv, tires_test.csv)
3. Run all cells sequentially

#### Option 3: Python Script
```bash
python "Tire Defects Detection.py"
```

#### Option 4: Dashboard
```bash
python dashboard/app.py
```

## 📊 Pipeline Steps

### 1. **Data Loading**
- Loads train/validation/test CSV files
- Displays dataset dimensions and target distribution
- Handles missing files with error reporting

### 2. **Feature Engineering**
- Automatically separates numeric and categorical columns
- Excludes non-predictive columns: `TireProductionID`, `timestamp`, `defect_type`, `defect_probability_true`
- Creates X (features) and y (target) splits

### 3. **Preprocessing Pipeline**
- **Numeric features**: Median imputation
- **Categorical features**: Most-frequent imputation + one-hot encoding
- Scikit-learn `ColumnTransformer` for reproducibility

### 4. **Model Training**
- **Algorithm**: XGBoost (Gradient Boosting Trees)
- **Hyperparameters**:
  - 2000 estimators with early stopping (50 rounds patience)
  - Learning rate: 0.05
  - Max depth: 5
  - Subsample: 0.85
  - Colsample by tree: 0.85
  - L2 regularization (lambda=1.0)
- **Class Imbalance**: Automatic `scale_pos_weight` calculation

### 5. **Evaluation**
Metrics computed on validation and test sets:
- **Accuracy**: Overall correctness
- **Precision**: True defects among predicted defects
- **Recall**: Defects caught by the model
- **F1-Score**: Harmonic mean of precision & recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **PR-AUC**: Area under the precision-recall curve

### 6. **SHAP Explainability**
- **Summary Plot (Beeswarm)**: Each point represents one sample's SHAP value
- **Feature Importance (Bar Plot)**: Mean absolute SHAP values
- **Waterfall Plot**: Explains how individual features push a prediction toward/away from defect class

### 7. **Threshold Tuning**
- Evaluates model performance across different decision thresholds (0.2 to 0.7)
- Helps optimize for business requirements (catch defects vs. minimize false alarms)

### 8. **Model Persistence**
- Saves preprocessing pipeline to `preprocess.joblib`
- Saves trained XGBoost model to `xgb_tire_defect_model.joblib`

## 📈 Expected Output

### Console Output
```
✓ Data loaded successfully
Train: (9000, 45) | Val: (2000, 45) | Test: (1000, 45)

📊 Model features: 40
Target distribution (train): {0: 8100, 1: 900}
Defect rate (train): 10.00%

⚖️ Train defect rate: 0.1000 → scale_pos_weight: 9.00
🚀 Training model...
✓ Training complete. Best iteration: 487

📈 [Validation Set Evaluation]
Accuracy : 0.9450
Precision: 0.8234
Recall   : 0.7856
F1 Score : 0.8041
ROC-AUC  : 0.9123
PR-AUC   : 0.8567
```

### Visualizations
1. Confusion matrices (train, val, test)
2. SHAP summary plot (global feature impact)
3. SHAP feature importance bar chart
4. SHAP waterfall plot (single prediction explanation)

## 🔧 Configuration

Edit these variables in the notebook to customize behavior:

- `DATA_DIR`: Directory containing CSV files (default: `./`)
- `TARGET_COL`: Target column name (default: `defect_flag`)
- `RANDOM_STATE`: Random seed for reproducibility (default: `42`)
- `SAMPLE_N`: Number of samples for SHAP analysis (default: `1500`)
- XGBoost hyperparameters in the model initialization section

## 💡 Tips for Threshold Selection

The model's default threshold is 0.5, but business requirements may suggest alternatives:

- **Lower threshold (0.2–0.3)**: Maximize recall, catch more defects (higher false positives)
- **Higher threshold (0.6–0.7)**: Maximize precision, fewer false alarms (miss some defects)
- **Default (0.5)**: Balanced F1-score

## 📦 Dependencies

- `xgboost`: Gradient boosting framework
- `scikit-learn`: Preprocessing and metrics
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `matplotlib` & `seaborn`: Visualization
- `shap`: Model explainability
- `joblib`: Model serialization

## 🎓 Key Concepts

- **Imbalanced Classification**: Handles skewed class distributions using weighted loss
- **Early Stopping**: Prevents overfitting by monitoring validation performance
- **SHAP Values**: Explains model predictions using cooperative game theory
- **One-Hot Encoding**: Converts categorical variables to numeric format
- **Cross-validation**: Separates train/val/test to ensure unbiased evaluation

## 📝 License & Attribution

This project demonstrates best practices in interpretable machine learning for manufacturing quality control.

## ✅ Checklist for Running

- [ ] All CSV files present (tires_train.csv, tires_validation.csv, tires_test.csv)
- [ ] Python environment has required packages
- [ ] Random seed set for reproducibility
- [ ] Model training completes with convergence
- [ ] Test set evaluation shows reasonable metrics (>85% accuracy expected)
- [ ] SHAP plots generated without errors
- [ ] Model artifacts saved (preprocess.joblib, xgb_tire_defect_model.joblib)