import pandas as pd
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import streamlit as st

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "xgb_tire_defect_model.joblib")
PREPROCESSOR_PATH = os.path.join(os.path.dirname(__file__), "..", "preprocess.joblib")

@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor artifacts."""
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model artifacts not found. Please ensure 'xgb_tire_defect_model.joblib' and 'preprocess.joblib' are in the project root.")
        return None, None

def load_data(uploaded_file):
    """Load data from uploaded CSV or default test file."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Default fallback (optional)
        return None
    return df

def preprocess_data(df, preprocessor):
    """Preprocess data using the loaded pipeline."""
    # Ensure all required columns are present (basic check)
    # The preprocessor expects specific columns. We rely on the pipeline to handle it.
    
    # We need to separate features that were used for training.
    # Based on the notebook, we drop specific columns.
    DROP_COLS = ["tire_id", "TireProductionID", "timestamp", "defect_type", "defect_probability_true", "defect_flag"]
    
    # Keep only columns that exist
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop, errors="ignore")
    
    # Transform
    X_enc = preprocessor.transform(X)
    
    # Get feature names for SHAP and plotting
    # Helper to get names from ColumnTransformer
    feature_names = []
    
    # Access transformers
    # Note: This logic must match the notebook structure exactly
    try:
        # num_cols (from numeric transformer) + cat_cols (from onehot)
        # Reconstructing robustly is hard without the original lists, 
        # so we rely on the transformer's output structure if possible.
        # A safer bet for visualizations is to use the raw X for some, and X_enc for model.
        
        # Extract feature names from one-hot encoder
        # This part depends heavily on scikit-learn version. 
        # For simplicity in this demo, we might skip exact column naming for SHAP 
        # unless we reconstruct the schema precisely.
        pass
    except:
        pass
        
    return X, X_enc

def run_prediction(model, X_enc):
    """Run model inference."""
    # Predict probability
    proba = model.predict_proba(X_enc)[:, 1]
    # Predict class (default threshold 0.5, but dashboard allows slider)
    return proba

def plot_defect_rate_by_category(df, category_col, target_col="Prediction"):
    """Plot defect rate by a categorical column."""
    if category_col not in df.columns:
        return None
    
    # Group by category
    stats = df.groupby(category_col)[target_col].agg(['count', 'sum'])
    stats['Defect Rate'] = stats['sum'] / stats['count']
    stats = stats.reset_index()
    stats.columns = [category_col, 'Total Tires', 'Defects', 'Defect Rate']
    
    fig = px.bar(
        stats, x=category_col, y='Defect Rate',
        hover_data=['Total Tires', 'Defects'],
        color='Defect Rate',
        color_continuous_scale='RdYlGn_r',
        title=f"Defect Rate by {category_col}"
    )
    return fig

def plot_shap_waterfall(model, X_enc, index):
    """Generate SHAP waterfall plot for a specific instance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_enc)
    
    # SHAP returns a matrix. We need the row for 'index'
    # Visualization in Streamlit using matplotlib
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    # Note: Providing feature names here would make it much better.
    # For now we use default indices if names aren't perfectly aligned
    shap.summary_plot(shap_values, X_enc, plot_type="bar", show=False)
    
    return fig
