import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import utils

# --- Page Config ---
st.set_page_config(
    page_title="Tire Defect Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load CSS ---
with open(os.path.join(os.path.dirname(__file__), "style.css"), "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("📈 Tire Analytics")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Notebook Analysis", "Dashboard Overview", "Live Prediction", "Model Explainability"])

# --- Load Models ---
# Only load models if we are NOT on the notebook page to save time/resources (optional optimization)
if page != "Notebook Analysis":
    model, preprocessor = utils.load_model_and_preprocessor()
    if model is None or preprocessor is None:
        st.stop()

# --- Main App ---

if page == "Notebook Analysis":
    st.title("📓 Model Training Report")
    st.markdown("This page provides a detailed breakdown of the model training pipeline, replicating the analysis from the Jupyter Notebook.")
    
    # Load Summary Data
    summary_path = os.path.join(os.path.dirname(__file__), "data", "training_summary.json")
    if not os.path.exists(summary_path):
        st.error("Training summary not found. Please run 'run_dashboard.bat' to generate results.")
        st.stop()
        
    import json
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    secs = summary["sections"]
    img_dir = os.path.join(os.path.dirname(__file__), "data", "images")
    
    # 1. Environment
    with st.expander("1. Environment Setup", expanded=False):
        st.success(f"Status: {secs['1_environment']['status']}")
        st.write(f"**Libraries Loaded:** {', '.join(secs['1_environment']['libraries'])}")
        st.info("**In simple terms:** The system prepares all tools needed to train, evaluate, and explain the model.")

    # 2. Data Loading
    with st.expander("2. Data Loading & Validation", expanded=True):
        st.success(secs['2_data_loading']['status'])
        shapes = secs['2_data_loading']['shapes']
        c1, c2, c3 = st.columns(3)
        c1.metric("Training Rows", shapes['train'][0])
        c2.metric("Validation Rows", shapes['val'][0])
        c3.metric("Test Rows", shapes['test'][0])
        st.info("**In simple terms:** The system confirms that the tire data is available and correctly split for learning and testing.")

    # 3. Data Overview
    with st.expander("3. Data Overview & Feature Inspection", expanded=True):
        data = secs['3_data_overview']
        c1, c2 = st.columns(2)
        c1.metric("Total Features", data['num_features'])
        c1.metric("Defect Rate", f"{data['defect_rate']:.2%}")
        c2.write("Target Distribution:")
        c2.json(data['target_dist'])
        st.write(f"**Categorical Columns ({len(data['cat_cols'])}):** {', '.join(data['cat_cols'])}")
        st.info("**In simple terms:** You get a quick understanding of what data the model will learn from and how common defects are.")

    # 4. Class Imbalance
    with st.expander("4. Class Imbalance Handling", expanded=False):
        imb = secs['4_class_imbalance']
        c1, c2 = st.columns(2)
        c1.metric("Defect Rate", imb['defect_rate_percent'])
        c2.metric("Scale Pos Weight", imb['scale_pos_weight'])
        st.info("**In simple terms:** The model is told to pay extra attention to rare defective tires so they are not ignored.")

    # 5. Preprocessing
    with st.expander("5. Data Preprocessing & Encoding", expanded=False):
        st.metric("Final Encoded Feature Count", secs['5_preprocessing']['encoded_features_count'])
        st.info("**In simple terms:** The raw tire data is cleaned and converted into a format the model can understand.")

    # 6. Training
    with st.expander("6. Model Training (XGBoost)", expanded=True):
        train = secs['6_training']
        st.success(train['status'])
        st.metric("Best Iteration", train['best_iteration'])
        st.info("**In simple terms:** The model learns patterns that distinguish defective tires while stopping before overfitting.")

    # 7. Evaluation
    with st.expander("7. Model Evaluation", expanded=True):
        eval_data = secs['7_evaluation']
        metrics = eval_data['test_metrics']
        
        st.subheader("Test Set Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall", f"{metrics['recall']:.2%}")
        c4.metric("F1 Score", f"{metrics['f1']:.2%}")
        
        st.subheader("Confusion Matrices")
        cm1, cm2 = st.columns(2)
        cm1.image(os.path.join(img_dir, "cm_validation.png"), caption="Validation Set")
        cm2.image(os.path.join(img_dir, "cm_test.png"), caption="Test Set")
        
        st.info("**In simple terms:** You see how accurate the model is, how well it finds defects, and how many mistakes it makes.")

    # 8. Global Explainability
    with st.expander("8. Global Explainability (SHAP)", expanded=True):
        st.image(os.path.join(img_dir, "shap_summary.png"), caption="Feature Impact Summary")
        st.image(os.path.join(img_dir, "shap_bar.png"), caption="Mean Feature Importance")
        st.info("**In simple terms:** You learn which tire characteristics matter most when predicting defects.")

    # 9. Local Explainability
    with st.expander("9. Local Explainability (Example)", expanded=False):
        local = secs['9_local_shap']
        c1, c2 = st.columns(2)
        c1.metric("Predicted Prob", f"{local['predicted_prob']:.2%}")
        c2.metric("Predicted Class", "Defect" if local['predicted_class'] == 1 else "OK")
        st.image(os.path.join(img_dir, "shap_waterfall.png"), caption=f"Waterfall Plot for Sample #{local['sample_index']}")
        st.info("**In simple terms:** You can explain why the model classified one specific tire as defective or not.")

    # 10. Model Saving
    with st.expander("10. Model Saving", expanded=False):
        st.success("Models saved successfully:")
        st.write(secs['10_model_saving']['files'])
        st.info("**In simple terms:** The trained system is saved so it can be reused without retraining.")

    # 11. Threshold Tuning
    with st.expander("11. Threshold Tuning Analysis", expanded=True):
        tuning = secs['11_threshold_tuning']
        df_tune = pd.DataFrame(tuning)
        st.dataframe(df_tune.style.highlight_max(axis=0, subset=['F1']), use_container_width=True)
        st.info("**In simple terms:** You can adjust how strict the model is depending on whether missing defects or false alarms is more costly.")

elif page == "Dashboard Overview":
    st.title("🏭 Production Quality Overview")
    
    # Load data
    default_data_path = os.path.join(os.path.dirname(__file__), "..", "tires_12000_full.csv")
    if os.path.exists(default_data_path):
        # Load with timestamp parsing
        df = pd.read_csv(default_data_path, parse_dates=['timestamp'])
    else:
        st.warning("Default dataset 'tires_12000_full.csv' not found. Please upload data in 'Live Prediction'.")
        st.stop()

    # --- Filters ---
    st.sidebar.subheader("Filter Data")
    
    # Date Range Filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    start_date, end_date = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Machine Filter
    if 'machine_id' in df.columns:
        machines = ['All'] + sorted(df['machine_id'].unique().tolist())
        selected_machine = st.sidebar.selectbox("Select Machine", machines)
    else:
        selected_machine = 'All'
        
    # Shift Filter
    if 'shift' in df.columns:
        shifts = ['All'] + sorted(df['shift'].unique().tolist())
        selected_shift = st.sidebar.selectbox("Select Shift", shifts)
    else:
        selected_shift = 'All'

    # Apply Filters
    mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
    if selected_machine != 'All':
        mask &= (df['machine_id'] == selected_machine)
    if selected_shift != 'All':
        mask &= (df['shift'] == selected_shift)
        
    df_filtered = df[mask]
    
    if len(df_filtered) == 0:
        st.error("No data available for the selected filters.")
        st.stop()

    # --- Metrics ---
    st.subheader(f"Metrics ({len(df_filtered)} tires filtered)")
    col1, col2, col3, col4 = st.columns(4)
    
    total_tires = len(df_filtered)
    defects = df_filtered[df_filtered['defect_flag'] == 1]
    defect_rate = (len(defects) / total_tires) * 100 if total_tires > 0 else 0
    
    col1.metric("Total Tires", f"{total_tires:,}")
    col2.metric("Defects", f"{len(defects):,}")
    col3.metric("Defect Rate", f"{defect_rate:.2f}%", delta_color="inverse")
    
    st.markdown("---")
    
    # --- Advanced Charts ---
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📈 Defect Trend Over Time")
        # Daily Defect Rate
        daily_stats = df_filtered.set_index('timestamp').resample('D')['defect_flag'].agg(['mean', 'count']).reset_index()
        daily_stats['defect_rate_pct'] = daily_stats['mean'] * 100
        # Rolling 7-day average for smoother trend
        daily_stats['Rolling 7D Avg'] = daily_stats['defect_rate_pct'].rolling(window=7, min_periods=1).mean()
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(x=daily_stats['timestamp'], y=daily_stats['defect_rate_pct'], mode='markers', name='Daily Rate', marker=dict(color='gray', opacity=0.5)))
        fig_trend.add_trace(go.Scatter(x=daily_stats['timestamp'], y=daily_stats['Rolling 7D Avg'], mode='lines', name='7-Day Trend', line=dict(color='#ff4b4b', width=3)))
        fig_trend.update_layout(title="Daily vs 7-Day Moving Average Defect Rate (%)", hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.subheader("🔥 Machine vs Shift Heatmap")
        if 'machine_id' in df_filtered.columns and 'shift' in df_filtered.columns:
            # Pivot table for heatmap
            heatmap_data = df_filtered.pivot_table(index='machine_id', columns='shift', values='defect_flag', aggfunc='mean') * 100
            
            import plotly.express as px
            fig_hm = px.imshow(heatmap_data, 
                               labels=dict(x="Shift", y="Machine", color="Defect Rate (%)"),
                               color_continuous_scale="RdYlGn_r",
                               text_auto=".1f",
                               title="Defect Concentration Map")
            st.plotly_chart(fig_hm, use_container_width=True)
    
    # Legacy Charts (if needed, kept below or modified)
    st.subheader("📉 Categorical Breakdown")
    cc1, cc2 = st.columns(2)
    with cc1:
        if 'machine_id' in df_filtered.columns:
            fig_machine = utils.plot_defect_rate_by_category(df_filtered, 'machine_id', target_col='defect_flag')
            st.plotly_chart(fig_machine, use_container_width=True)
    with cc2:
        if 'ambient_temp_c' in df_filtered.columns:
             # Binning temp
            df_filtered['temp_bin'] = pd.cut(df_filtered['ambient_temp_c'], bins=10)
            temp_stats = df_filtered.groupby('temp_bin')['defect_flag'].mean().reset_index()
            temp_stats['temp_bin'] = temp_stats['temp_bin'].astype(str)
            fig_temp = px.line(temp_stats, x='temp_bin', y='defect_flag', title="Visual Defect Rate vs Temp")
            st.plotly_chart(fig_temp, use_container_width=True)

elif page == "Live Prediction":
    st.title("🔍 Batch Prediction Analysis")
    
    uploaded_file = st.file_uploader("Upload Batch CSV", type="csv")
    
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_new)} tires.")
        
        if st.button("Run Diagnostics"):
            with st.spinner("Analyzing..."):
                X, X_enc = utils.preprocess_data(df_new, preprocessor)
                probs = utils.run_prediction(model, X_enc)
                
                # Threshold via slider
                threshold = st.slider("Defect Probability Threshold", 0.0, 1.0, 0.5, 0.05)
                
                df_new['Defect_Probability'] = probs
                df_new['Prediction'] = (probs >= threshold).astype(int)
                
                # Highlight High Risk
                st.subheader("⚠️ High Risk Alerts")
                st.dataframe(
                    df_new[df_new['Prediction'] == 1].style.background_gradient(subset=['Defect_Probability'], cmap='Reds'),
                    use_container_width=True
                )
                
                # Download
                csv = df_new.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv, "predictions.csv", "text/csv")

elif page == "Model Explainability":
    st.title("🧠 Why was this flagged?")
    st.info("Upload a file in 'Live Prediction' first, or use default data below.")
    
    # Re-load default for simplicity if nothing in session state
    default_data_path = os.path.join(os.path.dirname(__file__), "..", "tires_test.csv")
    
    if os.path.exists(default_data_path):
        df_exp = pd.read_csv(default_data_path)
        
        # Preprocess
        X, X_enc = utils.preprocess_data(df_exp, preprocessor)
        
        # Select Index
        tire_idx = st.number_input("Select Tire Index to Explain", min_value=0, max_value=len(df_exp)-1, value=0)
        
        # Show Row Data
        st.write("Raw Data:", df_exp.iloc[[tire_idx]])
        
        # Plot SHAP
        if st.button("Generate Explanation"):
            with st.spinner("Calculating SHAP values..."):
                # We need column names for better UX
                import shap
                import matplotlib.pyplot as plt
                
                # Explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_enc)
                
                # Create plots
                st.subheader(f"Feature Impact for Tire #{tire_idx}")
                
                # Waterfall (local)
                fig, ax = plt.subplots()
                
                # Note: We need feature names ideally. 
                # Attempt to get them from preprocessor if possible, else use generic
                # feature_names = utils.get_feature_names(preprocessor) # TODO implement robustly if needed
                
                shap.summary_plot(shap_values, X_enc, max_display=15, plot_type="bar", show=False)
                st.pyplot(plt.gcf())
                
                st.write("**Top contributing factors:** The bars above show which features pushed the prediction towards 'Defect' (positive) or 'Safe' (negative).")
    else:
        st.error("No data found for explanation.")
