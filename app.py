# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import structlog
import os
import warnings
from config import Config
from data_utils import load_and_preprocess_data, MissingDataError, InvalidPressureError
from analysis import analyze_relationships
from optimization import optimize_target
import shap  # For shap plots

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap.explainers._deep.deep_tf")

# Setup logging
structlog.configure(
    processors=[structlog.processors.JSONRenderer()],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
logger = structlog.get_logger()

def main():
    st.title("Parameter Analysis and Optimization App")

    # Sidebar for configurations
    st.sidebar.header("Configuration")
    task = st.sidebar.selectbox("Select task", ["Relationship Analysis", "Target Optimization", "Both"])
    row_choice = st.sidebar.text_input("Process all rows or select a number of random rows? (Enter 'all' or a number)", "all")
    n_rows = None if row_choice.lower() == 'all' else int(row_choice)
    save_files = st.sidebar.checkbox("Save and download Excel files and plots?")
    bubble_point = st.sidebar.number_input("Bubble point pressure (default 2000 psi)", value=2000.0)
    optimizer = st.sidebar.selectbox("Optimizer (for optimization)", ["ga", "bayesian"]) if task in ["Target Optimization", "Both"] else None
    interactive = st.sidebar.checkbox("Enable interactive parameter exploration? (for optimization)") if task in ["Target Optimization", "Both"] else False

    # File upload
    uploaded_files = st.file_uploader("Upload your Excel files containing parameters like Bo, Rs, P, etc.", type=["xlsx"], accept_multiple_files=True)

    if st.button("Run Analysis"):
        config = Config()
        rng = np.random.default_rng(config.random_seed)
        
        try:
            df, params = load_and_preprocess_data(uploaded_files, config, n_rows, rng)
            if not params or len(df) < config.min_rows:
                raise MissingDataError(f"No valid data or insufficient rows ({len(df)} < {config.min_rows})")
            
            save_dir = 'output'
            if save_files:
                os.makedirs(save_dir, exist_ok=True)
            
            all_figures = []
            
            if task in ["Relationship Analysis", "Both"]:
                st.header("Relationship Analysis")
                figs = analyze_relationships(df, params, bubble_point, config)
                all_figures.extend(figs)
                # Save matrices if save_files
                if save_files:
                    corr_matrix = df[params].corr(method='spearman')
                    corr_matrix.to_excel(f'{save_dir}/correlation_matrix.xlsx', index=True)
                    with open(f'{save_dir}/correlation_matrix.xlsx', 'rb') as f:
                        st.download_button("Download Correlation Matrix", f, file_name="correlation_matrix.xlsx")
                    # Similarly for others...
            
            if task in ["Target Optimization", "Both"]:
                st.header("Target Optimization")
                target_name = st.selectbox("Select target parameter", params)
                if target_name:
                    if df[target_name].isna().any():
                        st.error(f"Target {target_name} contains NaN values. Please choose another.")
                    else:
                        progress_bar = st.progress(0)
                        optimal_df, figs, shap_data = optimize_target(df, params, target_name, config, optimizer, progress_bar)
                        all_figures.extend(figs)
                        st.write("Optimal Parameters:", optimal_df)
                        if save_files:
                            optimal_df.to_excel(f'{save_dir}/optimal_{target_name}_{optimizer}.xlsx', index=False)
                            with open(f'{save_dir}/optimal_{target_name}_{optimizer}.xlsx', 'rb') as f:
                                st.download_button(f"Download Optimal {target_name}", f, file_name=f"optimal_{target_name}_{optimizer}.xlsx")
                        if interactive and shap_data:
                            st.header("Interactive Parameter Exploration")
                            # Implement sliders for parameters
                            sliders = {}
                            for param in [p for p in params if p != target_name]:
                                min_val, max_val = df[param].min(), df[param].max()
                                sliders[param] = st.slider(param, float(min_val), float(max_val), float(df[param].mean()))
                            if sliders:
                                model, scaler = train_neural_network(df[[p for p in params if p != target_name]], df[target_name], config)
                                X_test = pd.DataFrame([sliders])
                                X_test_scaled = scaler.transform(X_test)
                                prediction = predict_nn(model, X_test_scaled)[0][0]
                                st.write(f"Predicted {target_name}: {prediction}")
            
            # Display figures
            for i, fig in enumerate(all_figures):
                if isinstance(fig, plt.Figure):
                    st.pyplot(fig)
                elif isinstance(fig, go.Figure):
                    st.plotly_chart(fig)
                if save_files:
                    if isinstance(fig, plt.Figure):
                        fig.savefig(f'{save_dir}/plot_{i}.png', dpi=150)
                    else:
                        fig.write_image(f'{save_dir}/plot_{i}.png', scale=2)
                    with open(f'{save_dir}/plot_{i}.png', 'rb') as f:
                        st.download_button(f"Download Plot {i}", f, file_name=f"plot_{i}.png")
            
            st.success("Analysis complete.")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error("App error", error=str(e))

if __name__ == "__main__":
    main()
