import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import torch
import os
from src.feature_engineering import engineer_features
from src.utils import get_feature_columns
from src.model_transformer import GNSSSpoofTransformer, GNSSSequenceDataset, get_transformer_predictions

st.set_page_config(page_title="SpoofSentinel-GNSS", page_icon="🛰️", layout="wide")

@st.cache_resource
def load_models():
    if not os.path.exists('outputs/xgboost_model.pkl'):
        return None, None, None, None
    xgb_model = joblib.load('outputs/xgboost_model.pkl')
    meta_model = joblib.load('outputs/ensemble_model.pkl')
    threshold = joblib.load('outputs/threshold.pkl').get('threshold', 0.5)
    
    # Load Transformer (assuming 50 features for initialization, will re-init if needed)
    # In a real app we'd store the input_dim
    transformer = None
    if os.path.exists('outputs/transformer_model.pt'):
        # Just a placeholder for now, actual init needs feature count
        pass
    
    return xgb_model, meta_model, threshold, transformer

xgb_model, meta_model, threshold, _ = load_models()

st.title("🛰️ SpoofSentinel — GNSS Anti-Spoofing Detection")
st.markdown("AI-powered detector for authentic vs spoofed GNSS signals.")

if xgb_model is None:
    st.warning("⚠️ Models not found in `outputs/`. Please run `python train.py` first.")

tabs = st.tabs(["Upload & Predict", "Signal Analysis", "Model Explainability", "About"])

with tabs[0]:
    st.header("Batch Prediction")
    uploaded_file = st.file_uploader("Upload GNSS Signal CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        with st.spinner("Engineering features and predicting..."):
            df_eng = engineer_features(df)
            feat_cols = get_feature_columns(df_eng)
            
            # Simple XGBoost prediction for demo if ensemble not fully ready
            probs = xgb_model.predict_proba(df_eng[feat_cols])[:, 1]
            preds = (probs >= threshold).astype(int)
            
            df['Prediction'] = preds
            df['Status'] = df['Prediction'].map({0: "🟢 Authentic", 1: "🔴 Spoofed"})
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.dataframe(df[['PRN', 'RX_time', 'CN0', 'Status']].head(100))
            
            with col2:
                fig = px.pie(df, names='Status', title='Signal Distribution', color='Status',
                            color_discrete_map={"🟢 Authentic": "green", "🔴 Spoofed": "red"})
                st.plotly_chart(fig)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

with tabs[1]:
    st.header("Signal Characteristic Analysis")
    if uploaded_file is not None:
        # Doppler Shift Plot
        fig_doppler = px.line(df, x='RX_time', y='Carrier_Doppler_hz', color='Status', 
                             title="Carrier Doppler over Time", markers=True)
        st.plotly_chart(fig_doppler, use_container_width=True)
        
        # Correlator Symmetry
        if 'EC' in df.columns and 'LC' in df.columns:
            fig_corr = px.scatter(df, x='EC', y='LC', color='Status', 
                                title="Correlator Symmetry (EC vs LC)")
            st.plotly_chart(fig_corr, use_container_width=True)

with tabs[2]:
    st.header("Explainable AI (SHAP)")
    if os.path.exists('outputs/shap_feature_importance.png'):
        st.image('outputs/shap_feature_importance.png', caption="Global Feature Importance")
    if os.path.exists('outputs/shap_beeswarm.png'):
        st.image('outputs/shap_beeswarm.png', caption="SHAP Beeswarm Plot")

with tabs[3]:
    st.header("About SpoofSentinel-GNSS")
    st.markdown("""
    ### Hybrid Ensemble Architecture
    - **XGBoost**: Captures high-variance tabular relationships.
    - **Temporal Transformer**: Models temporal dependencies and sequential anomalies.
    - **Stacking**: Combines both using a Logistic Regression meta-learner.
    
    Built for Kaizen 2026 GNSS Anti-Spoofing Hackathon.
    """)
