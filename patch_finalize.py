import joblib
import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from src.feature_engineering import engineer_features
from src.model_xgboost import predict_xgboost
from src.model_transformer import GNSSSpoofTransformer, GNSSSequenceDataset, get_transformer_predictions
from src.ensemble import tune_ensemble_threshold
from src.explainability import explain_model
from src.utils import get_feature_columns, get_label_column

def patch():
    print("Running patch to finalize results...")
    train_path = 'data/train.csv'
    df = pd.read_csv(train_path)
    df_eng = engineer_features(df)
    
    label_col = get_label_column(df_eng)
    feature_cols = get_feature_columns(df_eng, label_col)
    
    X = df_eng[feature_cols]
    y = df_eng[label_col]
    
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Load models
    xgb_model = joblib.load('outputs/xgboost_model.pkl')
    meta_model = joblib.load('outputs/ensemble_model.pkl')
    
    transformer_model = GNSSSpoofTransformer(input_dim=len(feature_cols))
    transformer_model.load_state_dict(torch.load('outputs/transformer_model.pt', map_location='cpu'))
    
    # Validation probs
    xgb_val_probs = predict_xgboost(xgb_model, X_val)
    val_dataset = GNSSSequenceDataset(df_eng.loc[X_val.index], feature_cols, label_col)
    transformer_val_probs_raw = get_transformer_predictions(transformer_model, val_dataset)
    
    # Alignment
    valid_indices = val_dataset.indices
    y_val_aligned = df_eng.loc[valid_indices, label_col]
    xgb_val_probs_aligned = xgb_val_probs[np.where(np.isin(X_val.index, valid_indices))[0]]
    
    # Tune threshold
    combined_probs = meta_model.predict_proba(np.column_stack([xgb_val_probs_aligned, transformer_val_probs_raw]))[:, 1]
    best_threshold = tune_ensemble_threshold(combined_probs, y_val_aligned)
    
    joblib.dump({'threshold': best_threshold}, 'outputs/threshold.pkl')
    print(f"Threshold saved: {best_threshold}")
    
    # SHAP
    X_train_indices = df_eng.index.difference(X_val.index)
    X_train_sample = df_eng.loc[X_train_indices, feature_cols].sample(min(len(X_train_indices), 500), random_state=42)
    explain_model(xgb_model, X_train_sample, feature_cols)
    print("SHAP plots generated.")

if __name__ == '__main__':
    patch()
