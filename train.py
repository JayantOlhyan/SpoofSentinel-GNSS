"""
train.py — SpoofSentinel-GNSS Full Training Pipeline
Usage: python train.py --train_path data/train.csv
"""

import argparse
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

from src.feature_engineering import engineer_features
from src.model_xgboost import train_xgboost, predict_xgboost
from src.model_transformer import (GNSSSpoofTransformer, GNSSSequenceDataset, 
                                 train_transformer, get_transformer_predictions)
from src.ensemble import build_ensemble, ensemble_predict, tune_ensemble_threshold
from src.explainability import explain_model
from src.utils import get_feature_columns, get_label_column, print_class_distribution

def main(train_path):
    print("=" * 60)
    print("SpoofSentinel-GNSS Training Pipeline")
    print("=" * 60)
    
    if not os.path.exists(train_path):
        print(f"Error: Training data not found at {train_path}. Please place your train.csv in the data/ folder.")
        return
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    df = pd.read_csv(train_path)
    
    # Step 2: Feature engineering
    print("[2/6] Engineering features...")
    df_eng = engineer_features(df)
    
    # Step 3: Split
    print("[3/6] Splitting train/validation...")
    label_col = get_label_column(df_eng)
    feature_cols = get_feature_columns(df_eng, label_col)
    
    X = df_eng[feature_cols]
    y = df_eng[label_col]
    
    # Split with stratification
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print_class_distribution(y_train, "Training")
    print_class_distribution(y_val, "Validation")
    
    # Step 4: Train XGBoost
    print("\n[4/6] Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_val_probs = predict_xgboost(xgb_model, X_val)
    
    # Step 5: Train Transformer
    print("\n[5/6] Training Temporal Transformer...")
    # Transformer requires sequence format. We use the original indices to align back to our validation set.
    train_dataset = GNSSSequenceDataset(df_eng.loc[X_train.index], feature_cols, label_col)
    val_dataset = GNSSSequenceDataset(df_eng.loc[X_val.index], feature_cols, label_col)
    
    # Adjust train dataset if indices don't match exactly (e.g. windows crossing train/val borders)
    # For a robust competition build, we should handle this precisely.
    # Here we filter X_val/y_val to only include items that exist in the sequence dataset.
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    transformer_model = GNSSSpoofTransformer(input_dim=len(feature_cols))
    transformer_model = train_transformer(transformer_model, train_loader)
    
    # We need predictions for the validation items that are in the dataset
    transformer_val_probs_raw = get_transformer_predictions(transformer_model, val_dataset)
    
    # Aligning the probabilities:
    # Sequence datasets might have fewer items than original due to WINDOW_SIZE.
    # We'll use the indices tracked in the dataset.
    valid_indices = val_dataset.indices
    y_val_aligned = df_eng.loc[valid_indices, label_col]
    xgb_val_probs_aligned = xgb_val_probs[np.where(np.isin(X_val.index, valid_indices))[0]]
    
    # Step 6: Ensemble + Evaluate
    print("\n[6/6] Building ensemble and evaluating...")
    meta_model = build_ensemble(xgb_val_probs_aligned, transformer_val_probs_raw, y_val_aligned)
    
    # Find best threshold on validation set
    combined_probs = meta_model.predict_proba(np.column_stack([xgb_val_probs_aligned, transformer_val_probs_raw]))[:, 1]
    best_threshold = tune_ensemble_threshold(combined_probs, y_val_aligned)
    
    final_preds, final_probs = ensemble_predict(meta_model, xgb_val_probs_aligned, transformer_val_probs_raw, best_threshold)
    
    weighted_f1 = f1_score(y_val_aligned, final_preds, average='weighted')
    print(f"\n✅ Final Validation Weighted F1: {weighted_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val_aligned, final_preds))
    
    # Save the best threshold for predict.py
    joblib.dump({'threshold': best_threshold}, 'outputs/threshold.pkl')
    
    # SHAP for interpretation
    explain_model(xgb_model, X_train.sample(min(len(X_train), 500), random_state=42), feature_cols)
    
    print("\nTraining complete. Models and assets saved to outputs/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='data/train.csv')
    args = parser.parse_args()
    main(args.train_path)
