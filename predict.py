"""
predict.py — Generate predictions.csv for submission
Usage: python predict.py --test_path data/test.csv --submission_path data/sample_submission.csv
"""

import pandas as pd
import joblib
import torch
import argparse
import numpy as np
import os
from src.feature_engineering import engineer_features
from src.utils import get_feature_columns, get_label_column
from src.model_xgboost import predict_xgboost
from src.model_transformer import GNSSSpoofTransformer, GNSSSequenceDataset, get_transformer_predictions
from src.ensemble import ensemble_predict

def main(test_path, submission_path):
    print("Starting prediction pipeline...")
    
    if not os.path.exists(test_path):
        print(f"Error: Test data not found at {test_path}")
        return
    if not os.path.exists(submission_path):
        print(f"Error: Sample submission not found at {submission_path}")
        return

    print("Loading test data...")
    df_test = pd.read_csv(test_path)
    df_sub = pd.read_csv(submission_path)
    
    print("Engineering features...")
    df_test_eng = engineer_features(df_test)
    feature_cols = get_feature_columns(df_test_eng)
    X_test = df_test_eng[feature_cols]
    
    print("Loading models and assets...")
    xgb_model = joblib.load('outputs/xgboost_model.pkl')
    threshold_data = joblib.load('outputs/threshold.pkl')
    meta_model = joblib.load('outputs/ensemble_model.pkl')
    
    threshold = threshold_data.get('threshold', 0.5)
    
    # Load Transformer
    transformer_model = GNSSSpoofTransformer(input_dim=len(feature_cols))
    transformer_model.load_state_dict(torch.load('outputs/transformer_model.pt', map_location='cpu'))
    
    # Generate XGBoost probs
    xgb_probs = predict_xgboost(xgb_model, X_test)
    
    # Generate Transformer probs
    test_dataset = GNSSSequenceDataset(df_test_eng, feature_cols)
    transformer_probs_raw = get_transformer_predictions(transformer_model, test_dataset)
    
    # Alignment: Transformer dataset might have fewer rows due to WINDOW_SIZE.
    # For full submission, we fill missing initial predictions for each PRN with XGBoost-only results.
    # Or more simply, fill missing values with 0/authentic if no history exists.
    
    # Pre-fill final probabilities with XGBoost-only results (suitably adjusted)
    # This ensures we have a prediction for every row in df_test.
    final_probs_mapped = xgb_probs.copy() 
    
    # Map back the ensemble probabilities where we have them
    for i, idx in enumerate(test_dataset.indices):
        # idx is the original index in df_test_eng
        # Re-calc ensemble prob for those points
        meta_features = np.array([[xgb_probs[int(idx)], transformer_probs_raw[i]]])
        final_probs_mapped[int(idx)] = meta_model.predict_proba(meta_features)[:, 1]
    
    final_preds = (final_probs_mapped >= threshold).astype(int)
    
    # Save predictions in exact submission format: time, Spoofed, Confidence
    df_sub['Spoofed'] = final_preds
    df_sub['Confidence'] = final_probs_mapped
    
    os.makedirs('outputs', exist_ok=True)
    df_sub.to_csv('outputs/predictions.csv', index=False)
    
    print("✅ predictions.csv saved to outputs/")
    print(f"Total predictions: {len(df_sub)}")
    print(f"Spoofed detected: {final_preds.sum()}")
    print(f"Authentic detected: {(final_preds == 0).sum()}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', default='data/test.csv')
    parser.add_argument('--submission_path', default='data/sample_submission.csv')
    args = parser.parse_args()
    main(args.test_path, args.submission_path)
