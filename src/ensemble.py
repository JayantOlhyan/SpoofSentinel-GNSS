import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib
import os

def build_ensemble(xgb_probs: np.ndarray, transformer_probs: np.ndarray, y_true: np.ndarray) -> LogisticRegression:
    """
    xgb_probs: shape (N,) — XGBoost predicted probabilities
    transformer_probs: shape (N,) — Transformer predicted probabilities
    y_true: shape (N,) — ground truth labels (training data only)
    """
    meta_features = np.column_stack([xgb_probs, transformer_probs])
    meta_model = LogisticRegression()
    meta_model.fit(meta_features, y_true)
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(meta_model, 'outputs/ensemble_model.pkl')
    print("Ensemble meta-model saved to outputs/ensemble_model.pkl")
    return meta_model

def ensemble_predict(meta_model, xgb_probs: np.ndarray, transformer_probs: np.ndarray, threshold: float = 0.5):
    """
    Combines predictions and returns binary labels and final probabilities.
    """
    meta_features = np.column_stack([xgb_probs, transformer_probs])
    probs = meta_model.predict_proba(meta_features)[:, 1]
    return (probs >= threshold).astype(int), probs

def tune_ensemble_threshold(val_probs: np.ndarray, y_val: np.ndarray):
    """
    Finds the threshold that maximizes Weighted F1.
    """
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, average='weighted')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    
    print(f"Best threshold: {best_threshold:.2f}, Best Weighted F1: {best_f1:.4f}")
    return best_threshold
