import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
import os

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """Train and return fitted XGBoost model with SMOTE handling."""
    
    # 5.1 Class Imbalance Handling
    # Apply SMOTE to training data ONLY
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    # 5.2 Model Configuration
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    # 5.3 Cross-Validation Strategy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    weighted_f1 = make_scorer(f1_score, average='weighted')
    
    print("Running Cross-Validation for XGBoost...")
    scores = cross_val_score(model, X_res, y_res, cv=skf, scoring=weighted_f1)
    print(f"CV Weighted F1: {scores.mean():.4f} ± {scores.std():.4f}")
    
    # Fit on all resampled training data
    model.fit(X_res, y_res)
    
    # 5.4 Save the Model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/xgboost_model.pkl')
    print("XGBoost model saved to outputs/xgboost_model.pkl")
    
    return model

def predict_xgboost(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return probability of positive class (spoofed)."""
    return model.predict_proba(X)[:, 1]
