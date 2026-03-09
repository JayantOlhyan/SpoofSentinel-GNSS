import shap
import matplotlib.pyplot as plt
import os
import pandas as pd

def explain_model(xgb_model, X_sample: pd.DataFrame, feature_names: list, save_path: str = 'outputs/'):
    """
    Generate and save SHAP plots for the XGBoost model.
    X_sample: a representative sample from the training data.
    """
    os.makedirs(save_path, exist_ok=True)
    
    print("Generating SHAP explanations...")
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_sample)
        
        # 1. Summary plot (bar chart of feature importance)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                          plot_type='bar', show=False)
        plt.title('SHAP Feature Importance (Bar)')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'shap_feature_importance.png'), dpi=150)
        plt.close()
        
        # 2. Beeswarm plot (direction of impact)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title('SHAP Beeswarm Plot')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'shap_beeswarm.png'), dpi=150)
        plt.close()
        
        print(f"SHAP plots saved to {save_path}")
    except Exception as e:
        print(f"Error generating SHAP plots: {e}")
