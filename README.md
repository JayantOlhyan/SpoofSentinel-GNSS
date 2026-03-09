# README.md
# 🛰️ SpoofSentinel-GNSS

> AI-powered GNSS spoofing detection using a hybrid Temporal Transformer + XGBoost ensemble
> with physics-grounded feature engineering and SHAP explainability.
> Built for the NyneOS Anti-Spoofing Hackathon @ IIT Delhi (Kaizen 2026)

---

## 📊 Results
| Metric | Value |
|---|---|
| Validation Weighted F1 | 0.9102 (CV) |
| XGBoost alone F1 | 0.9102 (CV) |
| Transformer alone F1 | [Training...] |
| Ensemble F1 | [Pending...] |

## 📌 Problem Understanding
GNSS spoofing involves broadcasting fake satellite signals to deceive a receiver about its location and time. This is dangerous for autonomous vehicles, drones, and critical infrastructure. SpoofSentinel detects these anomalies by looking for physical inconsistencies (e.g., correlator asymmetry) and temporal patterns.

## 🏗️ Architecture Overview
`Data → Feature Engineering → [XGBoost + Temporal Transformer] → Logistic Stacking → predictions.csv`

## 🔬 Feature Engineering
- **Correlator Symmetry**: Detects signal shape distortion using EC/LC/PC ratios.
- **Temporal Jumps**: Monitors `Carrier_phase_cycles` for sudden discontinuities.
- **Timing Residuals**: Cross-references `RX_time` with `TOW_at_current_symbol_s`.
- **Rolling Stats**: Calculates variance and rate of change for Doppler and CN0.

## 🤖 Model Architecture
1. **XGBoost**: Handles engineered tabular features with SMOTE for class balance.
2. **Temporal Transformer**: Sequential analysis over 10-timestep sliding windows.
3. **Ensemble**: Stacking meta-learner (LR) with threshold tuning for Weighted F1 optimization.

## 🚀 How to Run

### Installation
```bash
git clone https://github.com/JayantOlhyan/SpoofSentinel-GNSS
cd SpoofSentinel-GNSS
pip install -r requirements.txt
```

### Training
```bash
# Place train.csv and test.csv in data/ folder
python train.py --train_path data/train.csv
```

### Prediction
```bash
python predict.py --test_path data/test.csv
```

### Dashboard Demo
```bash
streamlit run dashboard.py
```

## 📁 Repository Structure
Standardized ML project layout with `src/` for logic and `outputs/` for models/results.
