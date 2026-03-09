import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps. Returns df with new columns added."""
    df = df.copy()
    
    # --- Data Cleaning: Convert to numeric and drop NaNs from placeholders (e.g. 'ch0') ---
    cols_to_convert = ['PRN', 'Carrier_Doppler_hz', 'Pseudorange_m', 'RX_time', 'TOW', 
                       'Carrier_phase', 'EC', 'LC', 'PC', 'PIP', 'PQP', 'TCD', 'CN0']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where critical telemetry is missing (the 'ch0' placeholders)
    df = df.dropna(subset=['PRN', 'Carrier_Doppler_hz', 'EC']).reset_index(drop=True)
    
    # --- 4.1 Physics-Based Features ---
    # Correlator Symmetry Score
    # Real signals have EC ≈ LC (symmetric correlator outputs). Spoofed signals distort this.
    df['correlator_symmetry'] = (df['EC'] - df['LC']) / (df['PC'] + 1e-9)

    # Correlator Sum (total energy)
    df['correlator_sum'] = df['EC'] + df['LC'] + df['PC']

    # Early-Late ratio
    df['EC_LC_ratio'] = df['EC'] / (df['LC'] + 1e-9)

    # PIP to PQP ratio (quality metric cross-feature)
    df['quality_ratio'] = df['PIP'] / (df['PQP'] + 1e-9)

    # CN0 normalized
    df['CN0_normalized'] = (df['CN0'] - df['CN0'].mean()) / (df['CN0'].std() + 1e-9)
    
    # --- 4.2 Temporal/Rolling Features ---
    # Sort data by PRN and RX_time before computing these.
    df = df.sort_values(['PRN', 'RX_time']).reset_index(drop=True)
    
    target_cols = ['Carrier_Doppler_hz', 'Pseudorange_m', 'CN0', 'Carrier_phase', 'TCD']
    
    for col in target_cols:
        group = df.groupby('PRN')[col]
        df[f'{col}_rolling_mean'] = group.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{col}_rolling_std']  = group.transform(lambda x: x.rolling(5, min_periods=1).std().fillna(0))
        df[f'{col}_diff']         = group.transform(lambda x: x.diff().fillna(0))  # rate of change
        df[f'{col}_diff2']        = group.transform(lambda x: x.diff().diff().fillna(0))  # acceleration

    # --- 4.3 Phase Jump Detection ---
    # Sudden large jumps in carrier phase are a classic spoofing indicator
    df['phase_jump'] = df.groupby('PRN')['Carrier_phase'].transform(lambda x: x.diff().abs().fillna(0))
    df['phase_jump_flag'] = (df['phase_jump'] > df['phase_jump'].quantile(0.95)).astype(int)

    # --- 4.4 Timing Inconsistency Feature ---
    # Difference between receiver time and satellite transmit time
    df['timing_residual'] = df['RX_time'] - df['TOW']
    df['timing_residual_abs'] = df['timing_residual'].abs()

    # --- 4.5 PRN Consistency Feature ---
    # Count how many times each PRN satellite appears — abnormal repetition can indicate spoofing
    prn_counts = df['PRN'].value_counts()
    df['prn_frequency'] = df['PRN'].map(prn_counts)

    # --- 4.6 Z-Score Anomaly Features ---
    # For CN0 and Doppler: high z-score = signal looks very different from its PRN's typical behavior
    for col in ['CN0', 'Carrier_Doppler_hz']:
        group_mean = df.groupby('PRN')[col].transform('mean')
        group_std  = df.groupby('PRN')[col].transform('std').replace(0, 1e-9)
        df[f'{col}_zscore'] = (df[col] - group_mean) / group_std
        
    # Handle NaNs from rolling/diff (though rolling(min_periods=1) handles most)
    df = df.fillna(0)
    
    return df
