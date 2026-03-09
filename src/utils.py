import pandas as pd
import numpy as np

# These are columns that should be EXCLUDED from model features
NON_FEATURE_COLS = ['PRN', 'RX_time', 'TOW', 'time', 'channel']

def get_feature_columns(df: pd.DataFrame, label_col: str = None) -> list:
    """Return list of feature column names (excludes label and non-feature cols)."""
    exclude = set(NON_FEATURE_COLS)
    if label_col:
        exclude.add(label_col)
    return [c for c in df.columns if c not in exclude]

def get_label_column(df: pd.DataFrame) -> str:
    """
    Auto-detect the label/target column.
    Looks for columns with binary values (0/1) that aren't feature columns.
    Returns the column name.
    """
    # Common label column names
    candidates = ['label', 'spoofed', 'class', 'target', 'Label', 'Spoofed']
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("Could not auto-detect label column. Please check the dataset.")

def print_class_distribution(y, label='Dataset'):
    """Print class distribution for imbalance analysis."""
    counts = pd.Series(y).value_counts()
    total = len(y)
    print(f"\n{label} Class Distribution:")
    for cls, count in counts.items():
        print(f"  Class {cls}: {count} samples ({100*count/total:.1f}%)")
