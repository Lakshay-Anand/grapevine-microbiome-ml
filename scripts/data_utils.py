"""Data loading, filtering, validation, and label encoding utilities."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from analysis_config import N_METADATA_COLUMNS, APPROACH_A_CLASSES, APPROACH_B_CLASSES


def load_analysis_data(data_file, class_label, approach="B"):
    """Load the CLR-normalized matrix and apply manuscript class filtering."""
    data_file = Path(data_file)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Input file not found: {data_file}. See data/README.md."
        )
    df = pd.read_pickle(data_file)
    if class_label not in df.columns:
        raise KeyError(
            f"Required target column '{class_label}' is absent. Available metadata columns: {list(df.columns[-N_METADATA_COLUMNS:])}"
        )
    classes_by_approach = APPROACH_A_CLASSES if approach.upper() == "A" else APPROACH_B_CLASSES
    if class_label not in classes_by_approach:
        raise ValueError(f"{class_label} is not configured for Approach {approach.upper()}.")

    before = len(df)
    df = df[df[class_label].isin(classes_by_approach[class_label])].copy()
    if class_label == "Rootstock":
        df = df[~df["Rootstock"].isin(["unknown", "Shiraz"])].copy()
    elif class_label == "comb":
        df = df[~df["comb"].isin(["unknown", "Shiraz-Shiraz"])].copy()

    feature_columns = list(df.columns[:-N_METADATA_COLUMNS])
    if not feature_columns:
        raise ValueError("No feature columns were found before the metadata columns.")
    X = df[feature_columns]
    if not all(np.issubdtype(dtype, np.number) for dtype in X.dtypes):
        bad = X.select_dtypes(exclude=np.number).columns.tolist()
        raise TypeError(f"All feature columns must be numeric. Non-numeric columns: {bad[:10]}")
    if X.isna().any().any() or df[class_label].isna().any():
        raise ValueError("Missing values were detected in the features or target label.")

    y_text = df[class_label].astype(str)
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(y_text), index=df.index, name=class_label)
    print(f"Samples before filtering: {before}")
    print(f"Samples retained after filtering: {len(df)}")
    print("Class distribution after filtering:")
    print(y_text.value_counts().sort_index().to_string())
    return X, y, encoder


def save_label_mapping(encoder, output_file):
    mapping = {int(i): label for i, label in enumerate(encoder.classes_)}
    Path(output_file).write_text(json.dumps(mapping, indent=2), encoding="utf-8")
