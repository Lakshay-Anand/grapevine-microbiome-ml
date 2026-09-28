#!/usr/bin/env python3
"""SHAP interpretation of the saved Approach B scion Neural Network."""
from pathlib import Path
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from analysis_config import RESULTS_DIR

# The manuscript SHAP analysis is for Approach B scion classification.
TARGET = "Grape_variety"
TOP_N = 20


def normalize_shap_output(values, n_classes):
    """Return one (samples x features) array per output class."""
    if isinstance(values, list):
        return values
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f"Unexpected SHAP array shape: {values.shape}")
    if values.shape[-1] == n_classes:
        return [values[:, :, index] for index in range(n_classes)]
    if values.shape[0] == n_classes:
        return [values[index, :, :] for index in range(n_classes)]
    raise ValueError(f"Cannot identify class dimension in SHAP array: {values.shape}")


def main():
    input_dir = RESULTS_DIR / f"nn_approachB_{TARGET}"
    model_file = input_dir / "model.keras"
    background_file = input_dir / "shap_background.pkl"
    evaluation_file = input_dir / "shap_evaluation.pkl"
    for path in (model_file, background_file, evaluation_file):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}. Run the Approach B scion script first.")
    model = tf.keras.models.load_model(model_file)
    background = pd.read_pickle(background_file)
    evaluation = pd.read_pickle(evaluation_file)
    explainer = shap.DeepExplainer(model, background.values)
    raw_values = explainer.shap_values(evaluation.values, check_additivity=False)
    class_values = normalize_shap_output(raw_values, model.output_shape[-1])
    class_names = [str(i) for i in range(len(class_values))]
    mapping_file = input_dir / "label_mapping.json"
    if mapping_file.exists():
        import json
        mapping = json.loads(mapping_file.read_text())
        class_names = [mapping[str(i)] for i in range(len(class_values))]
    all_means, positive_rows, negative_rows = {}, [], []
    for name, values in zip(class_names, class_values):
        signed_mean = values.mean(axis=0)
        mean_positive = np.where(values > 0, values, 0).mean(axis=0)
        mean_negative = np.where(values < 0, values, 0).mean(axis=0)
        all_means[name] = signed_mean
        for rank, index in enumerate(np.argsort(mean_positive)[::-1][:TOP_N], start=1):
            positive_rows.append({"class": name, "rank": rank, "feature": evaluation.columns[index], "mean_positive_shap": mean_positive[index]})
        for rank, index in enumerate(np.argsort(mean_negative)[:TOP_N], start=1):
            negative_rows.append({"class": name, "rank": rank, "feature": evaluation.columns[index], "mean_negative_shap": mean_negative[index]})
    pd.DataFrame(all_means, index=evaluation.columns).to_csv(input_dir / "shap_mean_all_features.csv")
    pd.DataFrame(positive_rows).to_csv(input_dir / "shap_top_positive.csv", index=False)
    pd.DataFrame(negative_rows).to_csv(input_dir / "shap_top_negative.csv", index=False)
    print(f"SHAP tables saved to {input_dir}")


if __name__ == "__main__":
    main()
