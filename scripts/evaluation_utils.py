"""Reusable evaluation and figure-export functions."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score


def classification_metrics(y_true, y_pred):
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


def save_evaluation(y_true, y_pred, class_names, output_dir, prefix=""):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=list(class_names), zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(output_dir / f"{prefix}classification_report.csv")
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(max(7, len(class_names) * 0.8), max(6, len(class_names) * 0.65)))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=90, values_format=".2f", colorbar=False)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}confusion_matrix_normalized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
