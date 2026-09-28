#!/usr/bin/env python3
"""Train and evaluate the eight non-neural-network algorithms in the manuscript."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from analysis_config import DATA_FILE, RESULTS_DIR, RANDOM_SEED
from data_utils import load_analysis_data, save_label_mapping
from evaluation_utils import classification_metrics, save_evaluation

# EDIT THESE TWO VALUES TO SELECT THE ANALYSIS.
CLASS_LABEL = "Grape_variety"  # Country, Continent, or Grape_variety
APPROACH = "B"                 # A or B
TEST_SIZE = 0.20 if APPROACH == "B" else 0.25


def main():
    X, y, encoder = load_analysis_data(DATA_FILE, CLASS_LABEL, APPROACH)
    output_dir = RESULTS_DIR / f"classical_{CLASS_LABEL}_{APPROACH}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_label_mapping(encoder, output_dir / "label_mapping.json")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=8, stratify=y
    )
    n_features = X_train.shape[1]
    max_features_rf = min(50000, n_features)
    max_features_ada = min(45000, n_features)
    n_classes = len(encoder.classes_)
    uniform_priors = np.repeat(1.0 / n_classes, n_classes)

    models = {
        "random_forest": RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED, n_estimators=1000, max_features=max_features_rf, max_depth=250, class_weight="balanced_subsample"),
        "adaboost": AdaBoostClassifier(n_estimators=10000, estimator=DecisionTreeClassifier(max_depth=250, max_features=max_features_ada, class_weight="balanced"), random_state=RANDOM_SEED),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_SEED, max_depth=250, max_features=max_features_rf, n_estimators=500),
        "svm_linear": SVC(random_state=RANDOM_SEED, probability=True, kernel="linear", class_weight="balanced", gamma="scale", C=0.1),
        "svm_radial": SVC(random_state=RANDOM_SEED, probability=True, kernel="rbf"),
        "gaussian_naive_bayes": GaussianNB(),
        "bernoulli_naive_bayes": BernoulliNB(force_alpha=True, binarize=None, class_prior=uniform_priors, fit_prior=False),
        "knn": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        print(f"Training {name}")
        if name == "bernoulli_naive_bayes":
            Xtr = (X_train != 0).astype(np.int8)
            Xte = (X_test != 0).astype(np.int8)
        else:
            Xtr, Xte = X_train, X_test
        model.fit(Xtr, y_train)
        predictions = model.predict(Xte)
        metrics = classification_metrics(y_test, predictions)
        rows.append({"model": name, **metrics})
        save_evaluation(y_test, predictions, encoder.classes_, output_dir, prefix=f"{name}_")
        print(name, metrics)
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False)


if __name__ == "__main__":
    main()
