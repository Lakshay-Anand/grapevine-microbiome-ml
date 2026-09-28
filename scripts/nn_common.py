"""Shared Approach B Neural Network training workflow."""
import gc
import random
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import RepeatedStratifiedKFold
from analysis_config import DATA_FILE, RESULTS_DIR, RANDOM_SEED
from data_utils import load_analysis_data, save_label_mapping
from evaluation_utils import classification_metrics, save_evaluation


def set_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def build_model(n_features, n_classes, hidden_layers=3, nodes=1024, dropout=0.2):
    model = tf.keras.Sequential(name="approach_b_neural_network")
    for layer_index in range(hidden_layers):
        kwargs = {"input_shape": (n_features,)} if layer_index == 0 else {}
        model.add(tf.keras.layers.Dense(nodes, activation="relu", kernel_regularizer=tf.keras.regularizers.L2(0.001), **kwargs))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
    model.compile(
        loss=tf.keras.losses.CategoricalFocalCrossentropy(gamma=2.0, alpha=0.5),
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model


def train_approach_b(class_label, hidden_layers=3, nodes=1024, n_splits=5, n_repeats=1, epochs=20, batch_size=8):
    set_seeds()
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    X, y, encoder = load_analysis_data(DATA_FILE, class_label, approach="B")
    n_features, n_classes = X.shape[1], len(encoder.classes_)
    y_one_hot = tf.one_hot(y.to_numpy(), depth=n_classes).numpy()
    output_dir = RESULTS_DIR / f"nn_approachB_{class_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_label_mapping(encoder, output_dir / "label_mapping.json")
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_SEED)
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    fold_rows, last = [], None
    for fold, (train_index, test_index) in enumerate(cv.split(X, y), start=1):
        tf.keras.backend.clear_session()
        model = build_model(n_features, n_classes, hidden_layers, nodes)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]
        y_test_index = y.iloc[test_index].to_numpy()
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[callback], epochs=epochs, batch_size=batch_size, verbose=0)
        predictions = np.argmax(model.predict(X_test, batch_size=batch_size, verbose=0), axis=1)
        metrics = classification_metrics(y_test_index, predictions)
        fold_rows.append({"fold": fold, "epochs_run": len(history.history["loss"]), **metrics})
        print(f"Fold {fold}: {metrics}")
        last = (model, X_train.copy(), X_test.copy(), y_test_index.copy(), predictions.copy())
        gc.collect()
    pd.DataFrame(fold_rows).to_csv(output_dir / "metrics.csv", index=False)
    model, X_train, X_test, y_test_index, predictions = last
    save_evaluation(y_test_index, predictions, encoder.classes_, output_dir)
    model.save(output_dir / "model.keras")
    X_train.iloc[: min(100, len(X_train))].to_pickle(output_dir / "shap_background.pkl")
    X_test.to_pickle(output_dir / "shap_evaluation.pkl")
    print(pd.DataFrame(fold_rows).mean(numeric_only=True).to_string())
