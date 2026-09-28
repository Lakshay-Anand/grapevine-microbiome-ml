#!/usr/bin/env python3
"""Approach A Neural Network with a held-out test set."""
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from analysis_config import DATA_FILE, RESULTS_DIR, RANDOM_SEED
from data_utils import load_analysis_data, save_label_mapping
from evaluation_utils import classification_metrics, save_evaluation

# EDIT THIS VALUE: Country, Continent, or Grape_variety.
CLASS_LABEL = "Grape_variety"
TEST_SIZE = 0.25
BATCH_SIZE = 8
EPOCHS = 10


def build_model(n_features, n_classes, target):
    model = tf.keras.Sequential()
    if target == "Continent":
        for index in range(6):
            kwargs = {"input_shape": (n_features,)} if index == 0 else {}
            model.add(tf.keras.layers.Dense(800, activation="relu", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.L2(0.001), **kwargs))
            if index in (1, 3): model.add(tf.keras.layers.Dropout(0.5))
    else:
        units = [1000, 1200, 1500, 1000, 1200, 1500, 1000, 800, 1200, 800]
        for index, unit in enumerate(units):
            kwargs = {"input_shape": (n_features,)} if index == 0 else {}
            model.add(tf.keras.layers.Dense(unit, activation="relu", kernel_initializer="he_normal", kernel_regularizer=tf.keras.regularizers.L2(0.001), **kwargs))
            if index in (2, 5): model.add(tf.keras.layers.Dropout(0.3))
            if index == 7: model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.SGD(learning_rate=0.001), metrics=["accuracy"])
    return model


def main():
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); tf.random.set_seed(RANDOM_SEED)
    X, y, encoder = load_analysis_data(DATA_FILE, CLASS_LABEL, approach="A")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=8, stratify=y)
    weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(y_train), weights)}
    model = build_model(X.shape[1], len(encoder.classes_), CLASS_LABEL)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, class_weight=class_weights)
    predictions = np.argmax(model.predict(X_test, batch_size=BATCH_SIZE, verbose=0), axis=1)
    output_dir = RESULTS_DIR / f"nn_approachA_{CLASS_LABEL}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_label_mapping(encoder, output_dir / "label_mapping.json")
    save_evaluation(y_test, predictions, encoder.classes_, output_dir)
    pd.DataFrame([classification_metrics(y_test, predictions)]).to_csv(output_dir / "metrics.csv", index=False)
    model.save(output_dir / "model.keras")


if __name__ == "__main__":
    main()
