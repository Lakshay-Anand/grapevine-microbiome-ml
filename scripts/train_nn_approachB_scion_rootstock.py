#!/usr/bin/env python3
"""Approach B Neural Network for comb classification."""
from nn_common import train_approach_b

# Editable analysis settings.
CLASS_LABEL = "comb"
HIDDEN_LAYERS = 3
NODES_PER_LAYER = 1024
N_SPLITS = 5
N_REPEATS = 1
EPOCHS_MAX = 20
BATCH_SIZE = 8

if __name__ == "__main__":
    train_approach_b(CLASS_LABEL, HIDDEN_LAYERS, NODES_PER_LAYER, N_SPLITS, N_REPEATS, EPOCHS_MAX, BATCH_SIZE)
