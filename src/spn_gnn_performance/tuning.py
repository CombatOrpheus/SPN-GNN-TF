# tuning.py - Hyperparameter tuning for GNN and baseline models.

import keras_tuner as kt
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from spn_gnn_performance import baseline_models, models


def build_gcn_model(hp):
    """Builds a GCN model for KerasTuner."""
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model = models.GCNModel(units=units, output_dim=1)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(optimizer=optimizer, loss=loss)
    return model


def build_gat_model(hp):
    """Builds a GAT model for KerasTuner."""
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model = models.GATModel(units=units, output_dim=1, num_heads=num_heads)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(optimizer=optimizer, loss=loss)
    return model


def build_mpnn_model(hp):
    """Builds a MPNN model for KerasTuner."""
    message_dim = hp.Int("message_dim", min_value=32, max_value=128, step=32)
    next_state_dim = hp.Int("next_state_dim", min_value=32, max_value=128, step=32)
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model = models.MPNNModel(output_dim=1, message_dim=message_dim, next_state_dim=next_state_dim)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(optimizer=optimizer, loss=loss)
    return model


def tune_svm_model(X, y):
    """Tunes the SVM model using Bayesian optimization."""
    param_space = {
        'C': Real(1e-6, 1e+6, prior='log-uniform'),
        'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        'degree': Integer(1, 8),
        'kernel': Categorical(['linear', 'poly', 'rbf']),
    }

    model = baseline_models.SVMModel()

    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=32,
        cv=3,
        n_jobs=-1,
        verbose=0,
        scoring='neg_mean_absolute_percentage_error',
    )

    # Flatten the features and labels
    X_flattened = X.reshape(-1, X.shape[-1])
    y_flattened = y.ravel()
    # Create a mask for non-padded values
    mask = ~np.all(X_flattened == -1, axis=1)
    # Apply the mask
    X_filtered = X_flattened[mask]
    y_filtered = y_flattened[mask]

    search.fit(X_filtered, y_filtered)

    return search.best_params_, search.cv_results_


def build_mlp_model(hp):
    """Builds an MLP model for KerasTuner."""
    num_layers = hp.Int("num_layers", min_value=1, max_value=3, step=1)
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    activation = hp.Choice("activation", values=["relu", "tanh"])
    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(None, 7)))  # Shape from prepare_dataset_for_baseline
    model.add(tf.keras.layers.Masking(mask_value=-1.))

    for _ in range(num_layers):
        model.add(tf.keras.layers.Dense(units, activation=activation))

    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(optimizer=optimizer, loss=loss)
    return model
