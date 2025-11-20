# tuning.py - Hyperparameter tuning for GNN and baseline models.

import keras_tuner as kt
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from . import baseline_models, models


def build_gcn_model(graph_spec):
    """Returns a function that builds a GCN model for KerasTuner.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.

    Returns:
        function: A function that builds a GCN model.
    """
    def build_fn(hp):
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        model = models.GCNModel(graph_spec, units=units, output_dim=1)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        return model
    return build_fn


def build_het_gcn_model(graph_spec):
    """Returns a function that builds a heterogeneous GCN model for KerasTuner.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.

    Returns:
        function: A function that builds a heterogeneous GCN model.
    """
    def build_fn(hp):
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        model = models.HetGCNModel(graph_spec, units=units, output_dim=1)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        return model
    return build_fn


def build_het_graph_sage_model(graph_spec):
    """Returns a function that builds a heterogeneous GraphSAGE model for KerasTuner."""
    def build_fn(hp):
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        model = models.HetGraphSAGEModel(graph_spec, units=units, output_dim=1)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        return model
    return build_fn


def build_het_gat_model(graph_spec):
    """Returns a function that builds a heterogeneous GAT model for KerasTuner."""
    def build_fn(hp):
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        model = models.HetGATModel(graph_spec, units=units, output_dim=1, num_heads=num_heads)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        return model
    return build_fn


def build_het_mpnn_model(graph_spec):
    """Returns a function that builds a heterogeneous MPNN model for KerasTuner."""
    def build_fn(hp):
        message_dim = hp.Int("message_dim", min_value=32, max_value=128, step=32)
        next_state_dim = hp.Int("next_state_dim", min_value=32, max_value=128, step=32)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        model = models.HetMPNNModel(graph_spec, output_dim=1, message_dim=message_dim, next_state_dim=next_state_dim)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        return model
    return build_fn


def build_gat_model(graph_spec):
    """Returns a function that builds a GAT model for KerasTuner.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.

    Returns:
        function: A function that builds a GAT model.
    """
    def build_fn(hp):
        units = hp.Int("units", min_value=32, max_value=256, step=32)
        num_heads = hp.Int("num_heads", min_value=2, max_value=8, step=2)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        model = models.GATModel(graph_spec, units=units, output_dim=1, num_heads=num_heads)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        return model
    return build_fn


def build_mpnn_model(graph_spec):
    """Returns a function that builds an MPNN model for KerasTuner.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.

    Returns:
        function: A function that builds an MPNN model.
    """
    def build_fn(hp):
        message_dim = hp.Int("message_dim", min_value=32, max_value=128, step=32)
        next_state_dim = hp.Int("next_state_dim", min_value=32, max_value=128, step=32)
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")

        model = models.MPNNModel(graph_spec, output_dim=1, message_dim=message_dim, next_state_dim=next_state_dim)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss = tf.keras.losses.MeanAbsolutePercentageError()

        model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])
        return model
    return build_fn


def tune_svm_model(X, y):
    """Tunes the SVM model using Bayesian optimization.

    Args:
        X (np.ndarray): The training input samples.
        y (np.ndarray): The target values.

    Returns:
        tuple: A tuple containing the best hyperparameters and the cross-validation
            results.
    """
    param_space = [
        ({
            'kernel': Categorical(['linear', 'rbf']),
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
        }, 16),
        ({
            'kernel': Categorical(['poly']),
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform'),
            'degree': Integer(1, 8),
        }, 16),
    ]

    model = baseline_models.SVMModel()

    search = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=1,
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
    """Builds an MLP model for KerasTuner.

    Args:
        hp (kt.HyperParameters): The KerasTuner hyperparameter object.

    Returns:
        tf.keras.Model: A compiled MLP model.
    """
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
