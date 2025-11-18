# baseline_models.py - Baseline (non-GNN) models for performance comparison.

import networkx as nx
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from typing import Dict, Union, Tuple

def _graph_tensor_to_networkx(graph: tfgnn.GraphTensor) -> nx.DiGraph:
    """Converts a tfgnn.GraphTensor to a networkx.DiGraph.

    Args:
        graph (tfgnn.GraphTensor): The GraphTensor to convert.

    Returns:
        nx.DiGraph: The converted networkx graph.
    """
    g = nx.DiGraph()
    nodes = range(graph.node_sets["node"].sizes[0])
    edges = graph.edge_sets["edge"].adjacency.source.numpy(), graph.edge_sets["edge"].adjacency.target.numpy()

    g.add_nodes_from(nodes)
    g.add_edges_from(zip(*edges))
    return g


def extract_node_degree(graph: tfgnn.GraphTensor) -> np.ndarray:
    """Extracts in-degree and out-degree for each node.

    Args:
        graph (tfgnn.GraphTensor): The input graph.

    Returns:
        np.ndarray: A numpy array of shape (num_nodes, 2) with in-degree and
            out-degree for each node.
    """
    g = _graph_tensor_to_networkx(graph)
    in_degree = np.array([d for _, d in g.in_degree()])
    out_degree = np.array([d for _, d in g.out_degree()])
    return np.stack([in_degree, out_degree], axis=1)


def extract_pagerank_centrality(graph: tfgnn.GraphTensor) -> np.ndarray:
    """Extracts PageRank centrality for each node.

    Args:
        graph (tfgnn.GraphTensor): The input graph.

    Returns:
        np.ndarray: A numpy array of shape (num_nodes,) with the PageRank
            centrality of each node.
    """
    g = _graph_tensor_to_networkx(graph)
    pagerank = nx.pagerank(g)
    return np.array([pagerank.get(i, 0.0) for i in range(len(g.nodes))])


def extract_local_clustering_coefficient(graph: tfgnn.GraphTensor) -> np.ndarray:
    """Extracts local clustering coefficient for each node.

    Args:
        graph (tfgnn.GraphTensor): The input graph.

    Returns:
        np.ndarray: A numpy array of shape (num_nodes,) with the local
            clustering coefficient of each node.
    """
    g = _graph_tensor_to_networkx(graph)
    # Clustering coefficient is for undirected graphs.
    clustering = nx.clustering(g.to_undirected())
    return np.array([clustering.get(i, 0.0) for i in range(len(g.nodes))])


def engineer_features(graph: tfgnn.GraphTensor) -> np.ndarray:
    """Engineers additional features from the graph structure.

    Combines the original node features with degree, PageRank, and local
    clustering coefficient.

    Args:
        graph (tfgnn.GraphTensor): The input graph.

    Returns:
        np.ndarray: A numpy array of shape (num_nodes, num_features) with the
            engineered features.
    """
    original_features = graph.node_sets["node"]["hidden_state"].numpy()

    degree_features = extract_node_degree(graph)
    pagerank_features = extract_pagerank_centrality(graph)
    clustering_features = extract_local_clustering_coefficient(graph)

    # Reshape centrality and clustering features to be 2D arrays.
    pagerank_features = np.expand_dims(pagerank_features, axis=1)
    clustering_features = np.expand_dims(clustering_features, axis=1)

    return np.hstack([
        original_features,
        degree_features,
        pagerank_features,
        clustering_features
    ])

def prepare_dataset_for_baseline(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """Prepares a dataset for baseline models.

    Takes a dataset of GraphTensors, engineers features, and pads them in a
    scalable way.

    Args:
        dataset (tf.data.Dataset): A dataset of GraphTensors.

    Returns:
        tf.data.Dataset: A new dataset with engineered and padded features.
    """
    # First pass: find max_nodes without loading everything into memory.
    max_nodes = 0
    for graph in dataset:
        num_nodes = graph.node_sets["node"].sizes[0].numpy()
        if num_nodes > max_nodes:
            max_nodes = num_nodes

    # Determine the shape of the engineered features.
    # 3 (original) + 2 (degree) + 1 (pagerank) + 1 (clustering) = 7
    engineered_feature_dim = 7

    def _engineer_and_pad(graph):
        engineered_features = engineer_features(graph)
        label = graph.node_sets['node']['label']
        label_numpy = label.numpy()

        num_nodes = engineered_features.shape[0]
        pad_width = max_nodes - num_nodes

        padded_features = np.pad(
            engineered_features,
            ((0, pad_width), (0, 0)),
            'constant',
            constant_values=-1
        ).astype(np.float32)

        padded_label = np.pad(
            label_numpy,
            ((0, pad_width), (0, 0)),
            'constant',
            constant_values=-1
        ).astype(np.float32)

        return padded_features, padded_label

    def _map_fn(graph):
        features, labels = tf.py_function(
            _engineer_and_pad,
            inp=[graph],
            Tout=[tf.float32, tf.float32]
        )
        features.set_shape([max_nodes, engineered_feature_dim])
        labels.set_shape([max_nodes, 1])
        return features, labels

    return dataset.map(_map_fn)


from sklearn.svm import SVR
from sklearn.base import BaseEstimator, RegressorMixin


class SVMModel(BaseEstimator, RegressorMixin):
    """A wrapper for the scikit-learn SVR model.

    This class provides a scikit-learn compatible wrapper for the SVR model,
    allowing it to be used in hyperparameter tuning pipelines.

    Attributes:
        C (float): Regularization parameter.
        epsilon (float): Epsilon in the epsilon-SVR model.
        kernel (str): Specifies the kernel type to be used in the algorithm.
        gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        degree (int): Degree of the polynomial kernel function ('poly').
        model (SVR): The underlying scikit-learn SVR model.
    """

    def __init__(self, C=1.0, epsilon=0.1, kernel='rbf', gamma='scale', degree=3):
        """Initializes the SVMModel.

        Args:
            C (float, optional): Regularization parameter. Defaults to 1.0.
            epsilon (float, optional): Epsilon in the epsilon-SVR model.
                Defaults to 0.1.
            kernel (str, optional): Specifies the kernel type. Defaults to 'rbf'.
            gamma (str or float, optional): Kernel coefficient. Defaults to 'scale'.
            degree (int, optional): Degree of the polynomial kernel. Defaults to 3.
        """
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.model = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma, degree=degree)

    def fit(self, X, y):
        """Fits the SVM model to the training data.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.

        Returns:
            SVMModel: The fitted model.
        """
        # Flatten the features and labels
        X_flattened = X.reshape(-1, X.shape[-1])
        y_flattened = y.ravel()
        # Create a mask for non-padded values
        mask = ~np.all(X_flattened == -1, axis=1)
        # Apply the mask
        X_filtered = X_flattened[mask]
        y_filtered = y_flattened[mask]
        self.model.fit(X_filtered, y_filtered)
        return self

    def predict(self, X):
        """Predicts using the SVM model.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted values.
        """
        X_flattened = X.reshape(-1, X.shape[-1])
        return self.model.predict(X_flattened)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Masking


class MLPModel(BaseEstimator, RegressorMixin):
    """A Multi-Layer Perceptron model for regression using TensorFlow/Keras.

    This class provides a scikit-learn compatible wrapper for a Keras MLP model.

    Attributes:
        input_shape (tuple): The shape of the input data.
        layers (list): A list of integers representing the number of units in
            each hidden layer.
        epochs (int): The number of epochs to train for.
        batch_size (int): The batch size for training.
        verbose (int): The verbosity mode for training.
        model (tf.keras.Model): The underlying Keras model.
    """
    def __init__(self, input_shape, layers=[64, 32], epochs=10, batch_size=32, verbose=0):
        """Initializes the MLPModel.

        Args:
            input_shape (tuple): The shape of the input data.
            layers (list, optional): A list of integers for the hidden layer
                units. Defaults to [64, 32].
            epochs (int, optional): The number of epochs to train for.
                Defaults to 10.
            batch_size (int, optional): The batch size for training.
                Defaults to 32.
            verbose (int, optional): The verbosity mode for training.
                Defaults to 0.
        """
        self.input_shape = input_shape
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = self._build_model()

    def _build_model(self):
        """Builds the Keras MLP model."""
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        # Add a masking layer to ignore padded values
        model.add(Masking(mask_value=-1.))
        for units in self.layers:
            model.add(Dense(units, activation='relu'))
        model.add(Dense(1))  # Output layer for regression
        model.compile(optimizer='adam', loss='mse')
        return model

    def fit(self, X, y):
        """Fits the MLP model to the training data.

        Args:
            X (np.ndarray): The training input samples.
            y (np.ndarray): The target values.

        Returns:
            MLPModel: The fitted model.
        """
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def predict(self, X):
        """Predicts using the MLP model.

        Args:
            X (np.ndarray): The input samples.

        Returns:
            np.ndarray: The predicted values.
        """
        return self.model.predict(X)
