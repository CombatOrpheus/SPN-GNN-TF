# tf_dataset.py - TensorFlow Dataset handling for SPN data.

import json
import tensorflow as tf
import tensorflow_gnn as tfgnn
from typing import Tuple

def _parse_spn_json_and_build_graph(json_string: str) -> tfgnn.GraphTensor:
    """Parses a JSON-L string and constructs a GraphTensor.

    Extracts SPN data, node features, labels, and edge information to build a
    homogeneous graph representation. The regression labels are stored as a
    'label' feature in the node set.

    Args:
        json_string (str): A string containing the JSON-L data.

    Returns:
        tfgnn.GraphTensor: The constructed GraphTensor.
    """
    # ⚡ Bolt: Parse python string directly to avoid extremely slow tf.Tensor -> string decoding overhead inside tight loops
    data = json.loads(json_string)

    petri_net = data["petri_net"]
    num_places = len(petri_net)
    num_transitions = len(petri_net[0]) // 2 if num_places > 0 else 0

    # Extract initial marking and firing rates
    initial_marking = [row[-1] for row in petri_net]
    firing_rates = data["spn_labda"]

    # ⚡ Bolt: Build features purely in Python to avoid overhead of tf.stack and tf.concat on many small vectors
    places = [[1.0, float(im), 0.0] for im in initial_marking]
    transitions = [[0.0, 0.0, float(fr)] for fr in firing_rates]
    node_features = tf.constant(places + transitions, dtype=tf.float32)

    # ⚡ Bolt: Python concatenation instead of tf.concat for labels
    labels = tf.constant(data["spn_allmus"] + firing_rates, dtype=tf.float32)

    # ⚡ Bolt: Parse edges purely in Python to avoid slow tf.where / tf.unstack inside tight map loops
    edges_src = []
    edges_tgt = []
    edge_weights = []

    for p in range(num_places):
        # Pre-conditions
        for t in range(num_transitions):
            w = petri_net[p][t]
            if w > 0:
                edges_src.append(p)
                edges_tgt.append(t + num_places)
                edge_weights.append(w)

        # Post-conditions
        for t in range(num_transitions):
            w = petri_net[p][t + num_transitions]
            if w > 0:
                edges_src.append(t + num_places)
                edges_tgt.append(p)
                edge_weights.append(w)

    num_edges = len(edges_src)

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "node": tfgnn.NodeSet.from_fields(
                sizes=[num_places + num_transitions],
                features={
                    "hidden_state": node_features,
                    "label": tf.expand_dims(labels, axis=-1)
                }
            )
        },
        edge_sets={
            "edge": tfgnn.EdgeSet.from_fields(
                sizes=[num_edges],
                features={"weight": tf.expand_dims(tf.constant(edge_weights, dtype=tf.float32), axis=-1)},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("node", tf.constant(edges_src, dtype=tf.int32)),
                    target=("node", tf.constant(edges_tgt, dtype=tf.int32))
                )
            )
        }
    )

    return graph

def _fast_line_count(file_path: str) -> int:
    """Counts the number of lines in a file quickly using block reads.

    This avoids slow line-by-line iteration when setting dataset cardinality.
    """
    with open(file_path, 'rb') as f:
        lines = 0
        buf_size = 1024 * 1024
        read_f = f.read
        buf = read_f(buf_size)
        last_buf = b''
        while buf:
            lines += buf.count(b'\n')
            last_buf = buf
            buf = read_f(buf_size)
        if last_buf and not last_buf.endswith(b'\n'):
            lines += 1
    return lines

def load_dataset(file_path: str) -> tf.data.Dataset:
    """Creates a tf.data.Dataset from a JSON-L file of SPN data.

    Args:
        file_path (str): The path to the JSON-L file.

    Returns:
        tf.data.Dataset: A dataset of GraphTensors.
    """
    graph_spec = tfgnn.GraphTensorSpec.from_piece_specs(
        node_sets_spec={
            'node': tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    'hidden_state': tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                    'label': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                },
                sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32))
        },
        edge_sets_spec={
            'edge': tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={'weight': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)},
                sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'node', 'node',
                    index_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32)))
        })

    def generator():
        with open(file_path, 'r') as f:
            for line in f:
                # ⚡ Bolt: Yield python strings directly into the dataset instead of wrapping them in tf.constant
                yield _parse_spn_json_and_build_graph(line)

    # Fast line count to inform TF of cardinality to avoid slow fallbacks during split
    num_lines = _fast_line_count(file_path)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=graph_spec
    )

    # Cache the dataset to disk to avoid re-parsing JSON and rebuilding GraphTensors on every epoch
    cache_path = file_path + ".cache"
    dataset = dataset.cache(cache_path)

    import os
    if not os.path.exists(cache_path + ".index"):
        for _ in dataset:
            pass

    return dataset.apply(tf.data.experimental.assert_cardinality(num_lines))

def split_dataset(dataset: tf.data.Dataset, train_split=0.8, val_split=0.1, shuffle=True, seed=42) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Splits a dataset into training, validation, and test sets.

    Args:
        dataset (tf.data.Dataset): The dataset to split.
        train_split (float, optional): The proportion of the dataset to use for
            training. Defaults to 0.8.
        val_split (float, optional): The proportion of the dataset to use for
            validation. Defaults to 0.1.
        shuffle (bool, optional): Whether to shuffle the dataset before
            splitting. Defaults to True.
        seed (int, optional): The random seed for shuffling. Defaults to 42.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: A tuple
            containing the training, validation, and test datasets.
    """
    dataset_size = dataset.cardinality()
    if dataset_size == tf.data.experimental.UNKNOWN_CARDINALITY:
        # Fallback for datasets with unknown cardinality
        dataset_size = dataset.reduce(tf.constant(0, dtype=tf.int64), lambda x, _: x + 1).numpy()
    elif isinstance(dataset_size, tf.Tensor):
        dataset_size = dataset_size.numpy()

    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    if shuffle:
        # ⚡ Bolt: Set reshuffle_each_iteration=False to prevent data leakage across epochs.
        # Otherwise, skip() and take() will return different elements on every iteration.
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed, reshuffle_each_iteration=False)

    train_dataset = dataset.take(train_size)
    if shuffle:
        # ⚡ Bolt: Re-apply dynamic shuffle specifically to the training dataset to ensure
        # stochastic gradient descent receives data in a random order every epoch.
        train_dataset = train_dataset.shuffle(buffer_size=train_size, seed=seed, reshuffle_each_iteration=True)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    return train_dataset, val_dataset, test_dataset


def _parse_spn_json_and_build_heterogeneous_graph(json_string: str) -> tfgnn.GraphTensor:
    """Parses a JSON-L string and constructs a heterogeneous GraphTensor."""
    # ⚡ Bolt: Parse python string directly to avoid extremely slow tf.Tensor -> string decoding overhead inside tight loops
    data = json.loads(json_string)

    petri_net = data["petri_net"]
    num_places = len(petri_net)
    num_transitions = len(petri_net[0]) // 2 if num_places > 0 else 0

    # Extract features natively
    initial_marking = [[row[-1]] for row in petri_net]
    firing_rates = [[fr] for fr in data["spn_labda"]]

    place_features = tf.constant(initial_marking, dtype=tf.float32)
    transition_features = tf.constant(firing_rates, dtype=tf.float32)

    # Regression labels
    place_labels = tf.constant([[val] for val in data["spn_allmus"]], dtype=tf.float32)
    transition_labels = tf.constant([[val] for val in data["spn_labda"]], dtype=tf.float32)

    # ⚡ Bolt: Parse edges purely in Python to avoid slow tf.where / tf.unstack inside tight map loops
    p_in_idx = []
    t_in_idx = []
    weights_in = []

    p_out_idx = []
    t_out_idx = []
    weights_out = []

    for p in range(num_places):
        for t in range(num_transitions):
            w_in = petri_net[p][t]
            if w_in > 0:
                p_in_idx.append(p)
                t_in_idx.append(t)
                weights_in.append([w_in])

            w_out = petri_net[p][t + num_transitions]
            if w_out > 0:
                t_out_idx.append(t)
                p_out_idx.append(p)
                weights_out.append([w_out])

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "place": tfgnn.NodeSet.from_fields(
                sizes=[num_places],
                features={
                    "hidden_state": place_features,
                    "label": place_labels
                }
            ),
            "transition": tfgnn.NodeSet.from_fields(
                sizes=[num_transitions],
                features={
                    "hidden_state": transition_features,
                    "label": transition_labels
                }
            )
        },
        edge_sets={
            "p_to_t": tfgnn.EdgeSet.from_fields(
                sizes=[len(p_in_idx)],
                features={"weight": tf.constant(weights_in, dtype=tf.float32)},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("place", tf.constant(p_in_idx, dtype=tf.int32)),
                    target=("transition", tf.constant(t_in_idx, dtype=tf.int32))
                )
            ),
            "t_to_p": tfgnn.EdgeSet.from_fields(
                sizes=[len(p_out_idx)],
                features={"weight": tf.constant(weights_out, dtype=tf.float32)},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("transition", tf.constant(t_out_idx, dtype=tf.int32)),
                    target=("place", tf.constant(p_out_idx, dtype=tf.int32))
                )
            )
        }
    )
    return graph

def load_heterogeneous_dataset(file_path: str) -> tf.data.Dataset:
    """Creates a tf.data.Dataset from a JSON-L file of SPN data for heterogeneous models."""
    graph_spec = tfgnn.GraphTensorSpec.from_piece_specs(
        node_sets_spec={
            'place': tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    'hidden_state': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    'label': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                },
                sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32)),
            'transition': tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    'hidden_state': tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
                    'label': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
                },
                sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32))
        },
        edge_sets_spec={
            'p_to_t': tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={'weight': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)},
                sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'place', 'transition',
                    index_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32))),
            't_to_p': tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={'weight': tf.TensorSpec(shape=(None, 1), dtype=tf.float32)},
                sizes_spec=tf.TensorSpec(shape=(1,), dtype=tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'transition', 'place',
                    index_spec=tf.TensorSpec(shape=(None,), dtype=tf.int32)))
        })

    def generator():
        with open(file_path, 'r') as f:
            for line in f:
                # ⚡ Bolt: Yield python strings directly into the dataset instead of wrapping them in tf.constant
                yield _parse_spn_json_and_build_heterogeneous_graph(line)

    # Fast line count to inform TF of cardinality to avoid slow fallbacks during split
    num_lines = _fast_line_count(file_path)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=graph_spec
    )

    # Cache the dataset to disk to avoid re-parsing JSON and rebuilding GraphTensors on every epoch
    cache_path = file_path + ".cache"
    dataset = dataset.cache(cache_path)

    import os
    if not os.path.exists(cache_path + ".index"):
        for _ in dataset:
            pass

    return dataset.apply(tf.data.experimental.assert_cardinality(num_lines))
