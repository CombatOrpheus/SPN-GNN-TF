# tf_dataset.py - TensorFlow Dataset handling for SPN data.

import json
import tensorflow as tf
import tensorflow_gnn as tfgnn
from typing import Tuple

def _parse_spn_json_and_build_graph(json_string: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Parses a JSON-L string, extracts SPN data, and constructs a GraphTensor
    and the corresponding regression labels.
    """
    data = json.loads(json_string.numpy().decode("utf-8"))

    petri_net = tf.constant(data["petri_net"], dtype=tf.float32)
    num_places = tf.shape(petri_net)[0]
    num_transitions = tf.cast(tf.shape(petri_net)[1] // 2, dtype=tf.int32)

    # Node features
    initial_marking = petri_net[:, -1]
    firing_rates = tf.constant(data["spn_labda"], dtype=tf.float32)

    place_features = tf.stack([
        tf.ones(num_places),
        initial_marking,
        tf.zeros(num_places)
    ], axis=1)

    transition_features = tf.stack([
        tf.zeros(num_transitions),
        tf.zeros(num_transitions),
        firing_rates
    ], axis=1)

    node_features = tf.concat([place_features, transition_features], axis=0)

    # Edges and edge features
    pre_conditions = petri_net[:, :num_transitions]
    post_conditions = petri_net[:, num_transitions:2*num_transitions]

    p_in_idx, t_in_idx = tf.unstack(tf.transpose(tf.where(pre_conditions > 0)))
    src_in = p_in_idx
    tgt_in = t_in_idx + tf.cast(num_places, dtype=tf.int64)
    edges_in = tf.stack([src_in, tgt_in], axis=1)
    weights_in = tf.gather_nd(pre_conditions, tf.stack([p_in_idx, t_in_idx], axis=1))

    p_out_idx, t_out_idx = tf.unstack(tf.transpose(tf.where(post_conditions > 0)))
    src_out = t_out_idx + tf.cast(num_places, dtype=tf.int64)
    tgt_out = p_out_idx
    edges_out = tf.stack([src_out, tgt_out], axis=1)
    weights_out = tf.gather_nd(post_conditions, tf.stack([p_out_idx, t_out_idx], axis=1))

    edge_pairs = tf.concat([edges_in, edges_out], axis=0)
    edge_features = tf.concat([weights_in, weights_out], axis=0)

    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "node": tfgnn.NodeSet.from_fields(
                sizes=[num_places + num_transitions],
                features={"hidden_state": node_features}
            )
        },
        edge_sets={
            "edge": tfgnn.EdgeSet.from_fields(
                sizes=[tf.shape(edge_pairs)[0]],
                features={"weight": tf.expand_dims(edge_features, axis=-1)},
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("node", tf.cast(edge_pairs[:, 0], dtype=tf.int32)),
                    target=("node", tf.cast(edge_pairs[:, 1], dtype=tf.int32))
                )
            )
        }
    )

    # Regression labels
    avg_tokens_per_place = tf.constant(data["spn_allmus"], dtype=tf.float32)
    avg_firing_rates = tf.constant(data["spn_labda"], dtype=tf.float32)
    labels = tf.concat([avg_tokens_per_place, avg_firing_rates], axis=0)

    return graph, tf.expand_dims(labels, axis=-1)

def load_dataset(file_path: str) -> tf.data.Dataset:
    """
    Creates a tf.data.Dataset from a JSON-L file of SPN data.
    """
    graph_spec = tfgnn.GraphTensorSpec.from_piece_specs(
        node_sets_spec={
            'node': tfgnn.NodeSetSpec.from_field_specs(
                features_spec={'hidden_state': tf.TensorSpec(shape=(None, 3), dtype=tf.float32)},
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
                yield _parse_spn_json_and_build_graph(tf.constant(line))

    return tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            graph_spec,
            tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
        )
    )

def split_dataset(dataset: tf.data.Dataset, train_split=0.8, val_split=0.1, shuffle=True, seed=42):
    """Splits a dataset into training, validation, and test sets."""
    dataset_size = len(list(dataset.as_numpy_iterator()))
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    if shuffle:
        dataset = dataset.shuffle(dataset_size, seed=seed)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size).skip(val_size)

    return train_dataset, val_dataset, test_dataset
