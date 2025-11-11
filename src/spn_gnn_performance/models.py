# models.py - GNN models for performance evaluation of SPNs.

import tensorflow as tf
import tensorflow_gnn as tfgnn

def dense_layer(units, activation='relu'):
    """Creates a Sequential model with a single Dense layer."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation=activation),
    ])

def message_fn_factory(units, activation='relu'):
    """Creates a Sequential model for the message function in a SimpleConv."""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation=activation),
    ])

def GCNModel(graph_spec, units, output_dim):
    """Builds a GCN model using the Keras Functional API."""
    input_graph = tf.keras.Input(type_spec=graph_spec)
    graph = input_graph.merge_batch_to_components()

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"node": tfgnn.keras.layers.NodeSetUpdate(
            {"edge": tfgnn.keras.layers.SimpleConv(
                message_fn=message_fn_factory(units),
                reduce_type="sum",
                sender_edge_feature="weight",
                receiver_tag=tfgnn.TARGET)},
            tfgnn.keras.layers.NextStateFromConcat(dense_layer(units)))}
    )(graph)

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"node": tfgnn.keras.layers.NodeSetUpdate(
            {"edge": tfgnn.keras.layers.SimpleConv(
                message_fn=message_fn_factory(units),
                reduce_type="sum",
                sender_edge_feature="weight",
                receiver_tag=tfgnn.TARGET)},
            tfgnn.keras.layers.NextStateFromConcat(dense_layer(units)))}
    )(graph)

    node_features = graph.node_sets["node"]["hidden_state"]
    predictions = tf.keras.layers.Dense(output_dim)(node_features)
    return tf.keras.Model(inputs=input_graph, outputs=predictions)

def GATModel(graph_spec, units, output_dim, num_heads=4):
    """Builds a GAT model using the Keras Functional API."""
    input_graph = tf.keras.Input(type_spec=graph_spec)
    graph = input_graph.merge_batch_to_components()

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"node": tfgnn.keras.layers.NodeSetUpdate(
            {"edge": tf.keras.Sequential([
                tfgnn.keras.layers.GATv2Conv(
                    num_heads=num_heads,
                    per_head_channels=units // num_heads,
                    sender_edge_feature="weight"),
                tf.keras.layers.Flatten()])},
            tfgnn.keras.layers.NextStateFromConcat(dense_layer(units)))}
    )(graph)

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"node": tfgnn.keras.layers.NodeSetUpdate(
            {"edge": tf.keras.Sequential([
                tfgnn.keras.layers.GATv2Conv(
                    num_heads=1,
                    per_head_channels=units,
                    sender_edge_feature="weight"),
                tf.keras.layers.Flatten()])},
            tfgnn.keras.layers.NextStateFromConcat(dense_layer(units)))}
    )(graph)

    node_features = graph.node_sets["node"]["hidden_state"]
    predictions = tf.keras.layers.Dense(output_dim)(node_features)
    return tf.keras.Model(inputs=input_graph, outputs=predictions)

def MPNNModel(graph_spec, output_dim, message_dim=64, next_state_dim=64):
    """Builds an MPNN model using the Keras Functional API."""
    input_graph = tf.keras.Input(type_spec=graph_spec)
    graph = input_graph.merge_batch_to_components()

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"node": tfgnn.keras.layers.NodeSetUpdate(
            {"edge": tfgnn.keras.layers.SimpleConv(
                message_fn=message_fn_factory(message_dim, 'tanh'),
                reduce_type="sum",
                sender_edge_feature="weight",
                receiver_tag=tfgnn.TARGET)},
            tfgnn.keras.layers.NextStateFromConcat(dense_layer(next_state_dim, 'tanh')))}
    )(graph)

    graph = tfgnn.keras.layers.GraphUpdate(
        node_sets={"node": tfgnn.keras.layers.NodeSetUpdate(
            {"edge": tfgnn.keras.layers.SimpleConv(
                message_fn=message_fn_factory(message_dim, 'tanh'),
                reduce_type="sum",
                sender_edge_feature="weight",
                receiver_tag=tfgnn.TARGET)},
            tfgnn.keras.layers.NextStateFromConcat(dense_layer(next_state_dim, 'tanh')))}
    )(graph)

    node_features = graph.node_sets["node"]["hidden_state"]
    predictions = tf.keras.layers.Dense(output_dim)(node_features)
    return tf.keras.Model(inputs=input_graph, outputs=predictions)
