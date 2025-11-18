# models.py - GNN models for performance evaluation of SPNs.

import tensorflow as tf
import tensorflow_gnn as tfgnn

def dense_layer(units, activation='relu'):
    """Creates a Sequential model with a single Dense layer.

    Args:
        units (int): The number of units in the Dense layer.
        activation (str, optional): The activation function to use.
            Defaults to 'relu'.

    Returns:
        tf.keras.Sequential: A Sequential model containing a single Dense layer.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation=activation),
    ])

def message_fn_factory(units, activation='relu'):
    """Creates a Sequential model for the message function in a SimpleConv.

    Args:
        units (int): The number of units in the Dense layer.
        activation (str, optional): The activation function to use.
            Defaults to 'relu'.

    Returns:
        tf.keras.Sequential: A Sequential model for the message function.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation=activation),
    ])

def GCNModel(graph_spec, units, output_dim):
    """Builds a GCN model using the Keras Functional API.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.
        units (int): The number of units in the GCN layers.
        output_dim (int): The dimension of the output predictions.

    Returns:
        tf.keras.Model: A GCN model.
    """
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

def build_and_compile_gcn(graph_spec, hps):
    """Builds and compiles a GCN model from hyperparameters.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.
        hps (dict): A dictionary of hyperparameters.

    Returns:
        tf.keras.Model: A compiled GCN model.
    """
    model = GCNModel(graph_spec, units=hps['units'], output_dim=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hps['learning_rate']),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model

def build_and_compile_gat(graph_spec, hps):
    """Builds and compiles a GAT model from hyperparameters.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.
        hps (dict): A dictionary of hyperparameters.

    Returns:
        tf.keras.Model: A compiled GAT model.
    """
    model = GATModel(graph_spec, units=hps['units'], output_dim=1, num_heads=hps['num_heads'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hps['learning_rate']),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model

def build_and_compile_mpnn(graph_spec, hps):
    """Builds and compiles an MPNN model from hyperparameters.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.
        hps (dict): A dictionary of hyperparameters.

    Returns:
        tf.keras.Model: A compiled MPNN model.
    """
    model = MPNNModel(graph_spec, output_dim=1, message_dim=hps['message_dim'], next_state_dim=hps['next_state_dim'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hps['learning_rate']),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model

def build_and_compile_mlp(hps):
    """Builds and compiles an MLP model from hyperparameters.

    Args:
        hps (dict): A dictionary of hyperparameters.

    Returns:
        tf.keras.Model: A compiled MLP model.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(None, 7)))  # Shape from prepare_dataset_for_baseline
    model.add(tf.keras.layers.Masking(mask_value=-1.))
    for _ in range(hps['num_layers']):
        model.add(tf.keras.layers.Dense(hps['units'], activation=hps['activation']))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hps['learning_rate']),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model

def GATModel(graph_spec, units, output_dim, num_heads=4):
    """Builds a GAT model using the Keras Functional API.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.
        units (int): The number of units in the GAT layers.
        output_dim (int): The dimension of the output predictions.
        num_heads (int, optional): The number of attention heads. Defaults to 4.

    Returns:
        tf.keras.Model: A GAT model.
    """
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
    """Builds an MPNN model using the Keras Functional API.

    Args:
        graph_spec (tfgnn.GraphTensorSpec): The spec of the input graph.
        output_dim (int): The dimension of the output predictions.
        message_dim (int, optional): The dimension of the message vectors.
            Defaults to 64.
        next_state_dim (int, optional): The dimension of the next state vectors.
            Defaults to 64.

    Returns:
        tf.keras.Model: An MPNN model.
    """
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
