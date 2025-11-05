# models.py - GNN models for performance evaluation of SPNs.
# A 2-layer architecture is a common and effective starting point for many
# graph-based tasks, as it allows each node to gather information from its local
# neighborhood (up to two hops away). This is supported by the general
# understanding of GNNs where the number of layers corresponds to the size of
# the receptive field of the nodes, as explained in resources like Distill's "A
# Gentle Introduction to Graph Neural Networks". While deeper GNNs exist, as
# shown in "Training Graph Neural Networks with 1000 Layers", they are not
# typically necessary for baseline models and require more advanced techniques.


import tensorflow as tf
import tensorflow_gnn as tfgnn

def dense_layer(units, activation='relu'):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation=activation),
    ])

class ConcatFeatures(tf.keras.layers.Layer):
    """A Keras layer to concatenate sender node and edge features."""
    def call(self, inputs):
        sender_node_features, edge_features = inputs
        return tf.concat([sender_node_features, edge_features], axis=-1)

def message_fn_factory(units, activation='relu'):
    return tf.keras.Sequential([
        ConcatFeatures(),
        tf.keras.layers.Dense(units, activation=activation),
    ])

class GCNModel(tf.keras.Model):
    """
    A Graph Convolutional Network (GCN) model for regression on SPNs.
    """
    def __init__(self, units, output_dim):
        super().__init__()
        self.gcn1 = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "node": tfgnn.keras.layers.NodeSetUpdate(
                    {"edge": tfgnn.keras.layers.SimpleConv(
                        message_fn=message_fn_factory(units),
                        reduce_type="sum",
                        sender_edge_feature=tfgnn.DEFAULT_FEATURE_NAME,
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(dense_layer(units))
                )
            }
        )
        self.gcn2 = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "node": tfgnn.keras.layers.NodeSetUpdate(
                    {"edge": tfgnn.keras.layers.SimpleConv(
                        message_fn=message_fn_factory(units),
                        reduce_type="sum",
                        sender_edge_feature=tfgnn.DEFAULT_FEATURE_NAME,
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(dense_layer(units))
                )
            }
        )
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        hidden_state = self.gcn1(graph)
        hidden_state = self.gcn2(hidden_state)
        return self.dense(hidden_state.node_sets["node"]["hidden_state"])


class GATModel(tf.keras.Model):
    """
    A Graph Attention Network (GAT) model for regression on SPNs.
    """
    def __init__(self, units, output_dim, num_heads=4):
        super().__init__()
        self.gat1 = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "node": tfgnn.keras.layers.NodeSetUpdate(
                    {"edge": tf.keras.Sequential([
                        tfgnn.keras.layers.GATv2Conv(
                            num_heads=num_heads,
                            per_head_channels=units // num_heads,
                            sender_edge_feature=tfgnn.DEFAULT_FEATURE_NAME),
                        tf.keras.layers.Flatten()
                    ])},
                    tfgnn.keras.layers.NextStateFromConcat(dense_layer(units))
                )
            }
        )
        self.gat2 = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "node": tfgnn.keras.layers.NodeSetUpdate(
                    {"edge": tf.keras.Sequential([
                        tfgnn.keras.layers.GATv2Conv(
                            num_heads=1,
                            per_head_channels=units,
                            sender_edge_feature=tfgnn.DEFAULT_FEATURE_NAME),
                        tf.keras.layers.Flatten()
                    ])},
                    tfgnn.keras.layers.NextStateFromConcat(dense_layer(units))
                )
            }
        )
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        hidden_state = self.gat1(graph)
        hidden_state = self.gat2(hidden_state)
        return self.dense(hidden_state.node_sets["node"]["hidden_state"])


class MPNNModel(tf.keras.Model):
    """
    A Message-Passing Neural Network (MPNN) model for regression on SPNs.
    This model uses a different activation function and internal dimensions
    to provide a distinct alternative to the GCNModel.
    """
    def __init__(self, output_dim, message_dim=64, next_state_dim=64):
        super().__init__()
        self.mpnn1 = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "node": tfgnn.keras.layers.NodeSetUpdate(
                    {"edge": tfgnn.keras.layers.SimpleConv(
                        message_fn=message_fn_factory(message_dim, 'tanh'),
                        reduce_type="sum",
                        sender_edge_feature=tfgnn.DEFAULT_FEATURE_NAME,
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(dense_layer(next_state_dim, 'tanh'))
                )
            }
        )
        self.mpnn2 = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "node": tfgnn.keras.layers.NodeSetUpdate(
                    {"edge": tfgnn.keras.layers.SimpleConv(
                        message_fn=message_fn_factory(message_dim, 'tanh'),
                        reduce_type="sum",
                        sender_edge_feature=tfgnn.DEFAULT_FEATURE_NAME,
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(dense_layer(next_state_dim, 'tanh'))
                )
            }
        )
        self.dense = tf.keras.layers.Dense(output_dim)

    def call(self, graph: tfgnn.GraphTensor) -> tf.Tensor:
        hidden_state = self.mpnn1(graph)
        hidden_state = self.mpnn2(hidden_state)
        return self.dense(hidden_state.node_sets["node"]["hidden_state"])
