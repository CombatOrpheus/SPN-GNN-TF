# tests/test_tf_dataset.py

import unittest
import os
import tensorflow as tf
import tensorflow_gnn as tfgnn
from spn_gnn_performance.tf_dataset import create_dataset_from_jsonl

class TestTFDataset(unittest.TestCase):

    def setUp(self):
        self.test_data_path = "tests/test_data.jsonl"

    def test_dataset_creation_and_structure(self):
        dataset = create_dataset_from_jsonl(self.test_data_path)

        for graph, labels in dataset.take(1):
            self.assertIsInstance(graph, tfgnn.GraphTensor)
            self.assertIsInstance(labels, tf.Tensor)

            # Check node features
            self.assertIn("hidden_state", graph.node_sets["node"].features)
            self.assertEqual(graph.node_sets["node"].features["hidden_state"].shape[1], 3)

            # Check edge features
            self.assertIn("weight", graph.edge_sets["edge"].features)
            self.assertEqual(graph.edge_sets["edge"].features["weight"].shape[1], 1)

            # Check labels
            self.assertEqual(labels.shape[1], 1)

            # Check number of nodes and labels
            self.assertEqual(graph.node_sets["node"].sizes[0], labels.shape[0])

if __name__ == "__main__":
    unittest.main()
