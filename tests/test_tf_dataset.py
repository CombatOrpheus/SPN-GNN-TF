# tests/test_tf_dataset.py

import unittest
import os
import tensorflow as tf
import tensorflow_gnn as tfgnn
from spn_gnn_performance.tf_dataset import load_dataset, load_heterogeneous_dataset

class TestTFDataset(unittest.TestCase):

    def setUp(self):
        self.test_data_path = "tests/test_data.jsonl"

    def test_dataset_creation_and_structure(self):
        dataset = load_dataset(self.test_data_path)

        for graph in dataset.take(1):
            self.assertIsInstance(graph, tfgnn.GraphTensor)

            # Check node features and labels
            self.assertIn("hidden_state", graph.node_sets["node"].features)
            self.assertIn("label", graph.node_sets["node"].features)
            self.assertEqual(graph.node_sets["node"].features["hidden_state"].shape[1], 3)
            self.assertEqual(graph.node_sets["node"].features["label"].shape[1], 1)

            # Check edge features
            self.assertIn("weight", graph.edge_sets["edge"].features)
            self.assertEqual(graph.edge_sets["edge"].features["weight"].shape[1], 1)

            # Check number of nodes and labels
            self.assertEqual(graph.node_sets["node"].sizes[0], graph.node_sets["node"]["label"].shape[0])

    def test_heterogeneous_dataset_creation_and_structure(self):
        dataset = load_heterogeneous_dataset(self.test_data_path)

        for graph in dataset.take(1):
            self.assertIsInstance(graph, tfgnn.GraphTensor)

            # Check node sets
            self.assertIn("place", graph.node_sets)
            self.assertIn("transition", graph.node_sets)

            # Check place node features and labels
            self.assertIn("hidden_state", graph.node_sets["place"].features)
            self.assertIn("label", graph.node_sets["place"].features)
            self.assertEqual(graph.node_sets["place"].features["hidden_state"].shape[1], 1)
            self.assertEqual(graph.node_sets["place"].features["label"].shape[1], 1)
            self.assertEqual(graph.node_sets["place"].sizes[0], graph.node_sets["place"]["label"].shape[0])

            # Check transition node features and labels
            self.assertIn("hidden_state", graph.node_sets["transition"].features)
            self.assertIn("label", graph.node_sets["transition"].features)
            self.assertEqual(graph.node_sets["transition"].features["hidden_state"].shape[1], 1)
            self.assertEqual(graph.node_sets["transition"].features["label"].shape[1], 1)
            self.assertEqual(graph.node_sets["transition"].sizes[0], graph.node_sets["transition"]["label"].shape[0])

            # Check edge sets
            self.assertIn("p_to_t", graph.edge_sets)
            self.assertIn("t_to_p", graph.edge_sets)

            # Check edge features
            self.assertIn("weight", graph.edge_sets["p_to_t"].features)
            self.assertEqual(graph.edge_sets["p_to_t"].features["weight"].shape[1], 1)
            self.assertIn("weight", graph.edge_sets["t_to_p"].features)
            self.assertEqual(graph.edge_sets["t_to_p"].features["weight"].shape[1], 1)

if __name__ == "__main__":
    unittest.main()
