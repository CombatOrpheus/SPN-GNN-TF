# tests/test_baseline_models.py

import unittest
import numpy as np
import tensorflow as tf
from spn_gnn_performance.tf_dataset import create_dataset_from_jsonl
from spn_gnn_performance.baseline_models import SVMModel, MLPModel, prepare_dataset_for_baseline

class TestBaselineModels(unittest.TestCase):

    def setUp(self):
        self.test_data_path = "tests/test_data.jsonl"
        self.dataset = create_dataset_from_jsonl(self.test_data_path)
        self.padded_dataset = prepare_dataset_for_baseline(self.dataset)

        # Convert to numpy for testing properties
        X_list, y_list = [], []
        for features, labels in self.padded_dataset:
            X_list.append(features.numpy())
            y_list.append(labels.numpy())
        self.X = np.array(X_list)
        self.y = np.array(y_list)

    def test_prepare_dataset_for_baseline(self):
        self.assertIsInstance(self.padded_dataset, tf.data.Dataset)

        # Check shapes and types from the numpy-converted data
        self.assertEqual(self.X.shape[0], self.y.shape[0])
        self.assertEqual(self.X.shape[1], self.y.shape[1])
        self.assertEqual(self.X.shape[2], 7) # 3 original + 4 engineered
        self.assertEqual(self.y.shape[2], 1)

    def test_svm_model(self):
        model = SVMModel()
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0] * self.X.shape[1],))

    def test_mlp_model(self):
        input_shape = self.X.shape[1:]
        model = MLPModel(input_shape=input_shape, epochs=1)
        model.fit(self.X, self.y)
        predictions = model.predict(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0], self.X.shape[1], 1))

if __name__ == "__main__":
    unittest.main()
