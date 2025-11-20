import unittest
import os
import shutil
import subprocess
import json

class BaseIntegrationTest(unittest.TestCase):
    """
    Base class for integration tests, handles setup and teardown of test artifacts.
    """
    TUNING_RESULTS_DIR = "tuning_results"
    TRAINING_RESULTS_DIR = "training_results"
    EVALUATION_RESULTS_DIR = "evaluation_results"
    DATASET_PATH = "tests/sample.jsonl"

    def setUp(self):
        """Set up test environment, ensuring no leftover directories."""
        self.cleanup()

    def tearDown(self):
        """Clean up test environment after tests have run."""
        self.cleanup()

    def cleanup(self):
        """Remove directories created during tests."""
        for directory in [self.TUNING_RESULTS_DIR, self.TRAINING_RESULTS_DIR, self.EVALUATION_RESULTS_DIR]:
            if os.path.exists(directory):
                shutil.rmtree(directory)

    def run_script(self, command):
        """Run a script using subprocess and handle errors."""
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:src"
        process = subprocess.run(command, capture_output=True, text=True, env=env)
        if process.returncode != 0:
            self.fail(f"Script {' '.join(command)} failed with exit code {process.returncode}\n"
                      f"--- STDOUT ---\n{process.stdout}\n"
                      f"--- STDERR ---\n{process.stderr}")
        return process

class TestHetGCNIntegration(BaseIntegrationTest):
    def test_het_gcn_workflow(self):
        self.run_workflow("het_gcn")

    def test_het_graph_sage_workflow(self):
        self.run_workflow("het_graph_sage")

    def test_het_gat_workflow(self):
        self.run_workflow("het_gat")

    def test_het_mpnn_workflow(self):
        self.run_workflow("het_mpnn")

    def run_workflow(self, model_name):
        # 1. Run tuning
        self.run_script(["python", "run_tuning.py", model_name, self.DATASET_PATH])
        hyperparameters_path = os.path.join(self.TUNING_RESULTS_DIR, f"{model_name}_best_hps.json")
        self.assertTrue(os.path.exists(hyperparameters_path), f"Hyperparameters file not found for {model_name}")

        # 2. Run training
        self.run_script(["python", "run_training.py", model_name, self.DATASET_PATH, hyperparameters_path, "--epochs", "1", "--batch_size", "1"])
        model_weights_path = os.path.join(self.TRAINING_RESULTS_DIR, f"{model_name}_model.weights.h5")
        self.assertTrue(os.path.exists(model_weights_path), f"Model weights file not found for {model_name}")

        # 3. Run evaluation
        evaluation_output_dir = os.path.join(self.EVALUATION_RESULTS_DIR, model_name)
        self.run_script([
            "python", "run_evaluation.py",
            "--model-type", model_name,
            "--model-weights-path", model_weights_path,
            "--hyperparameters-path", hyperparameters_path,
            "--dataset-path", self.DATASET_PATH,
            "--output-dir", evaluation_output_dir
        ])
        metrics_path = os.path.join(evaluation_output_dir, "metrics.json")
        self.assertTrue(os.path.exists(metrics_path), f"Metrics file not found for {model_name}")

        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        self.assertIn("overall", metrics)
        self.assertIn("mae", metrics["overall"])
        self.assertIsNotNone(metrics["overall"]["mae"])

if __name__ == '__main__':
    unittest.main()
