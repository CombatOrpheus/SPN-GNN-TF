import os
import json
import unittest
import subprocess
import shutil

class TestRunEvaluation(unittest.TestCase):

    def setUp(self):
        # Create a dummy hyperparameters file
        self.hps = {
            "units": 16,
            "learning_rate": 0.01
        }
        with open("test_hps.json", "w") as f:
            json.dump(self.hps, f)

    def tearDown(self):
        # Clean up created files and directories
        if os.path.exists("test_hps.json"):
            os.remove("test_hps.json")
        if os.path.exists("evaluation_results"):
            shutil.rmtree("evaluation_results")

    def test_run_evaluation_gcn(self):
        # Run the evaluation script with GCN model
        subprocess.run([
            "python", "run_evaluation.py",
            "--model-type", "gcn",
            "--hyperparameters-path", "test_hps.json",
            "--dataset-path", "tests/test_data.jsonl",
            "--output-dir", "evaluation_results"
        ], check=True)

        # Check that the results were saved
        self.assertTrue(os.path.exists("evaluation_results/metrics.json"))
        self.assertTrue(os.path.exists("evaluation_results/metrics.csv"))
        self.assertTrue(os.path.exists("evaluation_results/predicted_vs_true.svg"))
        self.assertTrue(os.path.exists("evaluation_results/residual_plot.svg"))
        self.assertTrue(os.path.exists("evaluation_results/metrics_barchart.svg"))

if __name__ == "__main__":
    unittest.main()
