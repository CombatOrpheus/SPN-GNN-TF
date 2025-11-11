import os
import json
import unittest
import subprocess
import shutil

class TestRunTraining(unittest.TestCase):

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
        if os.path.exists("training_results"):
            shutil.rmtree("training_results")

    def test_run_training_gcn(self):
        # Run the training script with GCN model
        subprocess.run([
            "python", "run_training.py",
            "gcn",
            "tests/test_data.jsonl",
            "test_hps.json",
            "--epochs", "1"
        ], check=True)

        # Check that the training_results directory was created
        self.assertTrue(os.path.exists("training_results"))

if __name__ == "__main__":
    unittest.main()
