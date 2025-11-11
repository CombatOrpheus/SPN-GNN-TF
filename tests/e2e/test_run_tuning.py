import os
import json
import unittest
import subprocess
import shutil

class TestRunTuning(unittest.TestCase):

    def tearDown(self):
        # Clean up created files and directories
        if os.path.exists("tuning_results"):
            shutil.rmtree("tuning_results")

    def test_run_tuning_gcn(self):
        # Run the tuning script with GCN model
        subprocess.run([
            "python", "run_tuning.py",
            "gcn",
            "tests/test_data.jsonl"
        ], check=True)

        # Check that the results were saved
        self.assertTrue(os.path.exists("tuning_results/gcn_best_hps.json"))
        self.assertTrue(os.path.exists("tuning_results/gcn_trials.csv"))
        self.assertTrue(os.path.exists("tuning_results/gcn_tuning_plot.png"))

if __name__ == "__main__":
    unittest.main()
