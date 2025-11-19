# SPN-GNN Performance Evaluation

This project is dedicated to evaluating the performance of Graph Neural Networks (GNNs) for computing performance metrics of Stochastic Petri Nets (SPNs). The initial focus is on node regression tasks, where the goal is to predict performance metrics for each node in the SPN graph.

## Installation

This project uses `pixi` for package and environment management. To install `pixi`, run the following command in your terminal:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Once `pixi` is installed, you can install the project's dependencies by running the following command in the project's root directory:

```bash
pixi install
```

## Usage

This project provides three main scripts for hyperparameter tuning, training, and evaluation of the models.

**Note:** All scripts should be run with `pixi run python` to ensure the correct environment is used.

### 1. Hyperparameter Tuning

To run hyperparameter tuning for a specific model, use the `run_tuning.py` script. The following command will run Bayesian optimization to find the best hyperparameters for the GCN model using the provided dataset:

```bash
pixi run python run_tuning.py gcn path/to/dataset.jsonl
```

The script will save the best hyperparameters, all trial results, and a tuning plot to the `tuning_results` directory.

### 2. Training

Once you have tuned the hyperparameters, you can train a model using the `run_training.py` script. The following command will train a GCN model using the provided dataset and the best hyperparameters found in the tuning step:

```bash
pixi run python run_training.py gcn path/to/dataset.jsonl tuning_results/gcn_best_hps.json
```

The trained model will be saved to the `training_results` directory.

### 3. Evaluation

Finally, you can evaluate a trained model using the `run_evaluation.py` script. The following command will evaluate a trained GCN model on a given dataset:

```bash
pixi run python run_evaluation.py --model-type gcn --model-weights-path training_results/gcn_model.weights.h5 --hyperparameters-path tuning_results/gcn_best_hps.json --dataset-path path/to/dataset.jsonl --output-dir evaluation_results
```

The script will save the evaluation metrics (MAE, MSE, MAPE) and plots to the `evaluation_results` directory.

## Models

This project implements and evaluates the following models:

### GNN Models

*   **Graph Convolutional Network (GCN)**: A simple and effective GNN architecture for node-level predictions.
*   **Graph Attention Network (GAT)**: A GNN architecture that uses attention mechanisms to weigh the importance of different neighbors.
*   **Message Passing Neural Network (MPNN)**: A general framework for GNNs that involves message passing between nodes.

### Baseline Models

*   **Support Vector Machine (SVM)**: A classical machine learning model for regression.
*   **Multi-Layer Perceptron (MLP)**: A simple neural network architecture.
