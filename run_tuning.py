# run_tuning.py - Run hyperparameter tuning for all models.

import argparse
import json
import os
import pandas as pd
import keras_tuner as kt
from spn_gnn_performance import tuning, tf_dataset, baseline_models
import matplotlib.pyplot as plt
import tensorflow_gnn as tfgnn
import numpy as np

def main():
    """Runs hyperparameter tuning for a specified model.

    This script tunes the hyperparameters of a given model using a specified
    dataset. The tuning results, including the best hyperparameters, all trial
    results, and a tuning plot, are saved to the 'tuning_results' directory.

    Command-line arguments:
        model (str): The model to tune (choices: "gcn", "gat", "mpnn", "svm",
            "mlp").
        dataset_path (str): Path to the JSON-L dataset.
    """
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for SPN GNN models.")
    parser.add_argument("model", choices=["gcn", "gat", "mpnn", "svm", "mlp"], help="The model to tune.")
    parser.add_argument("dataset_path", help="Path to the JSON-L dataset.")
    args = parser.parse_args()

    # Create results directory
    os.makedirs("tuning_results", exist_ok=True)

    if args.model in ["gcn", "gat", "mpnn", "mlp"]:
        # Load dataset
        dataset = tf_dataset.load_dataset(args.dataset_path)
        train_dataset, val_dataset, test_dataset = tf_dataset.split_dataset(dataset)

        def extract_labels(graph: tfgnn.GraphTensor):
            labels = graph.node_sets['node']['label']
            features = graph.node_sets['node'].get_features_dict()
            del features['label']
            graph = graph.replace_features(node_sets={'node': features})
            return graph, labels.values

        train_dataset_with_labels = train_dataset.batch(32).map(extract_labels)
        val_dataset_with_labels = val_dataset.batch(32).map(extract_labels)

        # For GNN models, we need the graph_spec to build the model.
        graph_spec = train_dataset.element_spec
        features_spec = dict(graph_spec.node_sets_spec['node'].features_spec)
        del features_spec['label']
        input_graph_spec = tfgnn.GraphTensorSpec.from_piece_specs(
            edge_sets_spec=graph_spec.edge_sets_spec,
            node_sets_spec={'node': tfgnn.NodeSetSpec.from_field_specs(
                features_spec=features_spec,
                sizes_spec=graph_spec.node_sets_spec['node'].sizes_spec)}
        )

        if args.model == "gcn":
            build_fn = tuning.build_gcn_model(input_graph_spec)
        elif args.model == "gat":
            build_fn = tuning.build_gat_model(input_graph_spec)
        elif args.model == "mpnn":
            build_fn = tuning.build_mpnn_model(input_graph_spec)
        elif args.model == "mlp":
            train_dataset = baseline_models.prepare_dataset_for_baseline(train_dataset)
            val_dataset = baseline_models.prepare_dataset_for_baseline(val_dataset)
            build_fn = tuning.build_mlp_model

        tuner = kt.BayesianOptimization(
            build_fn,
            objective="val_mae",
            max_trials=1,
            directory="tuning_results",
            project_name=args.model,
        )

        tuner.search(train_dataset_with_labels, epochs=1, validation_data=val_dataset_with_labels, validation_steps=1)

        # Save best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        with open(f"tuning_results/{args.model}_best_hps.json", "w") as f:
            json.dump(best_hps.values, f, indent=4)

        # Save all trials
        df = pd.DataFrame([trial.hyperparameters.values for trial in tuner.oracle.trials.values()])
        df['score'] = [trial.score for trial in tuner.oracle.trials.values()]
        df.to_csv(f"tuning_results/{args.model}_trials.csv", index=False)

        # Plot tuning process
        plt.figure(figsize=(10, 6))
        plt.plot(df['score'])
        plt.title(f"{args.model.upper()} Hyperparameter Tuning")
        plt.xlabel("Trial")
        plt.ylabel("Validation Loss (MAPE)")
        plt.savefig(f"tuning_results/{args.model}_tuning_plot.png")

        print(f"Best hyperparameters for {args.model}:")
        print(best_hps.values)

    elif args.model == "svm":
        dataset = tf_dataset.load_dataset(args.dataset_path)
        train_dataset, _, _ = tf_dataset.split_dataset(dataset)

        # This can be memory intensive, consider using a smaller subset for tuning
        data = list(baseline_models.prepare_dataset_for_baseline(train_dataset).as_numpy_iterator())
        X_train = np.array([item[0] for item in data])
        y_train = np.array([item[1] for item in data])

        best_params, cv_results = tuning.tune_svm_model(X_train, y_train)

        # Save best hyperparameters
        with open(f"tuning_results/{args.model}_best_hps.json", "w") as f:
            json.dump(best_params, f, indent=4)

        # Save all trials
        df = pd.DataFrame(cv_results)
        df.to_csv(f"tuning_results/{args.model}_trials.csv", index=False)

        # Plot tuning process
        plt.figure(figsize=(10, 6))
        plt.plot(df['mean_test_score'] * -1) # Invert the score back to positive
        plt.title(f"{args.model.upper()} Hyperparameter Tuning")
        plt.xlabel("Iteration")
        plt.ylabel("Mean Test Score (MAPE)")
        plt.savefig(f"tuning_results/{args.model}_tuning_plot.png")

        print(f"Best hyperparameters for {args.model}:")
        print(best_params)


if __name__ == "__main__":
    main()
