# run_training.py - Train a model with tuned hyperparameters.

import argparse
import json
import os
import tensorflow as tf
import tensorflow_gnn as tfgnn
from src.spn_gnn_performance import tf_dataset, baseline_models, models
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

def main():
    """Trains a model with tuned hyperparameters.

    This script trains a specified model using a given dataset and a JSON file
    containing tuned hyperparameters. The trained model is saved to the
    'training_results' directory.

    Command-line arguments:
        model (str): The model to train (choices: "gcn", "gat", "mpnn",
            "svm", "mlp").
        dataset_path (str): Path to the JSON-L dataset.
        hyperparameters_path (str): Path to the JSON file with tuned
            hyperparameters.
        --epochs (int, optional): Number of training epochs. Defaults to 50.
        --batch_size (int, optional): Batch size for training. Defaults to 32.
    """
    parser = argparse.ArgumentParser(description="Train a model with tuned hyperparameters.")
    parser.add_argument("model", choices=["gcn", "gat", "mpnn", "svm", "mlp", "het_gcn"], help="The model to train.")
    parser.add_argument("dataset_path", help="Path to the JSON-L dataset.")
    parser.add_argument("hyperparameters_path", help="Path to the JSON file with tuned hyperparameters.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()

    # Create results directory
    os.makedirs("training_results", exist_ok=True)
    model_save_path = f"training_results/{args.model}_model"

    # Load hyperparameters
    with open(args.hyperparameters_path, 'r') as f:
        hps = json.load(f)

    # Load and split dataset
    print("Loading and splitting the dataset...")
    if args.model == "het_gcn":
        dataset = tf_dataset.load_heterogeneous_dataset(args.dataset_path)
    else:
        dataset = tf_dataset.load_dataset(args.dataset_path)
    train_dataset, val_dataset, test_dataset = tf_dataset.split_dataset(dataset, train_split=0.8, val_split=0.1)
    print("Dataset loaded and split.")

    if args.model in ["gcn", "gat", "mpnn", "mlp", "het_gcn"]:
        if args.model == "mlp":
            print("Preparing dataset for MLP baseline...")
            train_dataset = baseline_models.prepare_dataset_for_baseline(train_dataset)
            val_dataset = baseline_models.prepare_dataset_for_baseline(val_dataset)
            test_dataset = baseline_models.prepare_dataset_for_baseline(test_dataset)
            print("Dataset prepared.")

            model = models.build_and_compile_mlp(hps)
            train_dataset_with_labels = train_dataset.batch(args.batch_size)
            val_dataset_with_labels = val_dataset.batch(args.batch_size)
            test_dataset_with_labels = test_dataset.batch(args.batch_size)

        else: # GNN models
            graph_spec = train_dataset.element_spec
            if args.model == "het_gcn":
                features_spec_place = dict(graph_spec.node_sets_spec['place'].features_spec)
                features_spec_transition = dict(graph_spec.node_sets_spec['transition'].features_spec)
                del features_spec_place['label']
                del features_spec_transition['label']
                input_graph_spec = tfgnn.GraphTensorSpec.from_piece_specs(
                    edge_sets_spec=graph_spec.edge_sets_spec,
                    node_sets_spec={
                        'place': tfgnn.NodeSetSpec.from_field_specs(
                            features_spec=features_spec_place,
                            sizes_spec=graph_spec.node_sets_spec['place'].sizes_spec),
                        'transition': tfgnn.NodeSetSpec.from_field_specs(
                            features_spec=features_spec_transition,
                            sizes_spec=graph_spec.node_sets_spec['transition'].sizes_spec)
                    }
                )

                def extract_labels_het(graph: tfgnn.GraphTensor):
                    labels = {
                        "place": graph.node_sets["place"]["label"],
                        "transition": graph.node_sets["transition"]["label"]
                    }
                    features_place = graph.node_sets['place'].get_features_dict()
                    features_transition = graph.node_sets['transition'].get_features_dict()
                    del features_place['label']
                    del features_transition['label']
                    graph = graph.replace_features(node_sets={'place': features_place, 'transition': features_transition})
                    return graph, labels

                train_dataset_with_labels = train_dataset.batch(args.batch_size).map(extract_labels_het)
                val_dataset_with_labels = val_dataset.batch(args.batch_size).map(extract_labels_het)
                test_dataset_with_labels = test_dataset.batch(args.batch_size).map(extract_labels_het)

            else: # Homogeneous GNN models
                features_spec = dict(graph_spec.node_sets_spec['node'].features_spec)
                del features_spec['label']
                input_graph_spec = tfgnn.GraphTensorSpec.from_piece_specs(
                    edge_sets_spec=graph_spec.edge_sets_spec,
                    node_sets_spec={'node': tfgnn.NodeSetSpec.from_field_specs(
                        features_spec=features_spec,
                        sizes_spec=graph_spec.node_sets_spec['node'].sizes_spec)}
                )

                def extract_labels(graph: tfgnn.GraphTensor):
                    labels = graph.node_sets['node']['label']
                    features = graph.node_sets['node'].get_features_dict()
                    del features['label']
                    graph = graph.replace_features(node_sets={'node': features})
                    return graph, labels.values

                train_dataset_with_labels = train_dataset.batch(args.batch_size).map(extract_labels)
                val_dataset_with_labels = val_dataset.batch(args.batch_size).map(extract_labels)
                test_dataset_with_labels = test_dataset.batch(args.batch_size).map(extract_labels)

            builder_fn = getattr(models, f"build_and_compile_{args.model}")
            model = builder_fn(input_graph_spec, hps)

        print(f"Training {args.model.upper()} model for {args.epochs} epochs...")
        model.fit(
            train_dataset_with_labels,
            validation_data=val_dataset_with_labels,
            epochs=args.epochs,
            verbose=2
        )
        print("Training complete.")

        print("Evaluating model on the test set...")
        results = model.evaluate(test_dataset_with_labels, return_dict=True)
        print("Test evaluation results:")
        for key, value in results.items():
            print(f"  {key}: {value:.4f}")

        print(f"Saving model weights to {model_save_path}.weights.h5...")
        model.save_weights(f"{model_save_path}.weights.h5")
        print("Model weights saved.")

    elif args.model == "svm":
        print("Preparing dataset for SVM baseline...")
        train_data = list(baseline_models.prepare_dataset_for_baseline(train_dataset).as_numpy_iterator())
        X_train, y_train = np.array([item[0] for item in train_data]), np.array([item[1] for item in train_data])

        test_data = list(baseline_models.prepare_dataset_for_baseline(test_dataset).as_numpy_iterator())
        X_test, y_test = np.array([item[0] for item in test_data]), np.array([item[1] for item in test_data])
        print("Dataset prepared.")

        print("Training SVM model...")
        svm_model = baseline_models.SVMModel(**hps)
        svm_model.fit(X_train, y_train)
        print("Training complete.")

        print("Evaluating SVM model...")
        y_pred = svm_model.predict(X_test)

        y_test_flat = y_test.ravel()
        mask = y_test_flat != -1
        y_test_filtered = y_test_flat[mask]

        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_mask = ~np.all(X_test_flat == -1, axis=1)
        y_pred_filtered = y_pred[X_mask]

        mape = mean_absolute_percentage_error(y_test_filtered, y_pred_filtered)
        mse = mean_squared_error(y_test_filtered, y_pred_filtered)
        mae = mean_absolute_error(y_test_filtered, y_pred_filtered)

        print(f"Test MAPE: {mape:.4f}")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")

        print(f"Saving model to {model_save_path}.joblib...")
        joblib.dump(svm_model, f"{model_save_path}.joblib")
        print("Model saved.")

    print("Training pipeline finished.")

if __name__ == "__main__":
    main()
