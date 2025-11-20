import argparse
import os
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from src.spn_gnn_performance import tf_dataset, baseline_models, models
import tensorflow_gnn as tfgnn

def calculate_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of regression metrics.

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        dict: A dictionary of regression metrics (MAE, MSE, MAPE).
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return {"mae": float('nan'), "mse": float('nan'), "mape": float('nan')}
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }

def save_plots(y_true, y_pred, is_place, output_dir):
    """Generates and saves evaluation plots.

    Creates a scatter plot of predicted vs. true values and a residual plot.

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted values.
        is_place (np.ndarray): A boolean array indicating which nodes are
            places.
        output_dir (str): The directory to save the plots to.
    """
    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred, hue=np.where(is_place, 'Place', 'Transition'), alpha=0.6)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', linestyle='--')
    plt.title('Predicted vs. True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_true.svg'))
    plt.close()

    # Residual Plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, hue=np.where(is_place, 'Place', 'Transition'), alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (True - Predicted)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'residual_plot.svg'))
    plt.close()

def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Run evaluation on a pre-trained model.")
    parser.add_argument("--model-type", type=str, required=True, choices=["gcn", "gat", "mpnn", "svm", "mlp", "het_gcn", "het_graph_sage", "het_gat", "het_mpnn"], help="The model type to evaluate.")
    parser.add_argument("--model-weights-path", type=str, help="Path to the trained model weights file (.h5).")
    parser.add_argument("--model-path", type=str, help="Path to the trained scikit-learn model file (.joblib).")
    parser.add_argument("--hyperparameters-path", type=str, help="Path to the hyperparameters file for TF models.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the evaluation dataset in JSON-L format.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output files (metrics and plots).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading data...")

    # Load and potentially prepare the dataset
    is_baseline_model = args.model_type in ["svm", "mlp"]
    is_het_gnn = args.model_type in ["het_gcn", "het_graph_sage", "het_gat", "het_mpnn"]

    if is_het_gnn:
        dataset = tf_dataset.load_heterogeneous_dataset(args.dataset_path)
    else:
        dataset = tf_dataset.load_dataset(args.dataset_path)

    if is_baseline_model:
        eval_dataset = baseline_models.prepare_dataset_for_baseline(dataset)
    else:
        eval_dataset = dataset.batch(1)

    print("Data loaded successfully. Loading model...")

    # Load model
    if args.model_type == "svm":
        if not args.model_path:
            raise ValueError("SVM model requires --model-path.")
        model = joblib.load(args.model_path)
    elif args.model_type in ["gcn", "gat", "mpnn", "mlp", "het_gcn", "het_graph_sage", "het_gat", "het_mpnn"]:
        if not args.model_weights_path or not args.hyperparameters_path:
            raise ValueError("TensorFlow models require --model-weights-path and --hyperparameters-path.")
        with open(args.hyperparameters_path, 'r') as f:
            hps = json.load(f)

        if args.model_type == "mlp":
            model = models.build_and_compile_mlp(hps)
        else: # GNN models
            graph_spec = dataset.element_spec
            if is_het_gnn:
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
            else:
                features_spec = dict(graph_spec.node_sets_spec['node'].features_spec)
                del features_spec['label']
                input_graph_spec = tfgnn.GraphTensorSpec.from_piece_specs(
                    edge_sets_spec=graph_spec.edge_sets_spec,
                    node_sets_spec={'node': tfgnn.NodeSetSpec.from_field_specs(
                        features_spec=features_spec,
                        sizes_spec=graph_spec.node_sets_spec['node'].sizes_spec)}
                )
            builder_fn = getattr(models, f"build_and_compile_{args.model_type}")
            model = builder_fn(input_graph_spec, hps)

        model.load_weights(args.model_weights_path)
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    print("Model loaded successfully. Starting evaluation...")

    y_true_all, y_pred_all, is_place_all = [], [], []
    if is_baseline_model:
        # ... (same as before)
        print("Evaluating baseline model (not implemented fully in this snippet but keeping logic structure)...")
        # Assuming eval_dataset is (X, y) tuples
        for X_batch, y_batch in eval_dataset:
             preds = model.predict(X_batch)
             y_true_all.extend(y_batch)
             y_pred_all.extend(preds)
             # Can't determine place/transition easily for baseline without more info, skipping breakdown
             is_place_all.extend([False] * len(y_batch)) # Dummy
    elif is_het_gnn:
        for graph in eval_dataset:
            place_labels = graph.node_sets['place']['label'].values.numpy().flatten()
            transition_labels = graph.node_sets['transition']['label'].values.numpy().flatten()
            y_true_all.extend(place_labels)
            y_true_all.extend(transition_labels)

            is_place_all.extend([True] * len(place_labels))
            is_place_all.extend([False] * len(transition_labels))

            features_place = graph.node_sets['place'].get_features_dict()
            features_transition = graph.node_sets['transition'].get_features_dict()
            del features_place['label']
            del features_transition['label']
            graph = graph.replace_features(node_sets={'place': features_place, 'transition': features_transition})

            preds = model.predict_on_batch(graph)
            y_pred_all.extend(preds['place'].flatten())
            y_pred_all.extend(preds['transition'].flatten())
    else: # Homogeneous GNN model
        for graph in eval_dataset:
            labels = graph.node_sets['node']['label']
            y_true_all.extend(labels.values.numpy().flatten())
            features = graph.node_sets['node'].get_features_dict()
            del features['label']
            graph = graph.replace_features(node_sets={'node': features})
            preds = model.predict_on_batch(graph)
            y_pred_all.extend(preds.flatten())
            node_features = graph.node_sets["node"]["hidden_state"].numpy()[0]
            is_place_all.extend(node_features[:, 0] == 1)

    y_true, y_pred, is_place = np.array(y_true_all), np.array(y_pred_all), np.array(is_place_all)
    is_place = is_place.astype(bool)
    is_transition = ~is_place

    print("Calculating metrics...")
    metrics = {
        "overall": calculate_metrics(y_true, y_pred),
        "place": calculate_metrics(y_true[is_place], y_pred[is_place]),
        "transition": calculate_metrics(y_true[is_transition], y_pred[is_transition]),
    }

    print("\n--- Evaluation Results ---")
    print(f"Overall:    MAE={metrics['overall']['mae']:.4f}, MSE={metrics['overall']['mse']:.4f}, MAPE={metrics['overall']['mape']:.4f}")
    print(f"Places:     MAE={metrics['place']['mae']:.4f}, MSE={metrics['place']['mse']:.4f}, MAPE={metrics['place']['mape']:.4f}")
    print(f"Transitions: MAE={metrics['transition']['mae']:.4f}, MSE={metrics['transition']['mse']:.4f}, MAPE={metrics['transition']['mape']:.4f}")

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"))

    save_plots(y_true, y_pred, is_place, args.output_dir)

    metrics_df.plot(kind='bar', y=['mae', 'mse', 'mape'], figsize=(12, 7))
    plt.title('Performance Metrics by Node Type')
    plt.ylabel('Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_barchart.svg'))
    plt.close()

    print(f"\nOutputs saved to {args.output_dir}")

if __name__ == "__main__":
    main()
