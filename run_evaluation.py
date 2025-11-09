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
from src.spn_gnn_performance.tf_dataset import load_dataset
from src.spn_gnn_performance.baseline_models import prepare_dataset_for_baseline
from src.spn_gnn_performance.models import GCNModel, GATModel, MPNNModel
import tensorflow_gnn as tfgnn

def calculate_metrics(y_true, y_pred):
    """Calculates and returns a dictionary of regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }

def save_plots(y_true, y_pred, is_place, output_dir):
    """Generates and saves evaluation plots."""
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
    parser.add_argument("--model-path", type=str, required=True, help="Path to the pre-trained model file.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the evaluation dataset in JSON-L format.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save output files (metrics and plots).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Loading model and data...")

    # Load model and infer type
    model_extension = os.path.splitext(args.model_path)[1]
    is_baseline_model = False
    if model_extension == ".keras":
        custom_objects = {"GCNModel": GCNModel, "GATModel": GATModel, "MPNNModel": MPNNModel}
        model = tf.keras.models.load_model(args.model_path, custom_objects=custom_objects)
        if not any(isinstance(layer, tfgnn.keras.layers.GraphUpdate) for layer in model.layers):
            is_baseline_model = True
    elif model_extension == ".joblib":
        model = joblib.load(args.model_path)
        is_baseline_model = True
    else:
        raise ValueError(f"Unsupported model file extension: {model_extension}")

    # Load and potentially prepare the dataset
    dataset = load_dataset(args.dataset_path)
    if is_baseline_model:
        eval_dataset = prepare_dataset_for_baseline(dataset)
    else:
        eval_dataset = dataset.batch(1)

    print("Model and data loaded successfully. Starting evaluation...")

    # Get predictions and true labels
    y_true_all, y_pred_all, is_place_all = [], [], []
    if is_baseline_model:
        for features, labels in eval_dataset:
            features_np = features.numpy().reshape(-1, features.shape[-1])
            labels_np = labels.numpy().ravel()
            mask = ~np.all(features_np == -1, axis=1)
            y_true_all.extend(labels_np[mask])
            is_place_all.extend(features_np[mask, 0] == 1)
            preds = model.predict(features)
            y_pred_all.extend(preds.ravel()[mask])
    else: # GNN model
        for graph, label in eval_dataset:
            y_true_all.extend(label.numpy().flatten())
            preds = model.predict_on_batch(graph)
            y_pred_all.extend(preds.numpy().flatten())
            node_features = graph.node_sets["node"]["hidden_state"].numpy()[0]
            is_place_all.extend(node_features[:, 0] == 1)

    y_true, y_pred, is_place = np.array(y_true_all), np.array(y_pred_all), np.array(is_place_all)
    is_transition = ~is_place

    # Calculate metrics
    print("Calculating metrics...")
    metrics = {
        "overall": calculate_metrics(y_true, y_pred),
        "place": calculate_metrics(y_true[is_place], y_pred[is_place]),
        "transition": calculate_metrics(y_true[is_transition], y_pred[is_transition]),
    }

    # Console output
    print("\n--- Evaluation Results ---")
    print(f"Overall:    MAE={metrics['overall']['mae']:.4f}, MSE={metrics['overall']['mse']:.4f}, MAPE={metrics['overall']['mape']:.4f}")
    print(f"Places:     MAE={metrics['place']['mae']:.4f}, MSE={metrics['place']['mse']:.4f}, MAPE={metrics['place']['mape']:.4f}")
    print(f"Transitions: MAE={metrics['transition']['mae']:.4f}, MSE={metrics['transition']['mse']:.4f}, MAPE={metrics['transition']['mape']:.4f}")

    # Save metrics to files
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"))

    # Generate and save plots
    save_plots(y_true, y_pred, is_place, args.output_dir)

    # Metrics bar chart
    metrics_df.plot(kind='bar', y=['mae', 'mse', 'mape'], figsize=(12, 7))
    plt.title('Performance Metrics by Node Type')
    plt.ylabel('Value')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_barchart.svg'))
    plt.close()

    print(f"\nOutputs saved to {args.output_dir}")

if __name__ == "__main__":
    # export PYTHONPATH=$PYTHONPATH:src
    main()
