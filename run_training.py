# run_training.py - Train a model with tuned hyperparameters.

import argparse
import json
import os
import tensorflow as tf
from spn_gnn_performance import tf_dataset, baseline_models, models
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error


def build_and_compile_gcn(hps):
    """Builds and compiles a GCN model from hyperparameters."""
    model = models.GCNModel(units=hps['units'], output_dim=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hps['learning_rate']),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model

def build_and_compile_gat(hps):
    """Builds and compiles a GAT model from hyperparameters."""
    model = models.GATModel(units=hps['units'], output_dim=1, num_heads=hps['num_heads'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hps['learning_rate']),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model

def build_and_compile_mpnn(hps):
    """Builds and compiles an MPNN model from hyperparameters."""
    model = models.MPNNModel(output_dim=1, message_dim=hps['message_dim'], next_state_dim=hps['next_state_dim'])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hps['learning_rate']),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model

def build_and_compile_mlp(hps):
    """Builds and compiles an MLP model from hyperparameters."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(None, 7)))  # Shape from prepare_dataset_for_baseline
    model.add(tf.keras.layers.Masking(mask_value=-1.))
    for _ in range(hps['num_layers']):
        model.add(tf.keras.layers.Dense(hps['units'], activation=hps['activation']))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hps['learning_rate']),
                  loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a model with tuned hyperparameters.")
    parser.add_argument("model", choices=["gcn", "gat", "mpnn", "svm", "mlp"], help="The model to train.")
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
    dataset = tf_dataset.load_dataset(args.dataset_path)
    train_dataset, val_dataset, test_dataset = tf_dataset.split_dataset(dataset, train_split=0.8, val_split=0.1)
    print("Dataset loaded and split.")

    if args.model in ["gcn", "gat", "mpnn", "mlp"]:
        if args.model == "mlp":
            print("Preparing dataset for MLP baseline...")
            train_dataset = baseline_models.prepare_dataset_for_baseline(train_dataset)
            val_dataset = baseline_models.prepare_dataset_for_baseline(val_dataset)
            test_dataset = baseline_models.prepare_dataset_for_baseline(test_dataset)
            print("Dataset prepared.")

        builder_map = {
            "gcn": build_and_compile_gcn,
            "gat": build_and_compile_gat,
            "mpnn": build_and_compile_mpnn,
            "mlp": build_and_compile_mlp,
        }
        model = builder_map[args.model](hps)

        print(f"Training {args.model.upper()} model for {args.epochs} epochs...")
        model.fit(
            train_dataset.batch(args.batch_size),
            validation_data=val_dataset.batch(args.batch_size),
            epochs=args.epochs,
            verbose=2
        )
        print("Training complete.")

        print("Evaluating model on the test set...")
        loss, mse, mae = model.evaluate(test_dataset.batch(args.batch_size))
        print(f"Test MAPE (Loss): {loss:.4f}")
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")

        print(f"Saving model to {model_save_path}...")
        model.save(model_save_path)
        print("Model saved.")

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
