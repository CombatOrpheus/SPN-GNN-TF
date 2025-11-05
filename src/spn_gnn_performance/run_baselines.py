# run_baselines.py - Training and evaluation script for baseline models.

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from spn_gnn_performance.tf_dataset import create_dataset_from_jsonl
from spn_gnn_performance.baseline_models import SVMModel, MLPModel, prepare_dataset_for_baseline

def main():
    # Load and prepare the dataset
    file_path = "tests/test_data.jsonl"
    dataset = create_dataset_from_jsonl(file_path)
    padded_dataset = prepare_dataset_for_baseline(dataset)

    # Convert the dataset to numpy arrays for scikit-learn
    X_list, y_list = [], []
    for features, labels in padded_dataset:
        X_list.append(features.numpy())
        y_list.append(labels.numpy())
    X = np.array(X_list)
    y = np.array(y_list)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the SVM model
    print("Training SVM model...")
    svm_model = SVMModel()
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)

    # Flatten y_test and y_pred_svm for metric calculation, and filter out padded values
    y_test_flat = y_test.ravel()
    y_pred_svm_flat = y_pred_svm.ravel()
    mask = y_test_flat != -1

    mse_svm = mean_squared_error(y_test_flat[mask], y_pred_svm_flat[mask])
    print(f"SVM Mean Squared Error: {mse_svm}")

    # Train and evaluate the MLP model
    print("\nTraining MLP model...")
    input_shape = X_train.shape[1:]
    mlp_model = MLPModel(input_shape=input_shape, epochs=1)
    mlp_model.fit(X_train, y_train)
    y_pred_mlp = mlp_model.predict(X_test)

    # Flatten y_pred_mlp for metric calculation
    y_pred_mlp_flat = y_pred_mlp.ravel()

    mse_mlp = mean_squared_error(y_test_flat[mask], y_pred_mlp_flat[mask])
    print(f"MLP Mean Squared Error: {mse_mlp}")

if __name__ == "__main__":
    main()
