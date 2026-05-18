import tensorflow as tf
from src.spn_gnn_performance.tf_dataset import load_dataset, split_dataset
from src.spn_gnn_performance.baseline_models import prepare_dataset_for_baseline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = load_dataset('tests/sample.jsonl')
for _ in dataset:
    pass

train, val, test = split_dataset(dataset)
dataset = prepare_dataset_for_baseline(train)
data = list(dataset.as_numpy_iterator())
