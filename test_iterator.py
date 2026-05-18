import tensorflow as tf
from src.spn_gnn_performance.tf_dataset import load_dataset
from src.spn_gnn_performance.baseline_models import prepare_dataset_for_baseline
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
dataset = load_dataset('tests/sample.jsonl')
# simulate run_tuning.py behavior
def _max_nodes_reduce(state, graph):
    num_nodes = graph.node_sets["node"].sizes[0]
    return tf.maximum(state, num_nodes)

max_nodes = dataset.reduce(tf.constant(0, dtype=tf.int32), _max_nodes_reduce).numpy()
