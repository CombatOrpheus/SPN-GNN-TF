import time
import cProfile
from src.spn_gnn_performance.tf_dataset import load_dataset
from src.spn_gnn_performance.baseline_models import engineer_features, _graph_tensor_to_networkx
import numpy as np

dataset = load_dataset('tests/sample.jsonl')
graphs = list(dataset)

def _engineer_and_pad(graph):
    engineered_features = engineer_features(graph)
    label = graph.node_sets['node']['label']
    label_numpy = label.numpy()

    num_nodes = engineered_features.shape[0]
    pad_width = 100 - num_nodes

    padded_features = np.pad(
        engineered_features,
        ((0, pad_width), (0, 0)),
        'constant',
        constant_values=-1
    ).astype(np.float32)

    padded_label = np.pad(
        label_numpy,
        ((0, pad_width), (0, 0)),
        'constant',
        constant_values=-1
    ).astype(np.float32)

    return padded_features, padded_label

def run_all():
    for graph in graphs:
        _engineer_and_pad(graph)

cProfile.run('run_all()', sort='cumtime')
