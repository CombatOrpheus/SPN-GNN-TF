import time
from src.spn_gnn_performance.tf_dataset import load_dataset
from src.spn_gnn_performance.baseline_models import engineer_features, _graph_tensor_to_networkx
import networkx as nx
import numpy as np

dataset = load_dataset('tests/sample.jsonl')
graphs = list(dataset)

def _numpy_pagerank(g, alpha=0.85, max_iter=100, tol=1e-06):
    n = len(g)
    if n == 0:
        return np.array([])

    M = nx.to_numpy_array(g)
    out_degree = M.sum(axis=1)
    dangling_weights = (out_degree == 0).astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        M = M / out_degree[:, np.newaxis]
    M[np.isnan(M)] = 0
    M = M.T

    x = np.ones(n) / n
    p = np.ones(n) / n

    for _ in range(max_iter):
        xlast = x
        dangling_sum = np.dot(dangling_weights, x)
        x = alpha * (np.dot(M, x) + dangling_sum * p) + (1 - alpha) * p
        err = np.sum(np.abs(x - xlast))
        if err < n * tol:
            break

    return x

def my_engineer_features(graph):
    original_features = graph.node_sets["node"]["hidden_state"].numpy()

    # Create networkx graph ONCE
    g = _graph_tensor_to_networkx(graph)

    # Degree features
    in_degree = np.array([d for _, d in g.in_degree()])
    out_degree = np.array([d for _, d in g.out_degree()])
    degree_features = np.stack([in_degree, out_degree], axis=1)

    # PageRank features (NumPy)
    pagerank_features = _numpy_pagerank(g)

    # Clustering features
    clustering = nx.clustering(g.to_undirected())
    clustering_features = np.array([clustering.get(i, 0.0) for i in range(len(g.nodes))])

    # Reshape centrality and clustering features to be 2D arrays.
    pagerank_features = np.expand_dims(pagerank_features, axis=1)
    clustering_features = np.expand_dims(clustering_features, axis=1)

    return np.hstack([
        original_features,
        degree_features,
        pagerank_features,
        clustering_features
    ])

start_time = time.time()
for graph in graphs:
    engineer_features(graph)
print('engineer_features:', time.time() - start_time)

start_time = time.time()
for graph in graphs:
    my_engineer_features(graph)
print('my_engineer_features:', time.time() - start_time)
