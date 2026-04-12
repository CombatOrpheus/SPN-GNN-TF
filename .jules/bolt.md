## 2024-04-08 - TensorFlow Dataset Cardinality Bottleneck
**Learning:** `tf.data.Dataset.from_generator` returns a dataset with `tf.data.experimental.UNKNOWN_CARDINALITY`. The `split_dataset` method handled this fallback by running `len(list(dataset))`, which consumes the entire generator simply to count items, taking over a minute for moderately sized files and defeating the purpose of a lazy dataset.
**Action:** When loading datasets from text files with `from_generator`, count the lines natively in Python first, and then explicitly assert the cardinality on the dataset using `ds.apply(tf.data.experimental.assert_cardinality(num_lines))`. This removes the `UNKNOWN_CARDINALITY` and lets splitting logic run instantaneously without exhausting the iterator.
## 2026-04-10 - NetworkX PageRank Performance
**Learning:** `nx.pagerank` relies on SciPy sparse matrix operations, which introduces significant overhead for many small graphs (e.g., ~30 nodes).
**Action:** Implemented a `dense_pagerank` using NumPy power iteration which avoids sparse matrix overhead and drastically improves baseline feature extraction speed when caching networkx conversion. Always consider dense ops over sparse when the graphs are small and numerous.
## 2026-04-11 - Optimize max_nodes calculation in dataset preparation
**Learning:** In TensorFlow data pipelines, iterating over a `tf.data.Dataset` in standard Python with a `for` loop and calling `.numpy()` on each element is extremely slow.
**Action:** When finding a maximum value across a dataset (e.g., `max_nodes`), use `dataset.reduce` with TensorFlow operations to calculate it scalably and avoid the Python/TensorFlow boundary overhead.

## 2026-04-12 - TensorFlow Dataset Prefetching Performance
**Learning:** tf.data.Dataset pipelines block the main process and training loop if `.prefetch()` is missing, severely affecting performance when generating, loading, or preprocessing batches.
**Action:** Always append `.prefetch(tf.data.AUTOTUNE)` to the end of any tf.data.Dataset pipeline, especially after `.batch()` and `.map()`, to decouple data production (CPU) from consumption (GPU/training logic) and eliminate dataset-bound bottlenecks.
