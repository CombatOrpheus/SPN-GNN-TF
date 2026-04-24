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
## 2026-04-14 - Baseline Feature Engineering Bottleneck
**Learning:** For baseline models, feature engineering relied on converting `tfgnn.GraphTensor` to `nx.DiGraph` to use NetworkX for degree, PageRank, and local clustering coefficient calculations. Although a `dense_pagerank` optimization was added previously, `nx.clustering` and the NetworkX conversion itself still pose a severe bottleneck. Creating a NumPy adjacency matrix directly from TensorFlow GNN indices and implementing pure vectorized NumPy algorithms (e.g. matrix power for clustering) provides a ~3.3x speedup.
**Action:** When extracting graph structure features for baseline models, completely avoid NetworkX conversion. Use pure NumPy vectorization built directly from GraphTensor edge indices instead.

## 2023-10-27 - TF Dataset disk caching and truncation errors
**Learning:** Adding `.cache(filename)` right after `tf.data.Dataset.from_generator` prevents massive overhead from re-executing pure Python loops/parsers per epoch, generating a 5x parsing speedup. However, if standard dataset operations like `.take(k)` or `.skip(n)` (used inside `split_dataset`) partially iterate a dataset without fully realizing it, the disk cache will be aggressively discarded/truncated by TensorFlow, and subsequent operations like `.reduce()` will throw `InvalidArgumentError: Type mismatch: actual variant vs. expect int32`.
**Action:** When adding disk caching to a TF Dataset pipeline with `.cache(filename)` and `from_generator`, forcefully populate the cache completely right away by iterating over the entire dataset via `for _ in dataset: pass` if the `<filename>.cache.index` file does not exist, prior to applying operations like `.split` or `.take`.
## 2026-04-17 - NumPy Performance Anti-Patterns in Iterative Graph Algorithms
**Learning:** Using Python's built-in `sum()` function on a NumPy array (e.g. `sum(x[dangling])`) inside an iterative loop is an extreme performance anti-pattern. It bypasses C-level vectorization and forces element-by-element iteration in Python. Additionally, `np.linalg.matrix_power(A, 3)` calculates the full dense $A^3$ matrix in $O(N^3)$ operations, which is overkill if only the diagonal is needed (e.g. for counting triangles in the local clustering coefficient).
**Action:** Always use NumPy's methods (e.g. `array.sum()`) instead of built-ins. When calculating the diagonal of $A^3$, use `A2 = np.dot(A, A)` and `triangles = np.einsum('ij,ji->i', A, A2) / 2.0` to perform a single matrix multiplication and an $O(N^2)$ dot product, saving massive amounts of compute. Always hoist loop-invariant constants to avoid redundant math.
## 2024-05-14 - [Dataset Cardinality Read Bottleneck]
**Learning:** Python's built-in generator iteration (`sum(1 for _ in f)`) on large JSON-L dataset files in `tf_dataset.py` is extremely slow and blocks the data pipeline initialization before TF cardinality assertion. Reading the file in binary chunks and using C-optimized `bytes.count()` (`sum(buf.count(b'\n') for buf in iter(lambda: f.read(1024 * 1024), b''))`) reduces reading time by ~8x on massive files.
**Action:** Always prefer chunked binary reads when doing quick exploratory scans or line counting of large unstructured data files before passing them to `tf.data.Dataset`.

## 2024-05-14 - [NumPy Vectorized Summation]
**Learning:** Using Python's native `sum()` function on a NumPy array slice (like `sum(x[dangling_weights])`) inside an iterative loop (like PageRank) forces Python to iterate the array element-by-element instead of utilizing NumPy's C-level vectorization, causing severe slowdowns.
**Action:** Always call the native `.sum()` method on NumPy array objects (e.g., `x[dangling_weights].sum()`) to ensure optimal vectorization.
