## 2026-05-18 - NumPy einsum vs array sum in iterative math
**Learning:** Using `np.einsum('ij,ij->i', A_undir, A2)` for extracting the diagonal sum from the element-wise multiplication of two symmetric matrices is roughly 10% faster than computing the full matrix `(A_undir * A2)` and then calling `.sum(axis=1)`. The latter approach forces an expensive memory allocation of a large intermediate matrix before the summation step.
**Action:** Use `np.einsum` when calculating partial sums of element-wise operations on large matrices to minimize memory allocation.

## 2026-05-18 - Fast Array Padding in TF map Functions
**Learning:** Calling `np.pad` on small arrays inside a tight inner loop (e.g. `tf.py_function` map of a dataset) performs very poorly because `np.pad` contains significant dynamic type checking and multiple sub-allocations. Instead, pre-allocating an array with `np.empty((max_nodes, num_feats))` and using a slice assignment `padded_features[:num_nodes] = features` offers a ~16x speedup.
**Action:** When padding many small, similarly-shaped arrays dynamically inside an iterator loop, avoid `np.pad`. Use `np.empty` and manually populate it via slicing instead for massive performance gains.

## 2026-05-18 - Tensor Slicing Overhead in TF/Numpy interaction
**Learning:** Slicing a TensorFlow tensor *before* calling `.numpy()` (e.g., `sizes[0].numpy()`) invokes the TensorFlow graph execution engine for the slice operation, introducing significant overhead. Conversely, calling `.numpy()` to materialize the array first and then indexing purely in Python/NumPy (e.g., `sizes.numpy()[0]`) is substantially faster (over 30x faster for single-element extractions).
**Action:** Always extract the numpy array from a `tf.Tensor` before performing standard Python slicing or indexing when inside custom numpy feature extraction functions.
