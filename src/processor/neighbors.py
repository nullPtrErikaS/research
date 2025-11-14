"""Nearest-neighbor utilities extracted from `parse.py`."""
from pathlib import Path
import numpy as np

try:
    from sklearn.neighbors import NearestNeighbors
    NEIGHBORS_AVAILABLE = True
except Exception:
    NEIGHBORS_AVAILABLE = False


def _ensure_dense_for_embedding(X, n_components_for_init=50):
    # lightweight copy of parse._ensure_dense_for_embedding behaviour
    import numpy as _np
    try:
        if X is None:
            return None
        if hasattr(X, 'toarray') and not isinstance(X, _np.ndarray):
            try:
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=min(n_components_for_init, X.shape[1]-1), random_state=42)
                Xred = svd.fit_transform(X)
                return Xred
            except Exception:
                try:
                    return X.toarray()
                except Exception:
                    return None
        if isinstance(X, _np.ndarray):
            return X
        return _np.array(X)
    except Exception:
        return None


def compute_neighbors(X_or_coords, n_neighbors=10, output_dir='artifacts', metric='euclidean', algorithm='auto'):
    """Compute nearest neighbors on provided data (dense or coords) and save indices/distances.

    Parameters
    ----------
    X_or_coords : array-like or sparse matrix
        Input data to compute neighbors on.
    n_neighbors : int
        Number of neighbors to compute.
    output_dir : str
        Directory where nn_indices.npy and nn_distances.npy will be saved.
    metric : str or callable, default 'euclidean'
        Distance metric to pass to sklearn.neighbors.NearestNeighbors (e.g. 'euclidean' or 'cosine').
    algorithm : str, default 'auto'
        Algorithm parameter passed through to NearestNeighbors.

    Returns
    -------
    indices, distances
        Neighbor indices and distances arrays (same as sklearn API).
    """
    if not NEIGHBORS_AVAILABLE:
        print("sklearn NearestNeighbors not available install scikit-learn")
        return None, None

    if X_or_coords is None:
        print("No input provided to compute_neighbors")
        return None, None

    Xdense = _ensure_dense_for_embedding(X_or_coords, n_components_for_init=50)
    if Xdense is None:
        print("Unable to prepare input for nearest-neighbors")
        return None, None

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, Xdense.shape[0]-1), metric=metric, algorithm=algorithm)
    nn.fit(Xdense)
    distances, indices = nn.kneighbors(Xdense)

    Path(output_dir).mkdir(exist_ok=True)
    # Save canonical filenames for backward compatibility
    np.save(f'{output_dir}/nn_indices.npy', indices)
    np.save(f'{output_dir}/nn_distances.npy', distances)
    # Also save metric-tagged files so it's obvious which metric was used
    safe_metric = str(metric).replace(' ', '_')
    try:
        np.save(f'{output_dir}/nn_indices_{safe_metric}.npy', indices)
        np.save(f'{output_dir}/nn_distances_{safe_metric}.npy', distances)
    except Exception:
        # non-fatal: if metric cannot be used in filename, skip metric-tagged saves
        pass
    print(f"Saved nearest-neighbor indices/distances to {output_dir} (metric={metric})")
    return indices, distances
