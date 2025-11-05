"""Clustering utilities (kmeans / hdbscan) extracted from `parse.py`."""
from pathlib import Path
import numpy as np

try:
    from sklearn.cluster import KMeans
    CLUSTERING_AVAILABLE = True
except Exception:
    CLUSTERING_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False


def _ensure_dense_for_embedding(X, n_components_for_init=50):
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


def run_clustering(df, X_or_coords, config=None, output_dir='artifacts'):
    """Run clustering, add labels to df, and save results.

    Supports 'kmeans' (sklearn) and 'hdbscan' (if installed).
    Returns labels (numpy array) or None.
    """
    cfg = config or {'method': 'kmeans', 'n_clusters': 8}

    if X_or_coords is None:
        print("No input provided to run_clustering")
        return None

    Xdense = _ensure_dense_for_embedding(X_or_coords, n_components_for_init=50)
    if Xdense is None:
        print("Unable to prepare input for clustering")
        return None

    method = cfg.get('method', 'kmeans')
    labels = None

    if method == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            print("hdbscan not available; falling back to kmeans")
            method = 'kmeans'
        else:
            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg.get('min_cluster_size', 5))
                labels = clusterer.fit_predict(Xdense)
            except Exception as e:
                print(f"hdbscan failed: {e}")
                labels = None

    if method == 'kmeans':
        if not CLUSTERING_AVAILABLE:
            print("sklearn KMeans not available install scikit-learn")
            return None
        n_clusters = cfg.get('n_clusters', 8)
        try:
            km = KMeans(n_clusters=n_clusters, random_state=42)
            labels = km.fit_predict(Xdense)
        except Exception as e:
            print(f"KMeans failed: {e}")
            labels = None

    if labels is not None:
        try:
            df['cluster'] = labels.tolist()
        except Exception:
            df['cluster'] = list(labels)

        Path(output_dir).mkdir(exist_ok=True)
        np.save(f'{output_dir}/cluster_labels.npy', labels)
        try:
            df.to_csv(f'{output_dir}/processed_data_with_clusters.csv', index=False)
        except Exception:
            pass

        print(f"Saved cluster labels (method={method}) to {output_dir}/cluster_labels.npy")
    else:
        print("No cluster labels produced")

    return labels
