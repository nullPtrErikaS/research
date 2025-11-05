"""Dimensionality reduction helpers (PCA / TruncatedSVD).

Contains `run_pca` extracted from the legacy `parse.py` to start modularization.
"""
from pathlib import Path
import pickle
import numpy as np

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.decomposition import PCA as SKPCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def _ensure_dense_for_embedding(X, n_components_for_init=50):
    """Local helper to mirror parse._ensure_dense_for_embedding behavior."""
    if X is None:
        return None
    if hasattr(X, 'toarray') and not isinstance(X, np.ndarray):
        try:
            svd = TruncatedSVD(n_components=min(n_components_for_init, X.shape[1]-1), random_state=42)
            Xred = svd.fit_transform(X)
            return Xred
        except Exception:
            try:
                return X.toarray()
            except Exception:
                return None
    if isinstance(X, np.ndarray):
        return X
    try:
        return np.array(X)
    except Exception:
        return None


def run_pca(X, n_components=2, output_dir='artifacts'):
    """Run dimensionality reduction. Use TruncatedSVD if input is sparse.

    Returns coordinates (n_samples, n_components) as numpy array.
    """
    if X is None:
        print("No matrix provided to run_pca")
        return None

    Path(output_dir).mkdir(exist_ok=True)

    try:
        # prefer TruncatedSVD for sparse matrices
        if hasattr(X, 'toarray') and not isinstance(X, np.ndarray):
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            coords = svd.fit_transform(X)
            # save model
            with open(f'{output_dir}/svd_model.pkl', 'wb') as f:
                pickle.dump(svd, f)
        else:
            pca = SKPCA(n_components=n_components, random_state=42)
            coords = pca.fit_transform(X)
            with open(f'{output_dir}/pca_model.pkl', 'wb') as f:
                pickle.dump(pca, f)
    except Exception as e:
        print(f"Dimensionality reduction failed: {e}")
        return None

    # coords -> save
    np.save(f'{output_dir}/coords.npy', coords)
    print(f"Saved coords shape={coords.shape} to {output_dir}")
    return coords
